"""Cancel after a chosen streaming-prefill chunk begins and preserve the trace."""

import argparse
import json
import pathlib
import threading
import time
import urllib.request


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def request(path, body, timeout):
    value = urllib.request.Request(
        "http://127.0.0.1:30091" + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(value, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--streaming-events", type=pathlib.Path,
                        default=pathlib.Path("/out/streaming-events.jsonl"))
    parser.add_argument("--cancel-events", type=pathlib.Path,
                        default=pathlib.Path("/out/cancel-events.jsonl"))
    parser.add_argument("--trigger-start", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.trigger_start < 0:
        raise ValueError("trigger start must be nonnegative")

    rid = "r9-p0-midprefill-cancel"
    prompt = (
        "Summarize this list in one short sentence: "
        + " ".join(f"item{index}" for index in range(1, 61))
    )
    report = {
        "schema": 1,
        "status": "RUNNING",
        "scope": "P0_MIDPREFILL_CANCEL_TRACE",
        "request_id": rid,
        "trigger_start": args.trigger_start,
    }
    initial_streaming = len(read_jsonl(args.streaming_events))
    initial_cancel = len(read_jsonl(args.cancel_events))
    response_events = []
    generation_error = []

    def generate():
        body = {
            "rid": rid,
            "text": prompt,
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 8,
                "ignore_eos": False,
            },
        }
        try:
            with request("/generate", body, args.timeout) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    response_events.append(json.loads(payload))
        except BaseException as error:  # evidence must retain transport failures
            generation_error.append(repr(error))

    worker = threading.Thread(target=generate, name="round9-cancel-generate")
    worker.start()
    deadline = time.monotonic() + args.timeout
    trigger = None
    while time.monotonic() < deadline:
        events = read_jsonl(args.streaming_events)[initial_streaming:]
        trigger = next(
            (event for event in events
             if event.get("request_id") == rid
             and event.get("status") == "STREAMING_CHUNK_BEGIN"
             and event.get("start") == args.trigger_start),
            None,
        )
        if trigger is not None:
            break
        if not worker.is_alive():
            break
        time.sleep(0.05)
    if trigger is None:
        report["status"] = "FAIL_TRIGGER_NOT_OBSERVED"
    else:
        report["trigger_event"] = trigger
        report["streaming_events_at_abort_send"] = len(
            read_jsonl(args.streaming_events)[initial_streaming:])
        try:
            with request(
                "/abort_request", {"rid": rid, "abort_all": False}, 30
            ) as response:
                report["abort_http_status"] = response.status
                report["abort_http_body"] = response.read().decode()
        except BaseException as error:
            report["abort_error"] = repr(error)

    remaining = max(0.0, deadline - time.monotonic())
    worker.join(remaining)
    report["generation_thread_alive"] = worker.is_alive()
    report["generation_errors"] = generation_error
    report["response_events"] = response_events
    report["streaming_events"] = [
        event for event in read_jsonl(args.streaming_events)[initial_streaming:]
        if event.get("request_id") == rid
    ]
    report["cancel_events"] = read_jsonl(args.cancel_events)[initial_cancel:]
    statuses = [event.get("status") for event in report["cancel_events"]]
    stream_statuses = [event.get("status") for event in report["streaming_events"]]
    later_chunks = [
        event for event in report["streaming_events"]
        if event.get("status") == "STREAMING_CHUNK_BEGIN"
        and event.get("start", -1) > args.trigger_start
    ]
    terminal_abort = any(
        isinstance(event.get("meta_info", {}).get("finish_reason"), dict)
        and event["meta_info"]["finish_reason"].get("type") == "abort"
        for event in response_events
    )
    if report["status"] == "RUNNING":
        if "CANCEL_SEEN" not in statuses:
            report["status"] = "FAIL_CANCEL_NOT_SEEN"
        elif worker.is_alive():
            report["status"] = "FAIL_TERMINAL_RESPONSE_TIMEOUT"
        elif "CANCEL_CLEANUP_COMMITTED" not in statuses:
            report["status"] = "FAIL_CLEANUP_NOT_COMMITTED"
        elif later_chunks:
            report["status"] = "FAIL_POST_CANCEL_CHUNK"
        elif "STREAMING_ABORTED" not in stream_statuses or not terminal_abort:
            report["status"] = "FAIL_TERMINAL_ABORT_MISSING"
        else:
            reuse = {
                "rid": "r9-p0-post-cancel-reuse",
                "text": "Answer with exactly OK.",
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            }
            try:
                with request("/generate", reuse, args.timeout) as response:
                    report["reuse_response"] = json.load(response)
                report["status"] = "PASS_MIDPREFILL_CANCEL_AND_REUSE"
            except BaseException as error:
                report["reuse_error"] = repr(error)
                report["status"] = "FAIL_POST_CANCEL_REUSE"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["status"], flush=True)
    if report["status"] != "PASS_MIDPREFILL_CANCEL_AND_REUSE":
        raise RuntimeError(report["status"])


if __name__ == "__main__":
    main()
