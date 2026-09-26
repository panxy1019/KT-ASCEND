"""Run one bounded Round11 T1 model-process schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import threading
import time
import urllib.request

from round11_t1_protocol import (
    THRESHOLDS,
    build_schedule,
    validate_cancel_receipt,
    validate_fixture,
    validate_path,
    validate_response,
)


def read_jsonl(path):
    path = pathlib.Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"unterminated evidence file: {path.name}")
    try:
        return [json.loads(line) for line in raw.splitlines()]
    except json.JSONDecodeError as error:
        raise RuntimeError(f"malformed evidence file: {path.name}") from error


def post(endpoint, body, timeout=1800):
    request = urllib.request.Request(
        "http://127.0.0.1:30091" + endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def save(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _finish_type(value):
    return value.get("type") if isinstance(value, dict) else value


def cancel_summary(
    row,
    response_events,
    hybrid_events,
    cancel_events,
    *,
    abort_http_status,
    thread_alive,
):
    rid = row["request"]["rid"]
    related_cancel = [
        item
        for item in cancel_events
        if item.get("request_id") == rid or item.get("requested_id") == rid
    ]
    statuses = [item.get("status") for item in related_cancel]
    seen = next(
        (
            item.get("event_seq")
            for item in related_cancel
            if item.get("status") == "CANCEL_SEEN"
        ),
        None,
    )
    late_commit = seen is None or any(
        item.get("event_seq", -1) > seen
        and item.get("request_id") == rid
        and item.get("status") == "NORMAL_SUBMIT_ADMITTED"
        for item in related_cancel
    )
    finish_reasons = [
        _finish_type(item.get("meta_info", {}).get("finish_reason"))
        for item in response_events
        if isinstance(item, dict)
    ]
    return {
        "status": "PENDING_REUSE_VALIDATION",
        "request_id": rid,
        "trigger_start": 32,
        "abort_http_status": abort_http_status,
        "generation_thread_alive": thread_alive,
        "terminal_finish_reason": "abort" if "abort" in finish_reasons else None,
        "hybrid_events": [
            item for item in hybrid_events if item.get("request_id") == rid
        ],
        "cancel_events": related_cancel,
        "cancel_statuses": statuses,
        "late_commit": late_commit,
        "lease_returned": "REQUEST_RETIRED" in statuses,
        "response_events": response_events,
        "reuse": None,
    }


def run_cancel(row, out, *, post_fn=post, timeout=1800):
    hybrid_path = out / "hybrid-path.jsonl"
    cancel_path = out / "cancel-events.jsonl"
    hybrid_before = len(read_jsonl(hybrid_path))
    cancel_before = len(read_jsonl(cancel_path))
    response_events = []
    generation_errors = []

    def generate():
        try:
            with post_fn("/generate", row["request"], timeout) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    response_events.append(json.loads(payload))
        except BaseException as error:
            generation_errors.append(repr(error))

    worker = threading.Thread(target=generate, name="round11-t1-cancel-generate")
    worker.start()
    deadline = time.monotonic() + timeout
    trigger = None
    while time.monotonic() < deadline:
        rows = read_jsonl(hybrid_path)[hybrid_before:]
        trigger = next(
            (
                item
                for item in rows
                if item.get("request_id") == row["request"]["rid"]
                and item.get("stage") == "SELECTION"
                and item.get("start") == 32
            ),
            None,
        )
        if trigger is not None or not worker.is_alive():
            break
        time.sleep(0.05)
    abort_status = None
    abort_body = None
    if trigger is not None:
        with post_fn(
            "/abort_request",
            {"rid": row["request"]["rid"], "abort_all": False},
            30,
        ) as response:
            abort_status = response.status
            abort_body = response.read().decode()
    worker.join(max(0.0, deadline - time.monotonic()))
    while time.monotonic() < deadline:
        new_cancel = read_jsonl(cancel_path)[cancel_before:]
        if any(
            item.get("request_id") == row["request"]["rid"]
            and item.get("status") == "CANCEL_CLEANUP_COMMITTED"
            for item in new_cancel
        ):
            break
        if worker.is_alive():
            break
        time.sleep(0.05)
    summary = cancel_summary(
        row,
        response_events,
        read_jsonl(hybrid_path)[hybrid_before:],
        read_jsonl(cancel_path)[cancel_before:],
        abort_http_status=abort_status,
        thread_alive=worker.is_alive(),
    )
    summary.update(
        trigger_event=trigger,
        abort_http_body=abort_body,
        generation_errors=generation_errors,
    )
    if trigger is None:
        raise RuntimeError("Hybrid cancel trigger chunk was not observed")
    if generation_errors:
        raise RuntimeError(f"Hybrid cancel generation failed: {generation_errors}")
    return summary


def run(process, out, fixture_path, *, post_fn=post):
    out = pathlib.Path(out)
    result_path = out / "round11-t1-results.json"
    if result_path.exists() or list(out.glob("round11-t1-cancel-receipt-*.json")):
        raise FileExistsError("Round11 T1 output evidence must be fresh")
    fixture_path = pathlib.Path(fixture_path)
    fixture = json.loads(fixture_path.read_text())
    validate_fixture(fixture)
    schedule = build_schedule(fixture, process)
    report = {
        "schema": 1,
        "status": "RUNNING",
        "process": process,
        "baseline": "HFINAL_V107",
        "experimental_profile": process != "BOUNDARY",
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "threshold_tokens": THRESHOLDS[process],
        "chunk_tokens": 32,
        "selection_basis": "REQUEST_TOTAL",
        "generation_budget": len(schedule),
        "generation_attempts": 0,
        "abort_controls": 0,
        "automatic_retries": 0,
        "profiler": "OFF",
        "schedule": schedule,
        "results": [],
        "cancel_receipts": [],
        "started_wall": time.time(),
    }
    pending_cancel = None
    cancel_ordinal = 0
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            started_ns = time.monotonic_ns()
            started_wall = time.time()
            if row["phase"] == "cancel":
                cancel_ordinal += 1
                report["abort_controls"] += 1
                pending_cancel = run_cancel(row, out, post_fn=post_fn)
                result = {
                    "index": row["index"],
                    "case_id": row["case_id"],
                    "phase": row["phase"],
                    "request_id": row["request"]["rid"],
                    "started_wall": started_wall,
                    "ended_wall": time.time(),
                    "e2e_seconds": (time.monotonic_ns() - started_ns) / 1e9,
                    "cancel_status": pending_cancel["status"],
                    "cancel_ordinal": cancel_ordinal,
                }
                report["results"].append(result)
                save(result_path, report)
                continue

            with post_fn("/generate", row["request"], 1800) as response:
                value = json.load(response)
            result = {
                "index": row["index"],
                "case_id": row["case_id"],
                "phase": row["phase"],
                "request_id": row["request"]["rid"],
                "started_wall": started_wall,
                "ended_wall": time.time(),
                "e2e_seconds": (time.monotonic_ns() - started_ns) / 1e9,
                "response": value,
                "state": "RAW_RESPONSE_SAVED_BEFORE_VALIDATION",
            }
            report["results"].append(result)
            save(result_path, report)
            validated = validate_response(row, value)
            path = validate_path(
                row,
                read_jsonl(out / "hybrid-path.jsonl"),
                read_jsonl(out / "layerwise-events.jsonl"),
                read_jsonl(out / "streaming-events.jsonl"),
            )
            result.update(validated=validated, path=path, state="PASS")
            if row["phase"] == "reuse":
                if pending_cancel is None:
                    raise RuntimeError("reuse ran without preceding cancellation")
                pending_cancel["reuse"] = {
                    "request_id": row["request"]["rid"],
                    "validator_passed": True,
                    "output_ids": validated["output_ids"],
                }
                pending_cancel["status"] = "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"
                validate_cancel_receipt(pending_cancel)
                receipt_path = out / f"round11-t1-cancel-receipt-{cancel_ordinal:02d}.json"
                save(receipt_path, pending_cancel)
                report["cancel_receipts"].append(receipt_path.name)
                report["results"][-2]["cancel_status"] = pending_cancel["status"]
                pending_cancel = None
            save(result_path, report)

        if report["generation_attempts"] != report["generation_budget"]:
            raise RuntimeError("generation budget/schedule mismatch")
        expected_cancels = sum(row["phase"] == "cancel" for row in schedule)
        if pending_cancel is not None or len(report["cancel_receipts"]) != expected_cancels:
            raise RuntimeError("cancellation/reuse evidence missing")
        repeats = [
            item["validated"]["output_ids"]
            for item in report["results"]
            if item["case_id"] == "edge256"
            and item["phase"] in {"extended_prefill", "repeatability"}
        ]
        report["edge256_repeat_exact"] = None if not repeats else len(repeats) == 2 and repeats[0] == repeats[1]
        if process == "HYBRID256" and report["edge256_repeat_exact"] is not True:
            raise RuntimeError("fixed-seed edge256 repeatability mismatch")
        report["status"] = "PASS_ROUND11_T1_PROCESS"
        report["ended_wall"] = time.time()
        save(result_path, report)
        return report
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = repr(error)
        report["ended_wall"] = time.time()
        save(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process", choices=tuple(THRESHOLDS), required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = run(args.process, args.out, args.fixture)
    print(result["status"], args.process, result["generation_attempts"], flush=True)


if __name__ == "__main__":
    main()
