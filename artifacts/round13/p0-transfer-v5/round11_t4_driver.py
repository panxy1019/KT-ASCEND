#!/usr/bin/env python3
"""Run one Round11 static/dynamic streaming Decode economics process."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import urllib.request


LENGTHS = (32, 64, 128, 256)
MIGRATION_PHASES = {
    "route_statistics",
    "consumer_drain_sync",
    "state_fence_before",
    "selection",
    "migration_copy_and_sync",
    "mapping_publication",
    "state_fence_after",
    "cleanup",
    "total_exposed",
}


def post(endpoint, body, timeout=1800):
    request = urllib.request.Request(
        "http://127.0.0.1:30091" + endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"unterminated evidence file: {path.name}")
    return [json.loads(line) for line in raw.splitlines()]


def wait_dynamic_events(
    path,
    start,
    request_id,
    *,
    timeout=30.0,
    poll=0.05,
    read_fn=read_jsonl,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Wait only for the asynchronous request-drain receipt to reach JSONL."""
    deadline = clock() + timeout
    required = {
        "PREFILL_END_COMMIT",
        "ROUND11_DYNAMIC_TIMING",
        "PREFILL_END_REQUEST_DRAINED",
    }
    while True:
        try:
            rows = read_fn(path)[start:]
        except RuntimeError as error:
            if "unterminated evidence file" not in str(error):
                raise
            rows = []
        related = [row for row in rows if row.get("request_id") == request_id]
        counts = {status: sum(row.get("status") == status for row in related) for status in required}
        if all(counts[status] == 1 for status in required):
            return rows
        if any(counts[status] > 1 for status in required):
            raise RuntimeError("Dynamic request emitted duplicate lifecycle evidence")
        now = clock()
        if now >= deadline:
            raise RuntimeError(f"timed out waiting for Dynamic lifecycle closure: {counts}")
        sleep(min(poll, max(0.0, deadline - now)))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build_schedule(fixture, *, mode, block):
    if mode not in {"STATIC", "DYNAMIC"} or block not in {"A", "B"}:
        raise ValueError("invalid T4 process role")
    ids = fixture.get("cases", {}).get("edge64", {}).get("input_ids")
    if not isinstance(ids, list) or len(ids) != 64 or any(type(value) is not int for value in ids):
        raise ValueError("T4 requires the frozen exact 64-token fixture")
    values = (("warmup", 32),) + tuple(("measurement", length) for length in LENGTHS)
    rows = []
    for index, (phase, length) in enumerate(values, 1):
        rid = f"r11-t4-{mode.lower()}-{block.lower()}-{index:02d}-decode{length}-{phase}"
        rows.append(
            {
                "index": index,
                "phase": phase,
                "decode_tokens": length,
                "request": {
                    "rid": rid,
                    "input_ids": ids,
                    "stream": True,
                    "sampling_params": {
                        "temperature": 0,
                        "top_p": 1,
                        "sampling_seed": 20260920,
                        "max_new_tokens": length,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": -1,
                },
            }
        )
    return rows


def consume_stream(response, expected_tokens, *, clock=time.monotonic_ns):
    output_ids = []
    arrivals = []
    final_meta = None
    saw_done = False
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            saw_done = True
            break
        value = json.loads(payload)
        current = value.get("output_ids")
        if not isinstance(current, list) or any(type(token) is not int for token in current):
            raise RuntimeError("stream response has invalid output_ids")
        if current[: len(output_ids)] != output_ids:
            raise RuntimeError("stream output_ids are not a cumulative prefix")
        delta = len(current) - len(output_ids)
        if delta not in {0, 1}:
            raise RuntimeError("stream timing requires exactly one token per growth event")
        if delta == 1:
            output_ids = list(current)
            arrivals.append(int(clock()))
        final_meta = value.get("meta_info", final_meta)
    if len(output_ids) != expected_tokens or len(arrivals) != expected_tokens:
        raise RuntimeError("stream did not produce the exact Decode token budget")
    if not isinstance(final_meta, dict):
        raise RuntimeError("stream terminal metadata missing")
    logprobs = final_meta.get("output_token_logprobs")
    if (
        not isinstance(logprobs, list)
        or len(logprobs) != expected_tokens
        or any(
            not isinstance(row, (list, tuple))
            or not row
            or not isinstance(row[0], (int, float))
            or not math.isfinite(row[0])
            for row in logprobs
        )
    ):
        raise RuntimeError("stream output logprobs are missing or nonfinite")
    reason = final_meta.get("finish_reason")
    reason_type = reason.get("type") if isinstance(reason, dict) else reason
    if reason_type != "length" or final_meta.get("completion_tokens") != expected_tokens:
        raise RuntimeError("stream did not terminate at the requested length")
    return {
        "output_ids": output_ids,
        "token_arrival_monotonic_ns": arrivals,
        "finite_logprobs": True,
        "finish_reason": reason_type,
        "completion_tokens": expected_tokens,
        "saw_done_marker": saw_done,
    }


def validate_dynamic_events(request_id, events):
    related = [row for row in events if row.get("request_id") == request_id]
    commits = [row for row in related if row.get("status") == "PREFILL_END_COMMIT"]
    timings = [row for row in related if row.get("status") == "ROUND11_DYNAMIC_TIMING"]
    drains = [row for row in related if row.get("status") == "PREFILL_END_REQUEST_DRAINED"]
    if len(commits) != 1 or len(timings) != 1 or len(drains) != 1:
        raise RuntimeError("Dynamic request requires exactly one commit, timing receipt, and drain")
    commit = commits[0]
    timing = timings[0].get("timing_ns", {})
    if (
        commit.get("live_state_preserved") is not True
        or type(commit.get("changed_layers")) is not int
        or type(commit.get("copied_experts")) is not int
        or set(timing) != MIGRATION_PHASES
        or any(type(timing[name]) is not int or timing[name] < 0 for name in MIGRATION_PHASES)
    ):
        raise RuntimeError("Dynamic timing/transaction identity mismatch")
    child = sum(value for name, value in timing.items() if name != "total_exposed")
    if child > timing["total_exposed"] * 1.02 + 5_000_000:
        raise RuntimeError("Dynamic exclusive phases exceed exposed parent")
    return {
        "status": "PREFILL_END_COMMIT",
        "live_state_preserved": True,
        "changed_layers": commit["changed_layers"],
        "copied_experts": commit["copied_experts"],
        "generation": commit.get("generation"),
        "timing_ns": timing,
        "drain_generation": drains[0].get("generation"),
    }


def validate_hybrid_path(request_id, hybrid_events, layerwise_events):
    related = [row for row in hybrid_events if row.get("request_id") == request_id]
    selections = [row for row in related if row.get("stage") == "SELECTION"]
    stages = {row.get("stage") for row in related}
    if (
        not selections
        or any(row.get("path") != "HYBRID" or row.get("basis") != "REQUEST_TOTAL" for row in selections)
        or not {"CPU_SUBMIT", "CPU_RESULT", "NPU_RESIDENT_APPLY"}.issubset(stages)
        or any(row.get("request_id") == request_id for row in layerwise_events)
    ):
        raise RuntimeError("T4 request did not use the complete Hybrid path")
    return {
        "selection_count": len(selections),
        "dispatch_tokens": [row.get("batch_tokens") for row in selections],
        "cpu_submit_count": sum(row.get("stage") == "CPU_SUBMIT" for row in related),
        "cpu_result_count": sum(row.get("stage") == "CPU_RESULT" for row in related),
        "npu_resident_count": sum(row.get("stage") == "NPU_RESIDENT_APPLY" for row in related),
    }


def run(mode, block, out, fixture_path, parent_identity, *, post_fn=post):
    out = Path(out)
    result_path = out / "round11-t4-process.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    fixture = json.loads(Path(fixture_path).read_text())
    schedule = build_schedule(fixture, mode=mode, block=block)
    report = {
        "schema": 1,
        "status": "RUNNING",
        "mode": mode,
        "block": block,
        "parent_identity": parent_identity,
        "profiler": "OFF",
        "generation_budget": 5,
        "generation_attempts": 0,
        "automatic_retries": 0,
        "warmup": None,
        "results": [],
    }
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            rid = row["request"]["rid"]
            dynamic_before = len(read_jsonl(out / "dynamic-events.jsonl"))
            started = time.monotonic_ns()
            with post_fn("/generate", row["request"], 1800) as response:
                stream = consume_stream(response, row["decode_tokens"])
            ended = time.monotonic_ns()
            hybrid = validate_hybrid_path(
                rid,
                read_jsonl(out / "hybrid-path.jsonl"),
                read_jsonl(out / "layerwise-events.jsonl"),
            )
            dynamic_rows = (
                wait_dynamic_events(out / "dynamic-events.jsonl", dynamic_before, rid)
                if mode == "DYNAMIC"
                else read_jsonl(out / "dynamic-events.jsonl")[dynamic_before:]
            )
            value = {
                "decode_tokens": row["decode_tokens"],
                "input_tokens": 64,
                **stream,
                "path": "HYBRID",
                "path_evidence": hybrid,
                "e2e_seconds": (ended - started) / 1e9,
            }
            if mode == "DYNAMIC":
                value["migration"] = validate_dynamic_events(rid, dynamic_rows)
            elif any(item.get("request_id") == rid for item in dynamic_rows):
                raise RuntimeError("Static process emitted request-bound Dynamic evidence")
            if row["phase"] == "warmup":
                report["warmup"] = value
            else:
                report["results"].append(value)
            save(result_path, report)
        if report["generation_attempts"] != 5 or len(report["results"]) != 4 or report["warmup"] is None:
            raise RuntimeError("T4 process schedule incomplete")
        report["status"] = "PASS_ROUND11_T4_DRIVER_RAW"
        save(result_path, report)
        return report
    except BaseException as error:
        report["status"] = "FAIL_ROUND11_T4_DRIVER"
        report["error"] = f"{type(error).__name__}: {error}"
        save(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("STATIC", "DYNAMIC"), required=True)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--parent-identity", required=True)
    args = parser.parse_args()
    result = run(args.mode, args.block, args.out, args.fixture, args.parent_identity)
    print(result["status"], args.mode, args.block, flush=True)


if __name__ == "__main__":
    main()
