"""Run the bounded Round10 D1 Hybrid-to-Dynamic transaction schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

from round10_t1_driver import post, read_jsonl, run_cancel
from round10_t1_protocol import (
    select_path,
    validate_cancel_receipt,
    validate_fixture,
    validate_hybrid_path,
    validate_response,
)


SCHEDULE = (
    ("edge64", "warmup", 1, 4),
    ("edge64", "measurement", 2, 4),
    ("edge64", "measurement", 3, 4),
    ("cancel128", "cancel", 1, 8),
    ("canary_math391", "reuse", 3, 32),
)


def _save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build_schedule(fixture):
    validate_fixture(fixture)
    rows = []
    for index, (case_id, phase, repeat, max_new_tokens) in enumerate(SCHEDULE, 1):
        item = fixture["cases"][case_id]
        rows.append({
            "index": index,
            "case_id": case_id,
            "phase": phase,
            "repeat": repeat,
            "expected_path": select_path(total=len(item["input_ids"]), forward_mode="EXTEND"),
            "validator": item["validator"],
            "request": {
                "rid": f"r10-t4-d1-{index:02d}-{case_id}-{phase}",
                "input_ids": item["input_ids"],
                "stream": phase == "cancel",
                "sampling_params": {
                    "temperature": 0,
                    "top_p": 1,
                    "sampling_seed": 20260920,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": False,
                },
                "return_logprob": True,
                "logprob_start_len": -1,
            },
        })
    return rows


def validate_commit(request_id, events):
    related = [item for item in events if item.get("request_id") == request_id]
    commits = [item for item in related if item.get("status") == "PREFILL_END_COMMIT"]
    drained = [item for item in related if item.get("status") == "PREFILL_END_REQUEST_DRAINED"]
    if len(commits) != 1 or len(drained) != 1:
        raise RuntimeError("Dynamic success requires exactly one commit and one drain")
    event = commits[0]
    if (
        event.get("live_state_preserved") is not True
        or type(event.get("first_output_token_generation")) is not int
        or event.get("decode_generation") != event["first_output_token_generation"] + 1
        or event.get("generation") != event["decode_generation"]
        or type(event.get("changed_layers")) is not int
        or not 0 <= event["changed_layers"] <= 48
        or type(event.get("copied_experts")) is not int
        or not 0 <= event["copied_experts"] <= 48 * 32
        or len(event.get("plans", [])) != event["changed_layers"]
    ):
        raise RuntimeError("Dynamic commit transaction contract failed")
    return {"commit": event, "drain": drained[0]}


def validate_cancel_dynamic(request_id, events):
    related = [item for item in events if item.get("request_id") == request_id]
    if any(item.get("status") == "PREFILL_END_COMMIT" for item in related):
        raise RuntimeError("cancelled Dynamic request committed placement")
    drains = [item for item in related if item.get("status") == "DYNAMIC_REQUEST_ABORT_DRAINED"]
    if len(drains) != 1 or drains[0].get("changed_layers") != 0 or drains[0].get("copied_experts") != 0:
        raise RuntimeError("cancelled Dynamic request did not discard its observations")
    return drains[0]


def run(out, fixture_path, *, post_fn=post):
    out = pathlib.Path(out)
    result_path = out / "round10-t4-results.json"
    if result_path.exists():
        raise FileExistsError("T4 result must be fresh")
    fixture_path = pathlib.Path(fixture_path)
    fixture = json.loads(fixture_path.read_text())
    schedule = build_schedule(fixture)
    report = {
        "schema": 1,
        "status": "RUNNING",
        "candidate": "D1_HYBRID_PREFILL_END_DYNAMIC",
        "parent": "H0_ROUND10_V95",
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "generation_budget": len(schedule),
        "generation_attempts": 0,
        "abort_controls": 0,
        "automatic_retries": 0,
        "profiler": "OFF",
        "schedule": schedule,
        "results": [],
    }
    pending_cancel = None
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            before = len(read_jsonl(out / "dynamic-events.jsonl"))
            started = time.monotonic_ns()
            if row["phase"] == "cancel":
                report["abort_controls"] += 1
                pending_cancel = run_cancel(row, out, post_fn=post_fn)
                dynamic = read_jsonl(out / "dynamic-events.jsonl")[before:]
                abort_event = validate_cancel_dynamic(row["request"]["rid"], dynamic)
                report["results"].append({
                    "index": row["index"], "case_id": row["case_id"], "phase": row["phase"],
                    "request_id": row["request"]["rid"],
                    "e2e_seconds": (time.monotonic_ns() - started) / 1e9,
                    "cancel": pending_cancel, "dynamic_abort": abort_event,
                })
                _save(result_path, report)
                continue
            with post_fn("/generate", row["request"], 1800) as response:
                value = json.load(response)
            result = {
                "index": row["index"], "case_id": row["case_id"], "phase": row["phase"],
                "request_id": row["request"]["rid"],
                "e2e_seconds": (time.monotonic_ns() - started) / 1e9,
                "response": value, "state": "RAW_RESPONSE_SAVED_BEFORE_VALIDATION",
            }
            report["results"].append(result)
            _save(result_path, report)
            result["validated"] = validate_response(row, value)
            result["path"] = validate_hybrid_path(
                row,
                read_jsonl(out / "hybrid-path.jsonl"),
                read_jsonl(out / "layerwise-events.jsonl"),
                read_jsonl(out / "streaming-events.jsonl"),
            )
            result["dynamic"] = validate_commit(
                row["request"]["rid"], read_jsonl(out / "dynamic-events.jsonl")[before:]
            )
            result["state"] = "PASS"
            if row["phase"] == "reuse":
                if pending_cancel is None:
                    raise RuntimeError("Dynamic reuse has no preceding cancellation")
                pending_cancel["reuse"] = {
                    "request_id": row["request"]["rid"],
                    "validator_passed": True,
                    "output_ids": result["validated"]["output_ids"],
                }
                pending_cancel["status"] = "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"
                validate_cancel_receipt(pending_cancel)
                report["cancel_receipt"] = pending_cancel
                pending_cancel = None
            _save(result_path, report)
        if pending_cancel is not None or report["generation_attempts"] != 5 or report["abort_controls"] != 1:
            raise RuntimeError("T4 schedule or cancellation budget incomplete")
        report["status"] = "PASS_ROUND10_T4_D1_PROCESS"
        _save(result_path, report)
        return report
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = repr(error)
        _save(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = run(args.out, args.fixture)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
