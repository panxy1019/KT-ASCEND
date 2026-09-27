#!/usr/bin/env python3
"""Run one unprofiled Round12 QSA request A/B/B/A process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


ROLE_VERSIONS = {
    155: ("BASELINE", "A"),
    156: ("CANDIDATE", "A"),
    157: ("CANDIDATE", "B"),
    158: ("BASELINE", "B"),
}
WORKLOADS = (
    ("W1", "edge128", "round10", 16),
    ("W2", "edge128", "round10", 128),
    ("W3", "edge512", "round11", 16),
    ("W4", "edge64", "round10", 32),
)


def _save(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build_schedule(round10_fixture, round11_fixture, *, version, mode, block):
    if ROLE_VERSIONS.get(version) != (mode, block):
        raise ValueError("Round12 P3 role/version mismatch")
    fixtures = {"round10": round10_fixture, "round11": round11_fixture}
    rows = []
    index = 0
    for workload, case_id, source, decode_tokens in WORKLOADS:
        case = fixtures[source]["cases"][case_id]
        for repeat in range(4):
            index += 1
            phase = "warmup" if repeat == 0 else "measurement"
            rid = (f"r12-p3-v{version}-{mode.lower()}-{block.lower()}-"
                   f"{workload.lower()}-{phase}-{repeat}")
            rows.append({
                "index": index, "workload": workload, "case_id": case_id,
                "fixture_source": source, "phase": phase,
                "repeat": repeat, "input_tokens": len(case["input_ids"]),
                "decode_tokens": decode_tokens,
                "request": {
                    "rid": rid, "input_ids": list(case["input_ids"]), "stream": True,
                    "sampling_params": {
                        "temperature": 0, "top_p": 1,
                        "sampling_seed": 20260920,
                        "max_new_tokens": decode_tokens, "ignore_eos": True,
                    },
                    "return_logprob": True, "logprob_start_len": -1,
                },
            })
    return rows


def timing_metrics(started_ns, ended_ns, arrivals):
    if not arrivals or arrivals[0] < started_ns or ended_ns < arrivals[-1]:
        raise ValueError("stream timing order is invalid")
    return {
        "ttft_seconds": (arrivals[0] - started_ns) / 1e9,
        "e2e_seconds": (ended_ns - started_ns) / 1e9,
        "tpot_seconds": (
            (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1e9
            if len(arrivals) > 1 else None),
        "timing_primary": "CLIENT_FIRST_OUTPUT_EVENT_TTFT",
    }


def run(mode, block, version, out, round10_fixture_path,
        round11_fixture_path, *, post_fn=None):
    from round11_t4_driver import consume_stream, post, read_jsonl, validate_hybrid_path
    from round12_p2_driver import (
        _read_jsonl, validate_binding, validate_qsa_dispatch,
    )

    if post_fn is None:
        post_fn = post
    out = Path(out)
    result_path = out / "round12-p3-raw.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    r10_path, r11_path = map(Path, (round10_fixture_path, round11_fixture_path))
    r10, r11 = json.loads(r10_path.read_text()), json.loads(r11_path.read_text())
    schedule = build_schedule(r10, r11, version=version, mode=mode, block=block)
    report = {
        "schema": 1, "status": "RUNNING", "stage": "P3_QSA_ABBA",
        "mode": mode, "block": block, "version": version,
        "profiler": "OFF", "timing_primary": "CLIENT_FIRST_OUTPUT_EVENT_TTFT",
        "forced_hybrid_profile": True, "forced_hybrid_threshold": 513,
        "default_policy_promoted": False,
        "round10_fixture_sha256": hashlib.sha256(r10_path.read_bytes()).hexdigest(),
        "round11_fixture_sha256": hashlib.sha256(r11_path.read_bytes()).hexdigest(),
        "generation_budget": 16, "generation_attempts": 0,
        "automatic_retries": 0, "results": [],
    }
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            started = time.monotonic_ns()
            with post_fn("/generate", row["request"], 1800) as response:
                stream = consume_stream(response, row["decode_tokens"])
            ended = time.monotonic_ns()
            if stream["output_ids"][:2] != [3897, 248046]:
                raise RuntimeError("Round12 P3 request changed the frozen ACK prefix")
            rid = row["request"]["rid"]
            qsa = validate_qsa_dispatch(
                rid, _read_jsonl(out / "context-reference-events.jsonl"),
                mode=mode, require_decode_row=True)
            path = validate_hybrid_path(
                rid, read_jsonl(out / "hybrid-path.jsonl"),
                read_jsonl(out / "layerwise-events.jsonl"))
            report["results"].append({
                "index": row["index"], "workload": row["workload"],
                "case_id": row["case_id"], "phase": row["phase"],
                "repeat": row["repeat"], "request_id": rid,
                "input_tokens": row["input_tokens"],
                "decode_tokens": row["decode_tokens"],
                **timing_metrics(started, ended, stream["token_arrival_monotonic_ns"]),
                "quality_pass": True, "output_ids": stream["output_ids"],
                "finite_logprobs": stream["finite_logprobs"],
                "finish_reason": stream["finish_reason"],
                "completion_tokens": stream["completion_tokens"],
                "qsa_dispatch": qsa, "path": path,
            })
            _save(result_path, report)
        report["qsa_source_binding"] = validate_binding(
            _read_jsonl(out / "context-reference-events.jsonl"), mode)
        if (report["generation_attempts"] != 16
                or len(report["results"]) != 16
                or sum(row["phase"] == "measurement" for row in report["results"]) != 12):
            raise RuntimeError("Round12 P3 schedule is incomplete")
        report["status"] = "PASS_ROUND12_P3_DRIVER_RAW"
        _save(result_path, report)
        return report
    except BaseException as error:
        report.update(status="FAIL_ROUND12_P3_DRIVER",
                      error=f"{type(error).__name__}: {error}")
        _save(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("BASELINE", "CANDIDATE"), required=True)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    parser.add_argument("--version", type=int, choices=tuple(ROLE_VERSIONS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round10-fixture", type=Path, required=True)
    parser.add_argument("--round11-fixture", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.mode, args.block, args.version, args.out,
                 args.round10_fixture, args.round11_fixture)
    print(result["status"], args.mode, args.block, flush=True)


if __name__ == "__main__":
    main()
