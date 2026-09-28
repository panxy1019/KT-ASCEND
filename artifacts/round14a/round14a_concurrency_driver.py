#!/usr/bin/env python3
"""Run bounded true-engine Round14a concurrency smoke or mirror block."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import statistics
import threading
import time


ROLES = {
    322: ("smoke", "C2", (2,)),
    323: ("smoke", "C2", (2,)),
    324: ("smoke", "C2", (2,)),
    325: ("smoke", "C2", (2,)),
    326: ("smoke", "C4", (4,)),
    327: ("mirror", "A", (1, 2, 4)),
    328: ("mirror", "B", (4, 2, 1)),
    329: ("mirror", "C", (4, 2, 1)),
    330: ("mirror", "D", (1, 2, 4)),
}
WORKLOADS = {"W1": ("edge128", 16), "W2": ("edge128", 128)}
CANARIES = ("canary_marker", "canary_math391", "canary_json",
            "canary_quote", "canary_transform")


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"unterminated JSONL: {path.name}")
    return [json.loads(line) for line in raw.splitlines()]


def request(rid, ids, tokens, *, stream=True, ignore_eos=True):
    return {
        "rid": rid, "input_ids": list(ids), "stream": stream,
        "sampling_params": {
            "temperature": 0, "top_p": 1, "sampling_seed": 20260920,
            "max_new_tokens": tokens, "ignore_eos": ignore_eos,
        },
        "return_logprob": True, "logprob_start_len": -1,
    }


def _stream_one(row, barrier=None):
    from round11_t4_driver import consume_stream, post
    if barrier is not None:
        barrier.wait(timeout=30)
    started = time.monotonic_ns()
    with post("/generate", row["request"], 1800) as response:
        stream = consume_stream(response, row["decode_tokens"])
    ended = time.monotonic_ns()
    arrivals = stream["token_arrival_monotonic_ns"]
    if stream["output_ids"][:2] != [3897, 248046]:
        raise RuntimeError("fixed performance request lost ACK/EOS prefix")
    return {
        **{key: row[key] for key in row if key != "request"},
        "request_id": row["request"]["rid"],
        "client_started_ns": started, "client_ended_ns": ended,
        "completion_tokens": stream["completion_tokens"],
        "output_ids": stream["output_ids"],
        "finite_logprobs": stream["finite_logprobs"],
        "ttft_seconds": (arrivals[0] - started) / 1e9,
        "tpot_seconds": ((arrivals[-1] - arrivals[0]) /
                         (len(arrivals) - 1) / 1e9),
        "e2e_seconds": (ended - started) / 1e9,
    }


def run_wave(rows):
    if not rows:
        return [], None
    barrier = threading.Barrier(len(rows)) if len(rows) > 1 else None
    with ThreadPoolExecutor(max_workers=len(rows)) as pool:
        futures = [pool.submit(_stream_one, row, barrier) for row in rows]
        values = [future.result(timeout=1800) for future in futures]
    window = {
        "started_ns": min(row["client_started_ns"] for row in values),
        "ended_ns": max(row["client_ended_ns"] for row in values),
    }
    window["seconds"] = (window["ended_ns"] - window["started_ns"]) / 1e9
    window["requests"] = len(values)
    window["output_tokens"] = sum(row["completion_tokens"] for row in values)
    window["aggregate_output_tokens_per_second"] = window["output_tokens"] / window["seconds"]
    window["requests_per_second"] = len(values) / window["seconds"]
    return values, window


def _quality_one(row, barrier=None):
    from round11_t4_driver import post
    from round10_t1_protocol import validate_response
    if barrier is not None:
        barrier.wait(timeout=30)
    started = time.monotonic_ns()
    with post("/generate", row["request"], 1800) as response:
        value = json.load(response)
    ended = time.monotonic_ns()
    validated = validate_response(row, value)
    return {"request_id": row["request"]["rid"], "case_id": row["case_id"],
            "quality_pass": True, "validated": validated,
            "e2e_seconds": (ended - started) / 1e9}


def quality_wave(rows):
    barrier = threading.Barrier(len(rows)) if len(rows) > 1 else None
    with ThreadPoolExecutor(max_workers=len(rows)) as pool:
        return [future.result(timeout=1800)
                for future in [pool.submit(_quality_one, row, barrier) for row in rows]]


def quality_row(fixture, version, case_id, *, tokens=32, phase="quality",
                ignore_eos=False, validator=None):
    case = fixture["cases"][case_id]
    return {
        "kind": "quality", "case_id": case_id, "phase": phase,
        "validator": validator or case["validator"],
        "request": request(f"r14a-c-v{version}-{phase}-{case_id}",
                           case["input_ids"], tokens, stream=False,
                           ignore_eos=ignore_eos),
    }


def _annotate_engine(rows, events):
    for row in rows:
        rid = row["request_id"]
        begins = [event for event in events
                  if event.get("status") == "ROUND14A_ENGINE_BATCH_BEGIN"
                  and rid in event.get("request_ids", [])]
        if not begins:
            raise RuntimeError(f"request {rid} lacks true engine-batch evidence")
        decode = [event for event in begins if event.get("mode") == "decode"]
        row["engine"] = {
            "batch_sizes": [event["batch_size"] for event in begins],
            "decode_batch_sizes": [event["batch_size"] for event in decode],
            "max_batch_size": max(event["batch_size"] for event in begins),
            "max_decode_batch_size": max(
                (event["batch_size"] for event in decode), default=0),
        }
        first = min(event["monotonic_ns"] for event in begins)
        if "client_started_ns" in row:
            row["queue_to_first_engine_batch_seconds"] = max(
                0.0, (first - row["client_started_ns"]) / 1e9)


def _validate_planes(out, request_ids):
    from round11_t4_driver import validate_hybrid_path
    from round14a_gdn_model_driver import validate_qsa
    hybrid = read_jsonl(out / "hybrid-path.jsonl")
    layerwise = read_jsonl(out / "layerwise-events.jsonl")
    qsa = read_jsonl(out / "context-reference-events.jsonl")
    ple = read_jsonl(out / "ple-async-events.jsonl")
    result = {}
    total_async = 0
    for rid in request_ids:
        ple_rows = [row for row in ple
                    if row.get("status") == "PLE_BATCH_DRAINED"
                    and row.get("request_id") == rid]
        if not ple_rows:
            raise RuntimeError(f"request {rid} lacks a drained PLE lifecycle")
        async_rows = sum(row.get("async_engaged") is True for row in ple_rows)
        total_async += async_rows
        result[rid] = {
            "path": validate_hybrid_path(rid, hybrid, layerwise),
            "qsa": validate_qsa(rid, qsa),
            "ple": {"drained": len(ple_rows), "async": async_rows,
                    "sync_fallback": len(ple_rows) - async_rows},
        }
    if request_ids and total_async == 0:
        raise RuntimeError("concurrency workload never engaged the qualified PLE async ring")
    return result


def run_smoke(version, out, fixture, tier, report):
    rows = [quality_row(fixture, version, case_id) for case_id in CANARIES]
    rows.append(quality_row(
        fixture, version, "edge128", tokens=32, phase="multi_step",
        ignore_eos=True, validator={"kind": "multi_step", "minimum_tokens": 32}))
    quality = []
    for start in range(0, len(rows), tier):
        batch = rows[start:start + tier]
        report["generation_attempts"] += len(batch)
        save(out / "round14a-concurrency-raw.json", report)
        quality.extend(quality_wave(batch))
        save(out / "round14a-concurrency-raw.json", report)

    # Run one protected request while a peer is cancelled, then reuse a fresh
    # request ID in the same model process.
    from round10_t1_driver import run_cancel
    from round11_t4_driver import post
    cancel_case = fixture["cases"]["cancel128"]
    cancel_row = {
        "case_id": "cancel128", "phase": "cancel",
        "request": request(f"r14a-c-v{version}-cancel", cancel_case["input_ids"],
                           128, stream=True, ignore_eos=True),
    }
    protected = quality_row(
        fixture, version, "edge128", tokens=32, phase="protected",
        ignore_eos=True, validator={"kind": "multi_step", "minimum_tokens": 32})
    report["generation_attempts"] += 2
    report["abort_controls"] += 1
    save(out / "round14a-concurrency-raw.json", report)
    with ThreadPoolExecutor(max_workers=2) as pool:
        cancel_future = pool.submit(run_cancel, cancel_row, out, post_fn=post)
        protected_future = pool.submit(_quality_one, protected)
        cancel_receipt = cancel_future.result(timeout=1800)
        protected_result = protected_future.result(timeout=1800)
    reuse = quality_row(fixture, version, "canary_math391", phase="reuse")
    report["generation_attempts"] += 1
    save(out / "round14a-concurrency-raw.json", report)
    reuse_result = _quality_one(reuse)
    cancel_receipt["reuse"] = {
        "request_id": reuse_result["request_id"], "validator_passed": True,
        "output_ids": reuse_result["validated"]["output_ids"],
    }
    cancel_receipt["status"] = "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"
    from round10_t1_protocol import validate_cancel_receipt
    validate_cancel_receipt(cancel_receipt)
    report["quality"] = quality
    report["cancel_isolation"] = {
        "cancel": cancel_receipt, "protected": protected_result,
        "reuse": reuse_result, "pass": True,
    }
    events = read_jsonl(out / "round14a-concurrency-events.jsonl")
    _annotate_engine(quality + [protected_result, reuse_result], events)
    report["plane_evidence"] = _validate_planes(
        out, [row["request_id"] for row in quality] +
             [protected_result["request_id"], reuse_result["request_id"]])
    maximum = max((event.get("batch_size", 0) for event in events), default=0)
    if maximum < tier:
        raise RuntimeError(f"smoke did not reach real engine concurrency {tier}")
    report["maximum_engine_batch"] = maximum


def performance_row(fixture, version, tier, workload, phase, repeat):
    case_id, tokens = WORKLOADS[workload]
    case = fixture["cases"][case_id]
    return {
        "kind": "performance", "workload": workload, "tier": tier,
        "phase": phase, "repeat": repeat, "case_id": case_id,
        "input_tokens": len(case["input_ids"]), "decode_tokens": tokens,
        "request": request(
            f"r14a-c-v{version}-c{tier}-{workload.lower()}-{phase}-{repeat}",
            case["input_ids"], tokens),
    }


def run_mirror(version, out, fixture, tiers, report):
    # Four simultaneous max-pool requests establish allocation and batch shape
    # before any measured tier.
    max_tier = 4
    rows = [performance_row(fixture, version, max_tier, "W1", "pool_warmup", i)
            for i in range(max_tier)]
    report["generation_attempts"] += max_tier
    save(out / "round14a-concurrency-raw.json", report)
    unused, window = run_wave(rows)
    report["pool_warmup"] = window
    for tier in tiers:
        report["tiers"][str(tier)] = {"warmup": [], "measurements": {}, "windows": {}}
        warmups = [performance_row(fixture, version, tier, "W1", "warmup", i)
                   for i in range(2)]
        for start in range(0, 2, tier):
            wave_rows = warmups[start:start + tier]
            report["generation_attempts"] += len(wave_rows)
            save(out / "round14a-concurrency-raw.json", report)
            values, unused_window = run_wave(wave_rows)
            report["tiers"][str(tier)]["warmup"].extend(values)
        for workload in WORKLOADS:
            rows = [performance_row(
                fixture, version, tier, workload, "measurement", i)
                for i in range(12)]
            values, windows = [], []
            for start in range(0, len(rows), tier):
                wave_rows = rows[start:start + tier]
                report["generation_attempts"] += len(wave_rows)
                save(out / "round14a-concurrency-raw.json", report)
                wave, window = run_wave(wave_rows)
                values.extend(wave)
                windows.append(window)
                save(out / "round14a-concurrency-raw.json", report)
            seconds = sum(item["seconds"] for item in windows)
            tokens = sum(item["output_tokens"] for item in windows)
            aggregate = {
                "waves": len(windows), "seconds": seconds,
                "requests": len(values), "output_tokens": tokens,
                "aggregate_output_tokens_per_second": tokens / seconds,
                "requests_per_second": len(values) / seconds,
                "median_ttft_seconds": statistics.median(
                    item["ttft_seconds"] for item in values),
                "median_tpot_seconds": statistics.median(
                    item["tpot_seconds"] for item in values),
                "median_e2e_seconds": statistics.median(
                    item["e2e_seconds"] for item in values),
            }
            report["tiers"][str(tier)]["measurements"][workload] = values
            report["tiers"][str(tier)]["windows"][workload] = aggregate
    events = read_jsonl(out / "round14a-concurrency-events.jsonl")
    all_measurements = [row for tier in report["tiers"].values()
                        for rows in tier["measurements"].values() for row in rows]
    _annotate_engine(all_measurements, events)
    for tier in tiers:
        selected = [row for rows in report["tiers"][str(tier)]["measurements"].values()
                    for row in rows]
        achieved = max(row["engine"]["max_decode_batch_size"] for row in selected)
        report["tiers"][str(tier)]["achieved_decode_batch"] = achieved
        if achieved < tier:
            raise RuntimeError(f"tier {tier} did not reach true decode engine batch")
    report["plane_evidence"] = _validate_planes(
        out, [row["request_id"] for row in all_measurements])


def run(version, out, fixture_path):
    kind, block, tiers = ROLES[version]
    out = Path(out)
    result = out / "round14a-concurrency-raw.json"
    if result.exists():
        raise FileExistsError(result)
    fixture = json.loads(Path(fixture_path).read_text())
    expected_attempts = 9 if kind == "smoke" else 82
    expected_aborts = 1 if kind == "smoke" else 0
    report = {
        "schema": 1, "status": "RUNNING", "stage": "ROUND14A_C_TRUE_CONCURRENCY",
        "version": version, "kind": kind, "block": block, "tier_order": list(tiers),
        "server_max_concurrency": 2 if version in (322, 323, 324, 325) else 4,
        "generation_budget": expected_attempts, "generation_attempts": 0,
        "abort_controls": 0, "automatic_retries": 0, "profiler": "OFF",
        "baseline": "S0_UNIFORM_Q32_GDN_BASELINE_QSA_PROMOTED_PLE_ASYNC_H129",
        "tiers": {},
    }
    save(result, report)
    try:
        if kind == "smoke":
            run_smoke(version, out, fixture, tiers[0], report)
        else:
            run_mirror(version, out, fixture, tiers, report)
        if (report["generation_attempts"] != expected_attempts
                or report["abort_controls"] != expected_aborts):
            raise RuntimeError("Round14a concurrency schedule budget mismatch")
        report["status"] = "PASS_ROUND14A_C_CONCURRENCY_DRIVER_RAW"
        save(result, report)
        return report
    except BaseException as error:
        report.update(status="FAIL_ROUND14A_C_CONCURRENCY_DRIVER",
                      error=f"{type(error).__name__}: {error}")
        save(result, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=tuple(ROLES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    run(args.version, args.out, args.fixture)


if __name__ == "__main__":
    main()
