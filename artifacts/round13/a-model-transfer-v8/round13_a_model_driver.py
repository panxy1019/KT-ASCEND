#!/usr/bin/env python3
"""Run one 24-attempt Round13 static-placement model arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time


WORKLOADS = {
    "W0": ("round10", "edge64", 2),
    "W1": ("round11", "edge128", 16),
    "W2": ("round11", "edge128", 128),
    "W3": ("round11", "edge512", 16),
    "W4": ("round10", "edge64", 32),
}
QSA_LAYERS = list(range(3, 48, 4))
EXPECTED_ORDER = {
    196: ("uniform", "A"), 197: ("frequency", "A"),
    198: ("front_loading", "A"), 199: ("random_seed42", "A"),
    200: ("random_seed42", "B"), 201: ("front_loading", "B"),
    202: ("frequency", "B"), 203: ("uniform", "B"),
    204: ("uniform", "A"),
    205: ("uniform", "A"),
    206: ("uniform", "A"),
    207: ("uniform", "A"),
    208: ("frequency", "A"),
    209: ("front_loading", "A"),
    210: ("random_seed42", "A"),
    211: ("random_seed42", "B"),
    212: ("front_loading", "B"),
    213: ("frequency", "B"),
    214: ("uniform", "B"),
    223: ("front_loading", "A"),
    224: ("random_seed42", "A"),
    225: ("front_loading", "A"),
    226: ("random_seed42", "A"),
}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists(): return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"): raise RuntimeError(f"unterminated {path.name}")
    return [json.loads(line) for line in raw.splitlines()]


def request(rid, ids, tokens, *, stream, ignore_eos):
    return {"rid": rid, "input_ids": list(ids), "stream": stream,
        "sampling_params": {"temperature": 0, "top_p": 1, "sampling_seed": 20260920,
            "max_new_tokens": tokens, "ignore_eos": ignore_eos},
        "return_logprob": True, "logprob_start_len": -1}


def performance_schedule(fixtures, version, strategy, block):
    rows = []
    for workload, (source, case_id, decode_tokens) in WORKLOADS.items():
        case = fixtures[source]["cases"][case_id]
        for phase, repeat in (("warmup", 0), ("measurement", 1),
                              ("measurement", 2), ("measurement", 3)):
            rid = f"r13-a-v{version}-{strategy}-{block.lower()}-{workload.lower()}-{phase}-{repeat}"
            rows.append({"kind": "performance", "workload": workload, "case_id": case_id,
                "phase": phase, "repeat": repeat, "input_tokens": len(case["input_ids"]),
                "decode_tokens": decode_tokens,
                "request": request(rid, case["input_ids"], decode_tokens,
                                   stream=True, ignore_eos=True)})
    return rows


def quality_schedule(fixtures, version, strategy, block):
    f = fixtures["round11"]["cases"]
    if block == "A":
        definitions = [
            ("canary_marker", "quality", 32, None),
            ("canary_math391", "quality", 32, None),
            ("cancel256", "cancel", 16, {"kind": "abort"}),
            ("canary_json", "reuse", 32, None),
        ]
    else:
        definitions = [
            ("canary_json", "quality", 32, None),
            ("canary_quote", "quality", 32, None),
            ("canary_transform", "quality", 32, None),
            ("edge128", "multi_step", 32, {"kind": "multi_step", "minimum_tokens": 32}),
        ]
    rows = []
    for case_id, phase, tokens, override in definitions:
        case = f[case_id]
        rid = f"r13-a-v{version}-{strategy}-{block.lower()}-{case_id}-{phase}"
        rows.append({"kind": "quality", "case_id": case_id, "phase": phase,
            "validator": override or case["validator"], "input_tokens": len(case["input_ids"]),
            "decode_tokens": tokens,
            "request": request(rid, case["input_ids"], tokens,
                stream=phase == "cancel", ignore_eos=phase == "multi_step")})
    return rows


def validate_qsa(request_id, events):
    rows = [row for row in events if row.get("status") == "CONTEXT_REFERENCE_QSA_DISPATCH"
            and row.get("request_id") == request_id]
    vector = sorted({row.get("layer") for row in rows if row.get("strategy") == "VECTORIZED"})
    row = sorted({item.get("layer") for item in rows if item.get("strategy") == "ROW"})
    if vector != QSA_LAYERS or row != QSA_LAYERS:
        raise RuntimeError("promoted QSA Prefill/ROW Decode identity missing")
    return {"vector_layers": vector, "row_layers": row, "dispatch_count": len(rows)}


def validate_ple(request_id, events):
    if not any(row.get("status") == "PLE_ASYNC_BOUND" for row in events):
        raise RuntimeError("PLE ASYNC binding missing")
    rows = [row for row in events if row.get("status") == "PLE_BATCH_DRAINED"
            and row.get("request_id") == request_id]
    if not rows or not any(row.get("async_engaged") is True for row in rows):
        raise RuntimeError("request did not engage and drain PLE ASYNC")
    return {"drained": len(rows), "async": sum(row.get("async_engaged") is True for row in rows)}


def validate_round13_hybrid_path(request_id, hybrid_events, layerwise_events, profile):
    """Validate Hybrid evidence for mixed or cross-layer static placement.

    The inherited Round11 validator assumes every observed KTEP layer contains
    both CPU and NPU experts.  Front-loading instead has full-resident layers
    followed by CPU-only layers, so its bounded observer can legitimately see
    only the CPU half.  In that case the already-frozen q=512 profile supplies
    the NPU branch identity; the model's physical placement receipt is checked
    separately for the same profile before the request is accepted.
    """
    related = [row for row in hybrid_events if row.get("request_id") == request_id]
    selections = [row for row in related if row.get("stage") == "SELECTION"]
    stages = {row.get("stage") for row in related}
    counts = profile.get("per_layer_counts")
    if (not isinstance(counts, list) or len(counts) != 48
            or any(not isinstance(value, int) or value < 0 or value > 512 for value in counts)):
        raise RuntimeError("Round13 placement profile counts are invalid")
    if (
        not selections
        or any(row.get("path") != "HYBRID" or row.get("basis") != "REQUEST_TOTAL"
               for row in selections)
        or not {"CPU_SUBMIT", "CPU_RESULT", "ORIGINAL_HYBRID_APPLY"}.issubset(stages)
        or any(row.get("request_id") == request_id for row in layerwise_events)
    ):
        raise RuntimeError("Round13 request did not use the complete Hybrid path")
    if "NPU_RESIDENT_APPLY" in stages:
        npu_evidence = "OBSERVED_NPU_RESIDENT_APPLY"
    elif 512 in counts:
        npu_evidence = "FROZEN_FULL_RESIDENT_LAYER"
    else:
        raise RuntimeError("Round13 request did not use the complete Hybrid path")
    return {
        "selection_count": len(selections),
        "dispatch_tokens": [row.get("batch_tokens") for row in selections],
        "cpu_submit_count": sum(row.get("stage") == "CPU_SUBMIT" for row in related),
        "cpu_result_count": sum(row.get("stage") == "CPU_RESULT" for row in related),
        "npu_resident_count": sum(row.get("stage") == "NPU_RESIDENT_APPLY" for row in related),
        "npu_evidence": npu_evidence,
    }


def run(version, strategy, block, out, profiles_path, fixture10_path, fixture11_path):
    from round11_t4_driver import consume_stream, post
    from round10_t1_driver import run_cancel
    from round10_t1_protocol import validate_cancel_receipt, validate_response
    if EXPECTED_ORDER.get(version) != (strategy, block):
        raise ValueError("Round13 A version/order mismatch")
    out = Path(out); result_path = out / "round13-a-raw.json"
    if result_path.exists(): raise FileExistsError(result_path)
    fixtures = {"round10": json.loads(Path(fixture10_path).read_text()),
                "round11": json.loads(Path(fixture11_path).read_text())}
    profiles_raw = Path(profiles_path).read_bytes(); profiles = json.loads(profiles_raw)
    profile = profiles["profiles"][strategy]
    schedule = performance_schedule(fixtures, version, strategy, block) + quality_schedule(
        fixtures, version, strategy, block)
    report = {"schema": 1, "status": "RUNNING", "stage": "ROUND13_A_STATIC_PLACEMENT",
        "version": version, "strategy": strategy, "block": block, "profiler": "OFF",
        "profile_threshold": 513, "default_threshold": 129,
        "placement_profile_sha256": profile["profile_sha256"],
        "placement_document_sha256": hashlib.sha256(profiles_raw).hexdigest(),
        "generation_budget": 24, "generation_attempts": 0,
        "abort_controls": 0, "automatic_retries": 0, "results": [],
        "measurement_windows": {}, "physical": None}
    pending_cancel = None
    try:
        for workload in WORKLOADS:
            group = [row for row in schedule if row.get("workload") == workload]
            measurement_started = None; measurement_ended = None; measured_tokens = 0; successes = 0
            for row in group:
                report["generation_attempts"] += 1
                rid = row["request"]["rid"]
                if row["phase"] == "measurement" and measurement_started is None:
                    measurement_started = time.monotonic_ns()
                started = time.monotonic_ns()
                with post("/generate", row["request"], 1800) as response:
                    stream = consume_stream(response, row["decode_tokens"])
                ended = time.monotonic_ns()
                if report["physical"] is None:
                    physical = json.loads((out / "q32-physical.json").read_text())
                    if (physical.get("status") != "PASS_ROUND13_STATIC_PLACEMENT_PHYSICAL" or
                            physical.get("strategy") != strategy or
                            physical.get("profile_sha256") != profile["profile_sha256"] or
                            physical.get("total_experts") != 1536 or
                            physical.get("total_bytes") != profile["nominal_bf16_bytes"]):
                        raise RuntimeError("Round13 model physical placement receipt failed")
                    report["physical"] = physical
                path = validate_round13_hybrid_path(
                    rid, read_jsonl(out / "hybrid-path.jsonl"),
                    read_jsonl(out / "layerwise-events.jsonl"), profile)
                if stream["output_ids"][:2] != [3897, 248046]:
                    raise RuntimeError("performance workload lost frozen ACK prefix")
                item = {**{k: row[k] for k in ("workload","case_id","phase","repeat","input_tokens","decode_tokens")},
                    "request_id": rid, **stream,
                    "ttft_seconds": (stream["token_arrival_monotonic_ns"][0] - started) / 1e9,
                    "e2e_seconds": (ended - started) / 1e9,
                    "tpot_seconds": ((stream["token_arrival_monotonic_ns"][-1] -
                        stream["token_arrival_monotonic_ns"][0]) / (len(stream["output_ids"])-1) / 1e9
                        if len(stream["output_ids"]) > 1 else None),
                    "path": path,
                    "qsa": validate_qsa(rid, read_jsonl(out / "context-reference-events.jsonl")),
                    "ple": validate_ple(rid, read_jsonl(out / "ple-async-events.jsonl"))}
                del item["token_arrival_monotonic_ns"]
                report["results"].append(item)
                if row["phase"] == "measurement":
                    measured_tokens += stream["completion_tokens"]; successes += 1
                    measurement_ended = ended
                save(result_path, report)
            seconds = (measurement_ended - measurement_started) / 1e9
            report["measurement_windows"][workload] = {"seconds": seconds,
                "successful_requests": successes, "output_tokens": measured_tokens,
                "serial_output_tokens_per_second": measured_tokens / seconds,
                "requests_per_second": successes / seconds}
        for row in schedule[-4:]:
            report["generation_attempts"] += 1
            rid = row["request"]["rid"]; started = time.monotonic_ns()
            if row["phase"] == "cancel":
                report["abort_controls"] += 1
                pending_cancel = run_cancel(row, out, post_fn=post)
                report["results"].append({"case_id": row["case_id"], "phase": "cancel",
                    "request_id": rid, "e2e_seconds": (time.monotonic_ns()-started)/1e9,
                    "cancel_status": pending_cancel["status"]})
                save(result_path, report); continue
            with post("/generate", row["request"], 1800) as response:
                value = json.load(response)
            validated = validate_response(row, value)
            path = validate_round13_hybrid_path(
                rid, read_jsonl(out / "hybrid-path.jsonl"),
                read_jsonl(out / "layerwise-events.jsonl"), profile)
            item = {"case_id": row["case_id"], "phase": row["phase"], "request_id": rid,
                "e2e_seconds": (time.monotonic_ns()-started)/1e9, "quality_pass": True,
                "validated": validated, "path": path,
                "qsa": validate_qsa(rid, read_jsonl(out / "context-reference-events.jsonl")),
                "ple": validate_ple(rid, read_jsonl(out / "ple-async-events.jsonl"))}
            report["results"].append(item)
            if row["phase"] == "reuse":
                if pending_cancel is None: raise RuntimeError("reuse lacks preceding cancel")
                pending_cancel["reuse"] = {"request_id": rid, "validator_passed": True,
                    "output_ids": validated["output_ids"]}
                pending_cancel["status"] = "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"
                validate_cancel_receipt(pending_cancel)
                save(out / "round13-a-cancel-receipt.json", pending_cancel)
                report["results"][-2]["cancel_status"] = pending_cancel["status"]
                pending_cancel = None
            save(result_path, report)
        expected_abort = 1 if block == "A" else 0
        if (report["generation_attempts"] != 24 or report["abort_controls"] != expected_abort or
                pending_cancel is not None or len(report["measurement_windows"]) != 5):
            raise RuntimeError("Round13 A schedule/budget incomplete")
        report["status"] = "PASS_ROUND13_A_MODEL_DRIVER_RAW"
        save(result_path, report); return report
    except BaseException as error:
        report.update(status="FAIL_ROUND13_A_MODEL_DRIVER", error=f"{type(error).__name__}: {error}")
        save(result_path, report); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--block", choices=("A","B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--fixture10", type=Path, required=True)
    parser.add_argument("--fixture11", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.version,args.strategy,args.block,args.out,args.profiles,args.fixture10,args.fixture11)
    print(result["status"],args.strategy,args.block,flush=True)


if __name__ == "__main__": main()
