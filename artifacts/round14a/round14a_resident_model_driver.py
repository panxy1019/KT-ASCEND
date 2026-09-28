#!/usr/bin/env python3
"""Run one profiler-off Round14a incremental-residency request arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from round14a_gdn_model_driver import (
    CANARIES, WORKLOADS, read_jsonl, request, save, validate_gdn_bindings,
    validate_ple, validate_qsa,
)


ROLES = {
    306: ("baseline", "A"),
    307: ("fplus1", "A"),
    308: ("fplus1", "B"),
    309: ("baseline", "B"),
    310: ("baseline", "A"),
    311: ("fplus1", "A"),
    312: ("fplus1", "B"),
    313: ("baseline", "B"),
    314: ("baseline", "A"),
    315: ("fplus1", "A"),
    316: ("fplus1", "B"),
    317: ("baseline", "B"),
    318: ("baseline", "A"),
    319: ("fplus1", "A"),
    320: ("fplus1", "B"),
    321: ("baseline", "B"),
}


def _quality_row(fixture, version, phase, case_id, tokens=32, *, ignore_eos=False, validator=None):
    case = fixture["cases"][case_id]
    return {
        "kind": "quality", "case_id": case_id, "phase": phase,
        "validator": validator or case["validator"],
        "input_tokens": len(case["input_ids"]), "decode_tokens": tokens,
        "request": request(
            f"r14a-b-v{version}-{phase}-{case_id}", case["input_ids"], tokens,
            stream=False, ignore_eos=ignore_eos,
        ),
    }


def build_schedule(fixture, version):
    strategy, block = ROLES[version]
    rows = []
    for workload, (case_id, tokens) in WORKLOADS.items():
        case = fixture["cases"][case_id]
        for phase, repeat in (("warmup", 0), ("measurement", 1),
                              ("measurement", 2), ("measurement", 3)):
            rows.append({
                "kind": "performance", "workload": workload, "case_id": case_id,
                "phase": phase, "repeat": repeat, "input_tokens": len(case["input_ids"]),
                "decode_tokens": tokens,
                "request": request(
                    f"r14a-b-v{version}-{strategy}-{block.lower()}-{workload.lower()}-{phase}-{repeat}",
                    case["input_ids"], tokens, stream=True, ignore_eos=True,
                ),
            })
    rows.extend(_quality_row(fixture, version, "quality", case_id) for case_id in CANARIES)
    rows.append(_quality_row(
        fixture, version, "multi_step", "edge128", tokens=32, ignore_eos=True,
        validator={"kind": "multi_step", "minimum_tokens": 32},
    ))
    rows.append(_quality_row(fixture, version, "reuse_control", "canary_math391"))
    rows.append(_quality_row(fixture, version, "repeat_control", "canary_marker"))
    if len(rows) != 20:
        raise RuntimeError("Round14a resident request schedule is not 20 attempts")
    return rows


def validate_physical(path, strategy, profile_path):
    value = json.loads(Path(path).read_text())
    if strategy == "baseline":
        if (value.get("status") != "RECORDED_Q32_PHYSICAL_NOT_F7_QUALIFIED" or
                value.get("summary", {}).get("layers") != 48 or
                value.get("summary", {}).get("resident_capacity") != 32 or
                value.get("summary", {}).get("total_bytes") != 15099494400):
            raise RuntimeError("Round14a resident baseline physical receipt mismatch")
        return value
    raw = Path(profile_path).read_bytes()
    profile = json.loads(raw)["profiles"][strategy]
    if (value.get("status") != "PASS_ROUND14A_STATIC_RESIDENCY_PHYSICAL" or
            value.get("strategy") != strategy or
            value.get("document_sha256") != hashlib.sha256(raw).hexdigest() or
            value.get("profile_sha256") != profile["profile_sha256"] or
            value.get("total_experts") != 2016 or
            value.get("total_bytes") != profile["nominal_bf16_bytes"] or
            len(value.get("layers", [])) != 48 or
            value["layers"][0].get("provider") != "PURE_NPU" or
            any(row.get("provider") != "KTEP_MIXED" for row in value["layers"][1:])):
        raise RuntimeError("Round14a FPLUS1 physical receipt mismatch")
    return value


def run(version, out, fixture_path, profiles_path):
    from round11_t4_driver import consume_stream, post, validate_hybrid_path
    from round10_t1_protocol import validate_response

    strategy, block = ROLES[version]
    out = Path(out)
    result_path = out / "round14a-resident-model-raw.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    fixture = json.loads(Path(fixture_path).read_text())
    schedule = build_schedule(fixture, version)
    report = {
        "schema": 1, "status": "RUNNING", "stage": "ROUND14A_B_INCREMENTAL_RESIDENCY_REQUEST",
        "version": version, "strategy": strategy, "block": block, "profiler": "OFF",
        "threshold": 129, "dynamic": "OFF", "ple": "ASYNC",
        "gdn": "baseline", "qsa_prefill": "VECTORIZED", "qsa_decode": "DIRECT_BF16_ROW",
        "generation_budget": 20, "generation_attempts": 0, "abort_controls": 0,
        "automatic_retries": 0, "results": [], "measurement_windows": {}, "physical": None,
    }
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            rid = row["request"]["rid"]
            started = time.monotonic_ns()
            if row["kind"] == "performance":
                with post("/generate", row["request"], 1800) as response:
                    stream = consume_stream(response, row["decode_tokens"])
                ended = time.monotonic_ns()
                arrivals = stream["token_arrival_monotonic_ns"]
                if stream["output_ids"][:2] != [3897, 248046]:
                    raise RuntimeError("performance request lost frozen ACK prefix")
                if stream["completion_tokens"] != row["decode_tokens"]:
                    raise RuntimeError("performance request did not produce fixed output budget")
                item = {
                    **{key: row[key] for key in (
                        "workload", "case_id", "phase", "repeat", "input_tokens", "decode_tokens")},
                    "request_id": rid, "completion_tokens": stream["completion_tokens"],
                    "output_ids": stream["output_ids"], "finite_logprobs": stream["finite_logprobs"],
                    "finish_reason": stream["finish_reason"],
                    "ttft_seconds": (arrivals[0] - started) / 1e9,
                    "e2e_seconds": (ended - started) / 1e9,
                    "tpot_seconds": ((arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1e9
                                     if len(arrivals) > 1 else None),
                    "path": validate_hybrid_path(rid, read_jsonl(out / "hybrid-path.jsonl"),
                                                 read_jsonl(out / "layerwise-events.jsonl")),
                    "qsa": validate_qsa(rid, read_jsonl(out / "context-reference-events.jsonl")),
                    "ple": validate_ple(rid, read_jsonl(out / "ple-async-events.jsonl")),
                }
            else:
                with post("/generate", row["request"], 1800) as response:
                    value = json.load(response)
                checked = validate_response(row, value)
                item = {
                    "case_id": row["case_id"], "phase": row["phase"], "request_id": rid,
                    "e2e_seconds": (time.monotonic_ns() - started) / 1e9,
                    "quality_pass": True, "validated": checked,
                    "path": validate_hybrid_path(rid, read_jsonl(out / "hybrid-path.jsonl"),
                                                 read_jsonl(out / "layerwise-events.jsonl")),
                    "qsa": validate_qsa(rid, read_jsonl(out / "context-reference-events.jsonl")),
                    "ple": validate_ple(rid, read_jsonl(out / "ple-async-events.jsonl")),
                }
            report["results"].append(item)
            if report["physical"] is None:
                report["physical"] = validate_physical(out / "q32-physical.json", strategy, profiles_path)
            save(result_path, report)

        for workload in WORKLOADS:
            selected = [row for row in report["results"]
                        if row.get("workload") == workload and row.get("phase") == "measurement"]
            if len(selected) != 3:
                raise RuntimeError(f"{workload} measurement count mismatch")
            seconds = sum(row["e2e_seconds"] for row in selected)
            tokens = sum(row["completion_tokens"] for row in selected)
            report["measurement_windows"][workload] = {
                "seconds": seconds, "successful_requests": 3, "output_tokens": tokens,
                "serial_output_tokens_per_second": tokens / seconds,
                "requests_per_second": 3 / seconds,
            }
        if report["generation_attempts"] != 20:
            raise RuntimeError("Round14a resident schedule/budget incomplete")
        report["gdn_binding"] = validate_gdn_bindings(out / "round14a-gdn-branch.jsonl", "baseline")
        report["status"] = "PASS_ROUND14A_B_RESIDENT_MODEL_DRIVER_RAW"
        save(result_path, report)
        return report
    except BaseException as error:
        report.update(status="FAIL_ROUND14A_B_RESIDENT_MODEL_DRIVER",
                      error=f"{type(error).__name__}: {error}")
        save(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=tuple(ROLES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.version, args.out, args.fixture, args.profiles)
    print(result["status"], args.version, flush=True)


if __name__ == "__main__":
    main()
