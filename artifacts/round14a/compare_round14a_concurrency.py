#!/usr/bin/env python3
"""Strict four-block comparator for Round14a true C1/C2/C4 concurrency."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    327: ("A", [1, 2, 4]),
    328: ("B", [4, 2, 1]),
    329: ("C", [4, 2, 1]),
    330: ("D", [1, 2, 4]),
}
WORKLOADS = ("W1", "W2")
METRICS = ("median_ttft_seconds", "median_tpot_seconds", "median_e2e_seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def validate_smoke(path: Path, version: int, tier: int):
    value = read_json(path)
    if (value.get("status") != "PASS_ROUND14A_C_CONCURRENCY_PROCESS"
            or value.get("version") != version
            or value.get("kind") != "smoke"
            or value.get("generation_attempts") != 9
            or value.get("abort_controls") != 1
            or value.get("automatic_retries") != 0
            or value.get("profiler") != "OFF"
            or value.get("maximum_engine_batch", 0) < tier):
        raise RuntimeError(f"invalid c{tier} smoke identity: {path}")
    if len(value.get("quality", [])) != 6:
        raise RuntimeError(f"c{tier} smoke quality count mismatch")
    if any(not row.get("quality_pass") for row in value["quality"]):
        raise RuntimeError(f"c{tier} smoke quality failure")
    isolation = value.get("cancel_isolation", {})
    if not isolation.get("pass"):
        raise RuntimeError(f"c{tier} cancel/peer/reuse isolation failed")
    if value.get("cleanup") != {"guard_exit_ack": True,
                                "container_removed": True,
                                "physical12_after": "EMPTY"}:
        raise RuntimeError(f"c{tier} smoke cleanup failed")
    return {"path": str(path), "sha256": sha256(path),
            "maximum_engine_batch": value["maximum_engine_batch"]}


def validate_block(path: Path, version: int):
    value = read_json(path)
    block, order = EXPECTED[version]
    if (value.get("status") != "PASS_ROUND14A_C_CONCURRENCY_PROCESS"
            or value.get("version") != version
            or value.get("kind") != "mirror"
            or value.get("block") != block
            or value.get("tier_order") != order
            or value.get("generation_attempts") != 82
            or value.get("abort_controls") != 0
            or value.get("automatic_retries") != 0
            or value.get("profiler") != "OFF"):
        raise RuntimeError(f"invalid formal block identity: {path}")
    if value.get("cleanup") != {"guard_exit_ack": True,
                                "container_removed": True,
                                "physical12_after": "EMPTY"}:
        raise RuntimeError(f"formal block cleanup failed: {path}")
    tiers = value.get("tiers", {})
    if set(tiers) != {"1", "2", "4"}:
        raise RuntimeError(f"formal tier set mismatch: {path}")
    for tier in (1, 2, 4):
        row = tiers[str(tier)]
        if row.get("achieved_decode_batch", 0) < tier:
            raise RuntimeError(f"c{tier} lacks true Decode engine batch: {path}")
        if set(row.get("measurements", {})) != set(WORKLOADS):
            raise RuntimeError(f"c{tier} workload set mismatch: {path}")
        for workload in WORKLOADS:
            measurements = row["measurements"][workload]
            expected_tokens = 16 if workload == "W1" else 128
            if len(measurements) != 12:
                raise RuntimeError(f"c{tier}/{workload} sample count mismatch")
            if any(item.get("completion_tokens") != expected_tokens
                   or not item.get("finite_logprobs") for item in measurements):
                raise RuntimeError(f"c{tier}/{workload} output contract failed")
            window = row.get("windows", {}).get(workload, {})
            if (window.get("requests") != 12
                    or window.get("output_tokens") != 12 * expected_tokens
                    or window.get("aggregate_output_tokens_per_second", 0) <= 0):
                raise RuntimeError(f"c{tier}/{workload} measurement window invalid")
    if len(value.get("plane_evidence", {})) != 72:
        raise RuntimeError(f"formal plane evidence count mismatch: {path}")
    return value, {"path": str(path), "sha256": sha256(path)}


def ratio(numerator, denominator):
    if denominator <= 0:
        raise RuntimeError("non-positive comparison denominator")
    return numerator / denominator


def block_comparison(value):
    tiers = value["tiers"]
    results = {}
    c2_pass = True
    c4_pass = True
    for workload in WORKLOADS:
        windows = {tier: tiers[str(tier)]["windows"][workload]
                   for tier in (1, 2, 4)}
        throughput = {
            "c2_over_c1": ratio(windows[2]["aggregate_output_tokens_per_second"],
                                windows[1]["aggregate_output_tokens_per_second"]),
            "c4_over_c2": ratio(windows[4]["aggregate_output_tokens_per_second"],
                                windows[2]["aggregate_output_tokens_per_second"]),
            "c4_over_c1": ratio(windows[4]["aggregate_output_tokens_per_second"],
                                windows[1]["aggregate_output_tokens_per_second"]),
        }
        latency = {}
        for metric in METRICS:
            latency[metric] = {
                "c2_over_c1": ratio(windows[2][metric], windows[1][metric]),
                "c4_over_c1": ratio(windows[4][metric], windows[1][metric]),
            }
        c2_throughput = throughput["c2_over_c1"] >= 1.20
        c4_throughput = (throughput["c4_over_c2"] >= 1.10
                         and throughput["c4_over_c1"] >= 1.30)
        c2_latency = all(row["c2_over_c1"] <= 1.80 for row in latency.values())
        c4_latency = all(row["c4_over_c1"] <= 3.30 for row in latency.values())
        c2_pass &= c2_throughput and c2_latency
        c4_pass &= c4_throughput and c4_latency
        results[workload] = {
            "windows": windows,
            "throughput_ratios": throughput,
            "latency_ratios": latency,
            "c2_throughput_gate": c2_throughput,
            "c2_latency_gate": c2_latency,
            "c4_throughput_gate": c4_throughput,
            "c4_latency_gate": c4_latency,
        }
    return results, c2_pass, c4_pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-c2", type=Path, required=True)
    parser.add_argument("--smoke-c4", type=Path, required=True)
    for version in EXPECTED:
        parser.add_argument(f"--v{version}", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    if contract.get("status") != "FROZEN_BEFORE_FIRST_FORMAL_BLOCK_COMPLETED":
        raise RuntimeError("comparison contract is not the pre-result frozen identity")
    smoke = {
        "c2": validate_smoke(args.smoke_c2, 325, 2),
        "c4": validate_smoke(args.smoke_c4, 326, 4),
    }
    blocks = {}
    all_c2 = True
    all_c4 = True
    for version in EXPECTED:
        path = getattr(args, f"v{version}")
        value, evidence = validate_block(path, version)
        comparisons, c2_pass, c4_pass = block_comparison(value)
        all_c2 &= c2_pass
        all_c4 &= c4_pass
        blocks[EXPECTED[version][0]] = {
            "version": version,
            "tier_order": EXPECTED[version][1],
            "comparisons": comparisons,
            "c2_block_pass": c2_pass,
            "c4_block_pass": c4_pass,
            "evidence": evidence,
        }
    result = {
        "schema": 1,
        "status": "PASS_ROUND14A_C_CONCURRENCY_COMPARISON",
        "comparison_contract": {"path": str(args.contract),
                                "sha256": sha256(args.contract)},
        "smoke": smoke,
        "blocks": blocks,
        "c2_disposition": "QUALIFIED_TRUE_CONCURRENCY_2" if all_c2
                          else "NO_PROMOTION_C2_BELOW_FROZEN_GATE",
        "c4_disposition": "QUALIFIED_TRUE_CONCURRENCY_4" if all_c4
                          else "NO_PROMOTION_C4_BELOW_FROZEN_GATE",
        "default_max_concurrency": 4 if all_c4 else (2 if all_c2 else 1),
        "performance_objective_achieved": all_c2 or all_c4,
        "claim_scope": "ONE_MODEL_SERIAL_C128_SHARED_READ_ONLY_WEIGHTS_FORMAL_INPUT_LE128_OUTPUT_LE128",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["c2_disposition"],
          result["c4_disposition"], flush=True)


if __name__ == "__main__":
    main()
