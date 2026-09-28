#!/usr/bin/env python3
"""Strict two-block comparator for Round14a FPLUS1 request evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


EXPECTED = {
    "A": ((310, "baseline", "A"), (319, "fplus1", "A")),
    "B": ((321, "baseline", "B"), (320, "fplus1", "B")),
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    value = json.loads(path.read_text())
    if (value.get("status") != "PASS_ROUND14A_B_RESIDENT_MODEL_PROCESS" or
            value.get("generation_attempts") != 20 or value.get("abort_controls") != 0 or
            value.get("automatic_retries") != 0 or value.get("profiler") != "OFF" or
            value.get("threshold") != 129 or value.get("dynamic") != "OFF" or
            value.get("ple") != "ASYNC" or value.get("gdn") != "baseline" or
            value.get("qsa_prefill") != "VECTORIZED" or
            value.get("qsa_decode") != "DIRECT_BF16_ROW"):
        raise RuntimeError(f"invalid Round14a resident process contract: {path}")
    cleanup = value.get("cleanup", {})
    if cleanup != {"guard_exit_ack": True, "container_removed": True,
                   "physical12_after": "EMPTY"}:
        raise RuntimeError(f"incomplete cleanup: {path}")
    measurements = [row for row in value.get("results", []) if row.get("phase") == "measurement"]
    if len(measurements) != 9 or any(not row.get("finite_logprobs") for row in measurements):
        raise RuntimeError(f"measurement quality mismatch: {path}")
    quality = [row for row in value.get("results", []) if row.get("quality_pass")]
    if len(quality) != 8:
        raise RuntimeError(f"quality/control count mismatch: {path}")
    if any(row.get("path", {}).get("selection_count", 0) <= 0 for row in value["results"]):
        raise RuntimeError(f"actual Hybrid path missing: {path}")
    return value, {"path": str(path), "sha256": sha(path)}


def median(value, workload, field):
    rows = [row[field] for row in value["results"]
            if row.get("workload") == workload and row.get("phase") == "measurement"]
    if len(rows) != 3 or any(item is None for item in rows):
        raise RuntimeError(f"{workload}/{field} lacks three measurements")
    return statistics.median(rows)


def summarize(value):
    result = {}
    for workload in ("W1", "W2", "W4"):
        window = value["measurement_windows"][workload]
        result[workload] = {
            "ttft_median_seconds": median(value, workload, "ttft_seconds"),
            "tpot_median_seconds": median(value, workload, "tpot_seconds"),
            "e2e_median_seconds": median(value, workload, "e2e_seconds"),
            "serial_output_tokens_per_second": window["serial_output_tokens_per_second"],
            "requests_per_second": window["requests_per_second"],
            "output_tokens": window["output_tokens"],
        }
    return result


def ratio(candidate, baseline): return candidate / baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline_a", "candidate_a", "candidate_b", "baseline_b"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"A": (args.baseline_a, args.candidate_a),
             "B": (args.baseline_b, args.candidate_b)}
    blocks = {}
    all_pass = True
    for block, pair in paths.items():
        baseline, baseline_evidence = read(pair[0])
        candidate, candidate_evidence = read(pair[1])
        if ((baseline["version"], baseline["strategy"], baseline["block"]),
                (candidate["version"], candidate["strategy"], candidate["block"])) != EXPECTED[block]:
            raise RuntimeError(f"Round14a B block {block} identity/order mismatch")
        b = summarize(baseline); c = summarize(candidate)
        comparisons = {}
        for workload in ("W1", "W2", "W4"):
            comparisons[workload] = {
                "ttft_ratio": ratio(c[workload]["ttft_median_seconds"], b[workload]["ttft_median_seconds"]),
                "tpot_ratio": ratio(c[workload]["tpot_median_seconds"], b[workload]["tpot_median_seconds"]),
                "e2e_ratio": ratio(c[workload]["e2e_median_seconds"], b[workload]["e2e_median_seconds"]),
                "serial_throughput_ratio": ratio(c[workload]["serial_output_tokens_per_second"],
                                                   b[workload]["serial_output_tokens_per_second"]),
            }
        main_gate = (comparisons["W2"]["tpot_ratio"] <= .95 and
                     comparisons["W2"]["serial_throughput_ratio"] >= 1.05)
        protection = all(
            not (comparisons[w][metric] > 1.05) if metric != "serial_throughput_ratio"
            else not (comparisons[w][metric] < .95)
            for w in ("W1", "W4")
            for metric in ("ttft_ratio", "tpot_ratio", "e2e_ratio", "serial_throughput_ratio")
        )
        block_pass = main_gate and protection
        all_pass &= block_pass
        blocks[block] = {
            "baseline": b, "candidate": c, "comparison": comparisons,
            "main_w2_gate": main_gate, "w1_w4_protection": protection,
            "block_pass": block_pass,
            "evidence": {"baseline": baseline_evidence, "candidate": candidate_evidence},
        }
    result = {
        "schema": 1,
        "status": "PASS_ROUND14A_B_RESIDENT_REQUEST_COMPARISON",
        "candidate": "FPLUS1",
        "added_experts": 480,
        "added_nominal_bf16_bytes": 4718592000,
        "added_nominal_bf16_gib": 4.39453125,
        "blocks": blocks,
        "promotion_contract": {
            "both_blocks_w2_tpot_ratio_lte": .95,
            "both_blocks_w2_serial_throughput_ratio_gte": 1.05,
            "w1_w4_consistent_regression_forbidden": .05,
        },
        "promotion": "PROMOTE_FPLUS1" if all_pass else "NO_PROMOTION_BELOW_REQUEST_GATE",
        "default_profile": "FPLUS1" if all_pass else "UNIFORM_Q32",
        "performance_objective_achieved": all_pass,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["promotion"], flush=True)


if __name__ == "__main__": main()
