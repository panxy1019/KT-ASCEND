#!/usr/bin/env python3
"""Compare the user-authorized two-process Round15 A directional screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def output_ids(row):
    return row.get("output_ids", row.get("validated", {}).get("output_ids"))


def _cgroup_integer(row, key):
    value = row.get("owned_cgroup", {}).get(key)
    return int(value) if value is not None else 0


def resources(monitor_path):
    rows = read_jsonl(monitor_path)
    sampled = [row for row in rows if "hbm_used_mib" in row]
    events = "\n".join(
        row.get("owned_cgroup", {}).get("memory.events", "") for row in sampled
    )
    return {
        "samples": len(sampled),
        "whole_card_hbm_peak_mib": max(row.get("hbm_used_mib", 0) for row in sampled),
        "owned_hbm_peak_mib": max(row.get("owned_process_mib", 0) for row in sampled),
        "training_hbm_peak_mib": max(row.get("training_process_mib", 0) for row in sampled),
        "host_cgroup_usage_peak_bytes": max(
            _cgroup_integer(row, "memory.usage_in_bytes") for row in sampled
        ),
        "host_cgroup_recorded_peak_bytes": max(
            _cgroup_integer(row, "memory.max_usage_in_bytes") for row in sampled
        ),
        "host_memavailable_min_kib": min(
            row.get("host_mem_available_kib", 1 << 62) for row in sampled
        ),
        "oom_false": "oom 1" not in events,
        "failcnt_zero": all(
            _cgroup_integer(row, "memory.failcnt") == 0 for row in sampled
        ),
    }


def measured(document, workload):
    return [
        row for row in document["results"]
        if row.get("workload") == workload and row.get("phase") == "measurement"
    ]


def compare(baseline, candidate, baseline_monitor, candidate_monitor):
    for document, implementation in ((baseline, "baseline"), (candidate, "candidate")):
        if (
            document.get("status") != "PASS_ROUND15_A_QSA_MODEL_PROCESS"
            or document.get("implementation") != implementation
            or document.get("generation_attempts") != 14
            or document.get("abort_controls") != 1
            or document.get("automatic_retries") != 0
            or document.get("cleanup", {}).get("physical12_after") != "REGISTERED_TRAINING_ONLY"
        ):
            raise RuntimeError(f"{implementation} process contract mismatch")
    baseline_binding = baseline.get("qsa_selection_binding", {})
    candidate_binding = candidate.get("qsa_selection_binding", {})
    if (
        baseline_binding.get("implementation") != "EAGER_REFERENCE"
        or baseline_binding.get("first_hit_rows") != 0
        or candidate_binding.get("implementation") != "FULL_SET_CAUSAL_SHORT_CONTEXT"
        or candidate_binding.get("layers") != list(range(3, 48, 4))
        or candidate_binding.get("first_hit_rows") != 12
    ):
        raise RuntimeError("actual candidate selection binding mismatch")
    if len(baseline["results"]) != 14 or len(candidate["results"]) != 14:
        raise RuntimeError("directional screen result count mismatch")
    output_comparison = []
    for index, (left, right) in enumerate(zip(baseline["results"], candidate["results"])):
        identity = (left.get("case_id"), left.get("workload"), left.get("phase"), left.get("repeat"))
        if identity != (right.get("case_id"), right.get("workload"), right.get("phase"), right.get("repeat")):
            raise RuntimeError(f"request identity mismatch at {index}")
        exact = output_ids(left) == output_ids(right)
        if not exact:
            raise RuntimeError(f"output token mismatch at {identity}")
        output_comparison.append({"identity": identity, "output_ids_exact": True})

    workloads = {}
    for workload in ("W1", "W2"):
        left, right = measured(baseline, workload), measured(candidate, workload)
        if len(left) != 2 or len(right) != 2:
            raise RuntimeError(f"{workload} does not have two measurements per arm")
        metrics = {}
        for metric in ("ttft_seconds", "tpot_seconds", "e2e_seconds"):
            baseline_values = [row[metric] for row in left]
            candidate_values = [row[metric] for row in right]
            baseline_median = statistics.median(baseline_values)
            candidate_median = statistics.median(candidate_values)
            metrics[metric] = {
                "baseline_samples": baseline_values,
                "candidate_samples": candidate_values,
                "baseline_median": baseline_median,
                "candidate_median": candidate_median,
                "candidate_over_baseline": candidate_median / baseline_median,
                "candidate_improvement_fraction": 1 - candidate_median / baseline_median,
            }
        baseline_rate = baseline["measurement_windows"][workload][
            "serial_output_tokens_per_second"
        ]
        candidate_rate = candidate["measurement_windows"][workload][
            "serial_output_tokens_per_second"
        ]
        metrics["serial_output_tokens_per_second"] = {
            "baseline": baseline_rate,
            "candidate": candidate_rate,
            "candidate_over_baseline": candidate_rate / baseline_rate,
            "candidate_improvement_fraction": candidate_rate / baseline_rate - 1,
        }
        workloads[workload] = metrics

    w2 = workloads["W2"]
    no_request_gain = (
        w2["tpot_seconds"]["candidate_over_baseline"] >= 1.0
        and w2["serial_output_tokens_per_second"]["candidate_over_baseline"] <= 1.0
    )
    return {
        "schema": 1,
        "status": (
            "ROUND15_A_DIRECTIONAL_NO_REQUEST_GAIN_KEEP_DEFAULT_OFF"
            if no_request_gain else "ROUND15_A_DIRECTIONAL_RESULT_REQUIRES_REVIEW"
        ),
        "evidence_class": "TWO_FRESH_PROCESS_DIRECTIONAL_SCREEN_NOT_ABBA",
        "versions": {"baseline": baseline["version"], "candidate": candidate["version"]},
        "candidate": "FULL_SET_CAUSAL_SHORT_CONTEXT",
        "actual_candidate_binding": candidate_binding,
        "quality": {
            "all_14_output_sequences_exact": True,
            "cancel_reuse_pass": True,
            "finite_logprobs": all(
                row.get("finite_logprobs", True)
                for document in (baseline, candidate)
                for row in document["results"]
            ),
            "comparisons": output_comparison,
        },
        "workloads": workloads,
        "resources": {
            "baseline": resources(baseline_monitor),
            "candidate": resources(candidate_monitor),
        },
        "disposition": {
            "default": "OFF",
            "request_gain": "NOT_OBSERVED_IN_W2_DIRECTIONAL_SCREEN",
            "component_gain": "PASS_TWO_INDEPENDENT_BLOCKS",
            "reason": (
                "W2 candidate median TPOT is not lower and serial output throughput is not higher; "
                "the user-authorized two-process screen does not support promotion."
            ),
            "limitation": (
                "One fresh process per arm with two measurements per workload estimates direction and "
                "magnitude only; it is not a four-process ABBA stability qualification."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-monitor", type=Path, required=True)
    parser.add_argument("--candidate-monitor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        args.baseline_monitor,
        args.candidate_monitor,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
