#!/usr/bin/env python3
"""Compare the frozen Round13 F S0/final four-process mirror."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


WORKLOADS = ("W0", "W1", "W2", "W3", "W4")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def cgroup_peak_gib(rows):
    values = []
    for row in rows:
        raw = (row.get("owned_cgroup") or {}).get("memory.max_usage_in_bytes")
        if raw is not None:
            values.append(int(raw.strip()))
    return max(values) / 2**30 if values else None


def summarize_run(path):
    path = Path(path)
    root = path.parent
    document = json.loads(path.read_text())
    if document.get("status") != "PASS_ROUND13_F_MODEL_PROCESS":
        raise RuntimeError(f"non-PASS F result {path}")
    metrics = {}
    for workload in WORKLOADS:
        rows = [
            row for row in document["results"]
            if row.get("workload") == workload and row.get("phase") == "measurement"
        ]
        if len(rows) != 3:
            raise RuntimeError(f"{path} {workload} lacks three measurements")
        metrics[workload] = {
            "e2e_seconds_median": statistics.median(row["e2e_seconds"] for row in rows),
            "ttft_seconds_median": statistics.median(row["ttft_seconds"] for row in rows),
            "tpot_seconds_median": statistics.median(row["tpot_seconds"] for row in rows),
            "serial_output_tokens_per_second": document["measurement_windows"][workload]["serial_output_tokens_per_second"],
            "requests_per_second": document["measurement_windows"][workload]["requests_per_second"],
            "actual_output_tokens": sum(row["completion_tokens"] for row in rows),
            "successes": len(rows),
        }
    expected_strategy = (
        "DIRECT_BF16_ROW"
        if document["implementation"] == "candidate"
        else "ROW"
    )
    measured = [row for row in document["results"] if row.get("phase") == "measurement"]
    quality = [row for row in document["results"] if row.get("quality_pass")]
    cancel = [row for row in document["results"] if row.get("phase") == "cancel"]
    expected_quality = 3 if document["block"] == "A" else 4
    cancel_pass = (
        len(cancel) == 1
        and cancel[0].get("cancel_status") == "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"
    ) if document["block"] == "A" else not cancel
    monitor = read_jsonl(root / "monitor.jsonl")
    return {
        "version": document["version"],
        "implementation": document["implementation"],
        "block": document["block"],
        "path": str(path),
        "metrics": metrics,
        "qsa_complete": all(row.get("qsa", {}).get("decode_strategy") == expected_strategy for row in measured),
        "ple_complete": all(row.get("ple", {}).get("async", 0) > 0 for row in measured),
        "quality_complete": len(quality) == expected_quality and cancel_pass,
        "resource": {
            "whole_card_hbm_peak_mib": max((row.get("hbm_used_mib", 0) for row in monitor), default=None),
            "owned_hbm_peak_mib": max((row.get("owned_process_mib", 0) for row in monitor), default=None),
            "host_cgroup_peak_gib": cgroup_peak_gib(monitor),
        },
    }


def ratios(candidate, baseline):
    result = {}
    for workload in WORKLOADS:
        candidate_row = candidate["metrics"][workload]
        baseline_row = baseline["metrics"][workload]
        result[workload] = {
            "e2e": candidate_row["e2e_seconds_median"] / baseline_row["e2e_seconds_median"],
            "ttft": candidate_row["ttft_seconds_median"] / baseline_row["ttft_seconds_median"],
            "tpot": candidate_row["tpot_seconds_median"] / baseline_row["tpot_seconds_median"],
            "serial_output_tokens_per_second": candidate_row["serial_output_tokens_per_second"] / baseline_row["serial_output_tokens_per_second"],
            "requests_per_second": candidate_row["requests_per_second"] / baseline_row["requests_per_second"],
        }
    return result


def compare(runs):
    by = {(run["implementation"], run["block"]): run for run in runs}
    required = {
        ("baseline", "A"), ("candidate", "A"),
        ("candidate", "B"), ("baseline", "B"),
    }
    missing = sorted(f"{implementation}:{block}" for implementation, block in required - set(by))
    if missing:
        return {
            "schema": 1,
            "status": "F_PARTIAL_WITH_GAPS",
            "missing": missing,
            "runs": runs,
            "promoted": False,
        }
    block_ratios = {
        "A": ratios(by[("candidate", "A")], by[("baseline", "A")]),
        "B": ratios(by[("candidate", "B")], by[("baseline", "B")]),
    }
    primary = [block_ratios[block]["W2"]["tpot"] for block in ("A", "B")]
    regressions = []
    for workload in WORKLOADS:
        for metric in ("e2e", "ttft"):
            values = [block_ratios[block][workload][metric] for block in ("A", "B")]
            if all(value > 1.05 for value in values):
                regressions.append({"workload": workload, "metric": metric, "ratios": values})
        values = [
            block_ratios[block][workload]["serial_output_tokens_per_second"]
            for block in ("A", "B")
        ]
        if all(value < 0.95 for value in values):
            regressions.append({
                "workload": workload,
                "metric": "serial_output_tokens_per_second",
                "ratios": values,
            })
    quality = all(
        run["quality_complete"] and run["qsa_complete"] and run["ple_complete"]
        for run in runs
    )
    promoted = all(value <= 0.95 for value in primary) and not regressions and quality
    return {
        "schema": 1,
        "status": "F_FINAL_PROMOTED" if promoted else "F_FINAL_REJECTED",
        "runs": runs,
        "block_ratios": block_ratios,
        "primary_w2_tpot_ratios": primary,
        "protection_regressions": regressions,
        "quality_pass": quality,
        "promoted": promoted,
        "gate": {
            "primary": "W2 TPOT candidate/baseline <= 0.95 in both blocks",
            "protection": "No workload has TTFT/E2E >1.05 in both blocks or serial throughput <0.95 in both blocks",
        },
    }


def write_csv(path, runs):
    fields = [
        "line", "version", "implementation", "block", "workload",
        "e2e_seconds_median", "ttft_seconds_median", "tpot_seconds_median",
        "serial_output_tokens_per_second", "requests_per_second",
        "actual_output_tokens", "successes", "owned_hbm_peak_mib",
        "whole_card_hbm_peak_mib", "host_cgroup_peak_gib",
    ]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for workload in WORKLOADS:
                writer.writerow({
                    "line": "F",
                    "version": run["version"],
                    "implementation": run["implementation"],
                    "block": run["block"],
                    "workload": workload,
                    **run["metrics"][workload],
                    **run["resource"],
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    requests = [summarize_run(path) for path in sorted(args.root.rglob("round13-f-final.json"))]
    result = compare(requests)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(args.csv, requests)
    print(result["status"], result.get("promoted"), len(requests), flush=True)


if __name__ == "__main__":
    main()
