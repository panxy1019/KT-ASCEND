#!/usr/bin/env python3
"""Compare completed Round13 A arms without inventing missing mirror data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


WORKLOADS = ("W0", "W1", "W2", "W3", "W4")
STRATEGIES = ("uniform", "frequency", "front_loading", "random_seed42")


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def cgroup_peak_gib(rows):
    values = []
    for row in rows:
        cgroup = row.get("owned_cgroup") or {}
        raw = cgroup.get("memory.max_usage_in_bytes")
        if raw is not None:
            values.append(int(raw.strip()))
    return max(values) / 2**30 if values else None


def summarize_run(final_path):
    final_path = Path(final_path)
    root = final_path.parent
    document = json.loads(final_path.read_text())
    if document.get("status") != "PASS_ROUND13_A_MODEL_PROCESS":
        raise RuntimeError(f"non-PASS A result {final_path}")
    if document.get("generation_attempts") != 24:
        raise RuntimeError(f"A result budget mismatch {final_path}")
    metrics = {}
    for workload in WORKLOADS:
        rows = [
            row
            for row in document["results"]
            if row.get("workload") == workload
            and row.get("phase") == "measurement"
        ]
        if len(rows) != 3:
            raise RuntimeError(f"{final_path} {workload} lacks three measurements")
        metrics[workload] = {
            "e2e_seconds_median": statistics.median(
                row["e2e_seconds"] for row in rows
            ),
            "ttft_seconds_median": statistics.median(
                row["ttft_seconds"] for row in rows
            ),
            "tpot_seconds_median": statistics.median(
                row["tpot_seconds"] for row in rows
            ),
            "serial_output_tokens_per_second": document["measurement_windows"][
                workload
            ]["serial_output_tokens_per_second"],
            "requests_per_second": document["measurement_windows"][workload][
                "requests_per_second"
            ],
            "actual_output_tokens": sum(row["completion_tokens"] for row in rows),
            "successes": len(rows),
        }
    monitor = read_jsonl(root / "monitor.jsonl")
    quality = [row for row in document["results"] if row.get("quality_pass")]
    cancel = [row for row in document["results"] if row.get("phase") == "cancel"]
    physical = document.get("physical") or json.loads(
        (root / "q32-physical.json").read_text()
    )
    return {
        "version": document["version"],
        "strategy": document["strategy"],
        "block": document["block"],
        "path": str(final_path),
        "metrics": metrics,
        "quality_pass_count": len(quality),
        "cancel_pass": (
            len(cancel) == 1
            and cancel[0].get("cancel_status")
            == "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"
        )
        if document["block"] == "A"
        else True,
        "qsa_complete": all(
            row.get("qsa", {}).get("vector_layers") == list(range(3, 48, 4))
            and row.get("qsa", {}).get("row_layers") == list(range(3, 48, 4))
            for row in document["results"]
            if row.get("phase") != "cancel"
        ),
        "ple_complete": all(
            row.get("ple", {}).get("async", 0) > 0
            for row in document["results"]
            if row.get("phase") != "cancel"
        ),
        "physical": {
            "status": physical.get("status"),
            "total_experts": physical.get("total_experts"),
            "total_bytes": physical.get("total_bytes"),
            "per_layer_counts": physical.get("per_layer_counts"),
        },
        "resource": {
            "whole_card_hbm_peak_mib": max(
                (row.get("hbm_used_mib", 0) for row in monitor), default=None
            ),
            "owned_hbm_peak_mib": max(
                (row.get("owned_process_mib", 0) for row in monitor), default=None
            ),
            "host_cgroup_peak_gib": cgroup_peak_gib(monitor),
        },
    }


def ratios(candidate, baseline):
    result = {}
    for workload in WORKLOADS:
        c = candidate["metrics"][workload]
        b = baseline["metrics"][workload]
        result[workload] = {
            "e2e": c["e2e_seconds_median"] / b["e2e_seconds_median"],
            "ttft": c["ttft_seconds_median"] / b["ttft_seconds_median"],
            "tpot": c["tpot_seconds_median"] / b["tpot_seconds_median"],
            "serial_output_tokens_per_second": (
                c["serial_output_tokens_per_second"]
                / b["serial_output_tokens_per_second"]
            ),
            "requests_per_second": (
                c["requests_per_second"] / b["requests_per_second"]
            ),
        }
    return result


def compare(runs):
    original_runs = list(runs)
    selected = {}
    for run in original_runs:
        key = (run["strategy"], run["block"])
        if key not in selected or run["version"] > selected[key]["version"]:
            selected[key] = run
    runs = sorted(selected.values(), key=lambda run: run["version"])
    selected_versions = {run["version"] for run in runs}
    superseded = [
        run for run in original_runs if run["version"] not in selected_versions
    ]
    by = {(run["strategy"], run["block"]): run for run in runs}
    candidates = {}
    for strategy in STRATEGIES[1:]:
        missing = [
            f"{name}:{block}"
            for name, block in (
                ("uniform", "A"),
                (strategy, "A"),
                (strategy, "B"),
                ("uniform", "B"),
            )
            if (name, block) not in by
        ]
        if missing:
            observed = (
                ratios(by[(strategy, "A")], by[("uniform", "A")])
                if ("uniform", "A") in by and (strategy, "A") in by
                else None
            )
            candidates[strategy] = {
                "status": "INCOMPLETE_PAIR",
                "missing": missing,
                "observed_first_block_ratios": observed,
                "observed_first_block_primary_w2_tpot_pass": (
                    observed["W2"]["tpot"] <= 0.95 if observed else None
                ),
                "promotion_eligible": False,
            }
            continue
        block_ratios = {
            "A": ratios(by[(strategy, "A")], by[("uniform", "A")]),
            "B": ratios(by[(strategy, "B")], by[("uniform", "B")]),
        }
        primary = [block_ratios[block]["W2"]["tpot"] for block in ("A", "B")]
        primary_pass = all(value <= 0.95 for value in primary)
        regressions = []
        for workload in WORKLOADS:
            for metric in ("e2e", "ttft"):
                values = [
                    block_ratios[block][workload][metric] for block in ("A", "B")
                ]
                if all(value > 1.05 for value in values):
                    regressions.append(
                        {"workload": workload, "metric": metric, "ratios": values}
                    )
            values = [
                block_ratios[block][workload]["serial_output_tokens_per_second"]
                for block in ("A", "B")
            ]
            if all(value < 0.95 for value in values):
                regressions.append(
                    {
                        "workload": workload,
                        "metric": "serial_output_tokens_per_second",
                        "ratios": values,
                    }
                )
        run_quality = all(
            by[(strategy, block)]["qsa_complete"]
            and by[(strategy, block)]["ple_complete"]
            and by[(strategy, block)]["quality_pass_count"]
            == (3 if block == "A" else 4)
            and by[(strategy, block)]["physical"]["total_experts"] == 1536
            and by[(strategy, block)]["physical"]["total_bytes"] == 15099494400
            for block in ("A", "B")
        )
        candidates[strategy] = {
            "status": "MEASURED_TWO_BLOCKS",
            "block_ratios": block_ratios,
            "primary_w2_tpot_ratios": primary,
            "primary_pass": primary_pass,
            "protection_regressions": regressions,
            "quality_and_budget_pass": run_quality,
            "promotion_eligible": primary_pass and not regressions and run_quality,
        }
    eligible = [name for name, row in candidates.items() if row["promotion_eligible"]]
    promoted = None
    if eligible:
        promoted = min(
            eligible,
            key=lambda name: statistics.mean(
                candidates[name]["primary_w2_tpot_ratios"]
            ),
        )
    complete = all(row["status"] == "MEASURED_TWO_BLOCKS" for row in candidates.values())
    all_first_blocks_reject = all(
        row.get("observed_first_block_primary_w2_tpot_pass") is False
        for row in candidates.values()
    )
    observed_ranking = sorted(
        (
            {
                "strategy": name,
                "w2_tpot_ratio": row["observed_first_block_ratios"]["W2"]["tpot"],
            }
            for name, row in candidates.items()
            if row.get("observed_first_block_ratios") is not None
        ),
        key=lambda row: row["w2_tpot_ratio"],
    )
    return {
        "schema": 1,
        "status": (
            "A_PROMOTED" if promoted else "A_NO_PROMOTION"
        )
        if complete
        else (
            "A_NO_PROMOTION_WITH_MIRROR_GAPS"
            if all_first_blocks_reject
            else "A_PARTIAL_WITH_GAPS"
        ),
        "runs": runs,
        "superseded_pre_common_source_runs": superseded,
        "candidates": candidates,
        "promoted": promoted,
        "full_four_strategy_coverage": complete,
        "all_strategies_single_block_measured": (
            ("uniform", "A") in by
            and all(row.get("observed_first_block_ratios") is not None for row in candidates.values())
        ),
        "all_first_blocks_reject_primary_gate": all_first_blocks_reject,
        "single_block_w2_tpot_ranking": observed_ranking,
        "shared_uniform_control": True,
        "gate": {
            "primary": "W2 TPOT candidate/baseline <= 0.95 in both blocks",
            "protection": "No workload has TTFT/E2E >1.05 in both blocks or serial throughput <0.95 in both blocks",
        },
    }


def write_csv(path, runs):
    fields = [
        "line",
        "version",
        "strategy",
        "block",
        "workload",
        "e2e_seconds_median",
        "ttft_seconds_median",
        "tpot_seconds_median",
        "serial_output_tokens_per_second",
        "requests_per_second",
        "actual_output_tokens",
        "successes",
        "owned_hbm_peak_mib",
        "whole_card_hbm_peak_mib",
        "host_cgroup_peak_gib",
    ]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for workload in WORKLOADS:
                writer.writerow(
                    {
                        "line": "A",
                        "version": run["version"],
                        "strategy": run["strategy"],
                        "block": run["block"],
                        "workload": workload,
                        **run["metrics"][workload],
                        **run["resource"],
                    }
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.root.rglob("round13-a-final.json"))
    if not paths:
        raise RuntimeError("no Round13 A final results")
    result = compare([summarize_run(path) for path in paths])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(args.csv, result["runs"])
    print(result["status"], result["promoted"], len(result["runs"]), flush=True)


if __name__ == "__main__":
    main()
