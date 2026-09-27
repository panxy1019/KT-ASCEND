#!/usr/bin/env python3
"""Freeze the two-block Round13 B QSA Decode component gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_CHECKS = {
    "per_query_causal_future_rejected": True,
    "unqualified_geometry_strategy": "ROW",
    "unqualified_geometry_reference": True,
    "complete_batch_drained": True,
    "cancel_drained": True,
    "stale_completion_rejected": True,
    "stale_completion_fail_stop": True,
    "runtime_ready_after_cancel": True,
}


def summarize(path):
    path = Path(path)
    document = json.loads(path.read_text())
    rows = document.get("rows", [])
    checks = document.get("contract_checks", {})
    return {
        "path": str(path),
        "block": document.get("block"),
        "status": document.get("status"),
        "row_count": len(rows),
        "ratios": [row.get("candidate_over_baseline") for row in rows],
        "worst_candidate_over_baseline": max(
            (row.get("candidate_over_baseline", float("inf")) for row in rows),
            default=None,
        ),
        "numerical_pass": all(
            row.get("baseline_relative_l2", float("inf")) <= 0.001
            and row.get("candidate_relative_l2", float("inf")) <= 0.001
            and len(set(row.get("repeat_hashes", []))) == 1
            for row in rows
        ),
        "contract_checks_pass": all(checks.get(key) == value for key, value in REQUIRED_CHECKS.items()),
        "allocator_cleanup_pass": document.get("after_allocated")
        == document.get("baseline_allocated")
        and document.get("after_reserved")
        == document.get("baseline_reserved"),
    }


def compare(blocks):
    by = {row["block"]: row for row in blocks}
    missing = sorted({"A", "B"} - set(by))
    if missing:
        return {
            "schema": 1,
            "status": "ROUND13_B_COMPONENT_PARTIAL",
            "missing": missing,
            "blocks": blocks,
            "request_gate_open": False,
        }
    correctness = all(
        row["status"] == "PASS_ROUND13_B_QSA_DECODE_COMPONENT"
        and row["row_count"] == 9
        and row["numerical_pass"]
        and row["contract_checks_pass"]
        and row["allocator_cleanup_pass"]
        for row in by.values()
    )
    performance = all(
        len(row["ratios"]) == 9
        and all(isinstance(ratio, (int, float)) and ratio <= 0.90 for ratio in row["ratios"])
        for row in by.values()
    )
    return {
        "schema": 1,
        "status": (
            "PASS_ROUND13_B_COMPONENT_GATE"
            if correctness and performance
            else "ROUND13_B_COMPONENT_NO_PROMOTION"
        ),
        "blocks": [by["A"], by["B"]],
        "correctness_pass": correctness,
        "all_complete_calls_improve_10pct": performance,
        "request_gate_open": correctness and performance,
        "threshold": "each of 9 full-call candidate/baseline medians <= 0.90 in both independent blocks",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare([summarize(path) for path in args.results])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["request_gate_open"], flush=True)


if __name__ == "__main__":
    main()
