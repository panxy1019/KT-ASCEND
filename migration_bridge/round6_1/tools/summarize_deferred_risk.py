#!/usr/bin/env python3
"""Descriptive-only Round 6.1 risk summary; never derives an acceptance bound."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - index) + ordered[upper] * (index - lower))


def describe(rows: list[dict]) -> dict:
    distortion = [float(row["pairwise_abs_max"]) for row in rows]
    max_abs = [float(row["full_logit_max_abs"]) for row in rows]
    rel_l2 = [float(row["full_logit_rel_l2"]) for row in rows]
    flips = [row for row in rows if row["deferred_top1_id"] != row["baseline_top1_id"]]
    return {
        "positions": len(rows),
        "top1_flips": len(flips),
        "top1_flip_positions": [
            {"prompt_id": row["prompt_id"], "token_index": row["token_index"]}
            for row in flips
        ],
        "pairwise_abs_max": {
            "p50": percentile(distortion, 0.50),
            "p95": percentile(distortion, 0.95),
            "p99": percentile(distortion, 0.99),
            "max": max(distortion) if distortion else None,
        },
        "full_logit_max_abs": {
            "p50": percentile(max_abs, 0.50),
            "p95": percentile(max_abs, 0.95),
            "p99": percentile(max_abs, 0.99),
            "max": max(max_abs) if max_abs else None,
        },
        "full_logit_rel_l2": {
            "p50": percentile(rel_l2, 0.50),
            "p95": percentile(rel_l2, 0.95),
            "p99": percentile(rel_l2, 0.99),
            "max": max(rel_l2) if rel_l2 else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qd", type=Path, required=True)
    parser.add_argument("--hd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "schema_version": "round6.1-descriptive-risk-v1",
        "purpose": "descriptive_only_no_contract_derivation",
        "corpora": {},
    }
    for role, path in (("QD", args.qd), ("HD", args.hd)):
        payload = json.loads(path.read_text())
        rows = payload["rows"]
        groups = {
            "all": rows,
            "deferred_route_count_0": [r for r in rows if r["actual_deferred_route_count"] == 0],
            "deferred_route_count_1plus": [r for r in rows if r["actual_deferred_route_count"] >= 1],
            "cpu_hit_count_0": [r for r in rows if r["cpu_hit_count"] == 0],
            "cpu_hit_count_1": [r for r in rows if r["cpu_hit_count"] == 1],
            "cpu_hit_count_2plus": [r for r in rows if r["cpu_hit_count"] >= 2],
        }
        result["corpora"][role] = {
            "source_metrics_sha256": payload.get("sha256"),
            "groups": {name: describe(group) for name, group in groups.items()},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({role: data["groups"]["all"] for role, data in result["corpora"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
