#!/usr/bin/env python3
"""Derive a Q2-only pairwise bound and freeze the candidate contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


SOURCES = ("p99", "p99_5", "p99_9", "max")
FACTORS = (1.0, 1.25, 1.5)


def canonical_sha(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def evaluate(rows: list[dict], bound: float) -> dict:
    stable, ambiguous = [], []
    for row in rows:
        ambiguous_ids = [row["baseline_top1_id"]]
        ambiguous_ids.extend(
            token for token, margin in zip(row["candidate_ids"], row["baseline_pairwise_margins"])
            if margin <= bound + 1e-12
        )
        ambiguous_ids = list(dict.fromkeys(ambiguous_ids))
        is_stable = len(ambiguous_ids) == 1
        accepted = row["hybrid_top1_id"] == row["baseline_top1_id"] if is_stable else row["hybrid_top1_id"] in ambiguous_ids
        item = {
            "prompt_id": row["prompt_id"], "token_index": row["token_index"],
            "baseline_top1_id": row["baseline_top1_id"], "hybrid_top1_id": row["hybrid_top1_id"],
            "class": "PAIRWISE_STABLE" if is_stable else "PAIRWISE_AMBIGUOUS",
            "ambiguity_ids": ambiguous_ids, "ambiguity_size": len(ambiguous_ids), "accepted": accepted,
            "pairwise_abs_max": row["pairwise_abs_max"],
        }
        (stable if is_stable else ambiguous).append(item)
    stable_exact = sum(item["accepted"] for item in stable)
    membership = sum(item["accepted"] for item in ambiguous)
    return {
        "B_pair": bound, "stable": len(stable), "ambiguous": len(ambiguous),
        "stable_ratio": len(stable) / len(rows), "ambiguous_ratio": len(ambiguous) / len(rows),
        "stable_exact_count": stable_exact, "stable_exact_ratio": stable_exact / len(stable) if stable else 1.0,
        "ambiguity_membership_count": membership,
        "ambiguity_membership_ratio": membership / len(ambiguous) if ambiguous else 1.0,
        "pairwise_overflow_count": sum(row["pairwise_abs_max"] > bound + 1e-12 for row in rows),
        "ambiguity_size_p95": float(torch.quantile(torch.tensor([i["ambiguity_size"] for i in ambiguous], dtype=torch.float64), .95)) if ambiguous else 1.0,
        "ambiguity_size_max": max((i["ambiguity_size"] for i in ambiguous), default=1),
        "stable_failures": [item for item in stable if not item["accepted"]],
        "ambiguity_failures": [item for item in ambiguous if not item["accepted"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q2-metrics", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    parser.add_argument("--output-contract", type=Path, required=True)
    args = parser.parse_args()
    q2 = json.loads(args.q2_metrics.read_text())
    distribution = q2["pairwise_abs_distortion"]
    sensitivity = []
    for source in SOURCES:
        for factor in FACTORS:
            result = evaluate(q2["rows"], distribution[source] * factor)
            result.update({"bound_source": source, "safety_factor": factor})
            sensitivity.append(result)
    # Predeclared selection: max observed Q2 pairwise distortion with 25% safety
    # reserve. This is Q2-derived, has zero Q2 overflow, and does not inspect H2.
    selected = next(row for row in sensitivity if row["bound_source"] == "max" and row["safety_factor"] == 1.25)
    valid = (
        selected["stable_exact_ratio"] == 1.0
        and selected["ambiguity_membership_ratio"] == 1.0
        and selected["pairwise_overflow_count"] == 0
        and selected["stable_ratio"] > 0
        and q2["all_finite"]
    )
    analysis = {
        "schema_version": 1, "derivation_source": "Q2 only",
        "selection_rule": "1.25 times the Q2 maximum pairwise absolute distortion; predeclared safety reserve; H2 never inspected",
        "q2_metrics_sha256": q2["sha256"], "sensitivity": sensitivity,
        "selected": selected, "status": "Q2_QUALIFIED_CANDIDATE_FROZEN" if valid else "REJECTED_Q2",
    }
    analysis["sha256"] = canonical_sha(analysis)
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if not valid:
        raise SystemExit("Q2 does not admit the predeclared pairwise candidate contract")
    contract = {
        "version": "round4a4-v1-candidate", "state": "Q2_DERIVED_FROZEN_PENDING_H2",
        "expert_rel_l2_max": 0.01, "pairwise_margin_bound": selected["B_pair"],
        "pairwise_bound_source": "1.25 * Q2 max pairwise absolute distortion",
        "candidate_topk": q2["candidate_top_k"], "stable_top1_exact_required": True,
        "ambiguous_membership_required": True, "ambiguity_cardinality_gate": False,
        "same_path_determinism_required": True, "cpu_not_hit_exact_required": True,
        "all_finite_required": True, "quality_gate": "no_statistically_meaningful_regression",
        "q2_sha256": q2["corpus_sha256"], "q2_metrics_sha256": q2["sha256"],
        "derivation_analysis_sha256": analysis["sha256"],
    }
    contract["sha256"] = canonical_sha(contract)
    args.output_contract.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
