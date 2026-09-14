#!/usr/bin/env python3
"""Derive epsilon/C/tie sets from Q only and freeze a candidate contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def evaluate(rows: list[dict], epsilon: float, multiplier: int) -> dict:
    stable = []
    near_tie = []
    for row in rows:
        threshold = multiplier * epsilon
        tie_set = [
            token_id
            for token_id, logit in zip(row["baseline_top16_ids"], row["baseline_top16_logits"])
            if row["baseline_top1_logit"] - logit <= threshold + 1e-12
        ]
        item = {
            "prompt_id": row["prompt_id"],
            "token_index": row["token_index"],
            "baseline_margin": row["baseline_margin"],
            "baseline_top1_id": row["baseline_top1_id"],
            "hybrid_top1": row["hybrid_top1"],
            "tie_set": tie_set,
            "tie_set_size": len(tie_set),
        }
        (stable if row["baseline_margin"] > threshold else near_tie).append(item)
    stable_exact = sum(item["hybrid_top1"] == item["baseline_top1_id"] for item in stable)
    tie_pass = sum(item["hybrid_top1"] in item["tie_set"] for item in near_tie)
    return {
        "C": multiplier,
        "threshold": multiplier * epsilon,
        "stable_tokens": len(stable),
        "near_tie_tokens": len(near_tie),
        "near_tie_ratio": len(near_tie) / len(rows),
        "stable_top1_exact_count": stable_exact,
        "stable_top1_exact_ratio": stable_exact / len(stable) if stable else 1.0,
        "tie_set_pass_count": tie_pass,
        "tie_set_pass_ratio": tie_pass / len(near_tie) if near_tie else 1.0,
        "max_tie_set_size": max((item["tie_set_size"] for item in near_tie), default=1),
        "tie_set_size_gt5_count": sum(item["tie_set_size"] > 5 for item in near_tie),
        "stable_failures": [item for item in stable if item["hybrid_top1"] != item["baseline_top1_id"]],
        "near_tie_failures": [item for item in near_tie if item["hybrid_top1"] not in item["tie_set"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-metrics", type=Path, required=True)
    parser.add_argument("--expert-rel-l2-max", type=float, default=1e-2)
    parser.add_argument("--output-analysis", type=Path, required=True)
    parser.add_argument("--output-contract", type=Path, required=True)
    args = parser.parse_args()
    q = json.loads(args.q_metrics.read_text())
    epsilon = q["metrics"]["candidate_max_error"]["p99"]
    sensitivity = [evaluate(q["rows"], epsilon, multiplier) for multiplier in (1, 2, 3, 4)]
    eligible = [
        row for row in sensitivity
        if row["stable_top1_exact_ratio"] == 1.0
        and row["tie_set_pass_ratio"] == 1.0
        and row["max_tie_set_size"] <= 8
        and row["tie_set_size_gt5_count"] / len(q["rows"]) <= 0.01
    ]
    if not eligible:
        selected = None
        status = "REJECTED_Q"
    else:
        selected = eligible[0]
        status = "Q_QUALIFIED_CANDIDATE_FROZEN"
    analysis = {
        "schema_version": 1,
        "derivation_source": "Q distribution only",
        "q_metrics_sha256": q["sha256"],
        "epsilon_rule": "p99(candidate-set max absolute logit error)",
        "epsilon_logit": epsilon,
        "sensitivity": sensitivity,
        "selection_rule": "smallest C in {1,2,3,4} with 100% stable exact, 100% tie-set membership, max tie-set size <=8, and <=1% of all positions with tie-set size >5",
        "selected": selected,
        "status": status,
    }
    canonical = json.dumps(analysis, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    analysis["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if selected is None:
        raise SystemExit("Q does not admit a candidate contract")
    contract = {
        "version": "r4a3-v1-candidate",
        "state": "Q_DERIVED_FROZEN_PENDING_HELDOUT",
        "expert_rel_l2_max": args.expert_rel_l2_max,
        "candidate_logit_error_epsilon": epsilon,
        "candidate_logit_error_max": q["metrics"]["candidate_max_error"]["max"],
        "matched_history_max_abs_logit_error_max": q["metrics"]["max_abs_logit_diff"]["max"],
        "matched_history_relative_l2_max": q["metrics"]["relative_l2_logit"]["max"],
        "stable_margin_multiplier": selected["C"],
        "tie_set_rule": "baseline_top1_minus_candidate_le_C_times_epsilon",
        "stable_top1_required": True,
        "same_path_determinism_required": True,
        "near_tie_membership_required": True,
        "quality_gate": "no_statistically_meaningful_regression",
        "max_tie_set_size": selected["max_tie_set_size"],
        "tie_set_size_gt5_ratio_max": 0.01,
        "corpus_q_sha256": q["corpus_sha256"],
        "q_metrics_sha256": q["sha256"],
        "derivation_analysis_sha256": analysis["sha256"],
    }
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    contract["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output_contract.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
