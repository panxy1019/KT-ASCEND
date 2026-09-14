#!/usr/bin/env python3
"""Validate frozen pairwise contract without fitting H2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def canonical_sha(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--corpus-role",
        choices=("qualification", "heldout", "free", "p2", "p3"),
        default="heldout",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    contract = json.loads(args.contract.read_text())
    bound = float(contract["pairwise_margin_bound"])
    stable, ambiguous = [], []
    for row in metrics["rows"]:
        ambiguity_ids = [row["baseline_top1_id"]]
        ambiguity_ids.extend(
            token for token, margin in zip(row["candidate_ids"], row["baseline_pairwise_margins"])
            if margin <= bound + 1e-12
        )
        ambiguity_ids = list(dict.fromkeys(ambiguity_ids))
        probability_by_id = {row["baseline_top1_id"]: row["baseline_top1_probability"]}
        probability_by_id.update(zip(row["candidate_ids"], row["baseline_candidate_probabilities"]))
        item = {
            "prompt_id": row["prompt_id"], "token_index": row["token_index"],
            "baseline_top1_id": row["baseline_top1_id"], "hybrid_top1_id": row["hybrid_top1_id"],
            "ambiguity_ids": ambiguity_ids, "ambiguity_size": len(ambiguity_ids),
            "ambiguity_probability_mass": sum(probability_by_id[token] for token in ambiguity_ids),
        }
        (stable if len(ambiguity_ids) == 1 else ambiguous).append(item)
    stable_failures = [r for r in stable if r["hybrid_top1_id"] != r["baseline_top1_id"]]
    ambiguity_failures = [r for r in ambiguous if r["hybrid_top1_id"] not in r["ambiguity_ids"]]
    overflow = [
        {"prompt_id": r["prompt_id"], "token_index": r["token_index"], "pairwise_abs_max": r["pairwise_abs_max"]}
        for r in metrics["rows"] if r["pairwise_abs_max"] > bound + 1e-12
    ]
    sizes = torch.tensor([r["ambiguity_size"] for r in ambiguous], dtype=torch.float64) if ambiguous else torch.tensor([1.0])
    masses = torch.tensor([r["ambiguity_probability_mass"] for r in ambiguous], dtype=torch.float64) if ambiguous else torch.tensor([0.0])
    passed = not stable_failures and not ambiguity_failures and not overflow and metrics["all_finite"]
    payload = {
        "schema_version": 1, "contract_sha256": contract["sha256"],
        "corpus_sha256": metrics["corpus_sha256"], "B_pair": bound,
        "stable": len(stable), "stable_exact": len(stable) - len(stable_failures),
        "stable_failures": stable_failures, "ambiguous": len(ambiguous),
        "ambiguity_membership": len(ambiguous) - len(ambiguity_failures),
        "ambiguity_failures": ambiguity_failures, "pairwise_overflow_count": len(overflow),
        "pairwise_overflows": overflow, "ambiguity_size_p95": float(torch.quantile(sizes, .95)),
        "ambiguity_size_max": int(sizes.max()), "ambiguity_cardinality_gate": False,
        "ambiguity_mass_p95": float(torch.quantile(masses, .95)), "ambiguity_mass_max": float(masses.max()),
        "all_finite": metrics["all_finite"],
        "corpus_role": args.corpus_role,
        "status": (
            {
                "qualification": "Q2_QUALIFIED",
                "heldout": "HELDOUT_VERIFIED",
                "free": "FREE_NUMERICAL_VERIFIED",
                "p2": "P2_NUMERICAL_VERIFIED",
                "p3": "P3_NUMERICAL_VERIFIED",
            }[args.corpus_role]
            if passed
            else "REJECTED"
        ),
    }
    payload["sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit("H2 pairwise contract rejected")


if __name__ == "__main__":
    main()
