#!/usr/bin/env python3
"""Validate frozen Round4A3 contract against held-out matched-history metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    metrics = json.loads(args.metrics.read_text())
    epsilon = contract["candidate_logit_error_epsilon"]
    threshold = epsilon * contract["stable_margin_multiplier"]
    stable_failures = []
    near_tie_failures = []
    tie_sizes = []
    stable_count = near_count = 0
    for row in metrics["rows"]:
        tie_set = [
            token_id
            for token_id, logit in zip(row["baseline_top16_ids"], row["baseline_top16_logits"])
            if row["baseline_top1_logit"] - logit <= threshold + 1e-12
        ]
        record = {"prompt_id": row["prompt_id"], "token_index": row["token_index"], "margin": row["baseline_margin"], "baseline_top1": row["baseline_top1_id"], "hybrid_top1": row["hybrid_top1"], "tie_set": tie_set}
        if row["baseline_margin"] > threshold:
            stable_count += 1
            if row["hybrid_top1"] != row["baseline_top1_id"]:
                stable_failures.append(record)
        else:
            near_count += 1
            tie_sizes.append(len(tie_set))
            if row["hybrid_top1"] not in tie_set:
                near_tie_failures.append(record)
    overflows = {
        "candidate_error": metrics["metrics"]["candidate_max_error"]["max"] > contract["candidate_logit_error_max"],
        "max_abs": metrics["metrics"]["max_abs_logit_diff"]["max"] > contract["matched_history_max_abs_logit_error_max"],
        "relative_l2": metrics["metrics"]["relative_l2_logit"]["max"] > contract["matched_history_relative_l2_max"],
    }
    status = "HELDOUT_VERIFIED" if not stable_failures and not near_tie_failures and not any(overflows.values()) and metrics["all_finite"] and max(tie_sizes, default=1) <= contract["max_tie_set_size"] else "HELDOUT_CONTRACT_FAILURE"
    payload = {
        "schema_version": 1,
        "contract_sha256": contract["sha256"],
        "metrics_sha256": metrics["sha256"],
        "stable_tokens": stable_count,
        "near_tie_tokens": near_count,
        "near_tie_ratio": near_count / metrics["row_count"],
        "stable_failures": stable_failures,
        "near_tie_failures": near_tie_failures,
        "max_tie_set_size": max(tie_sizes, default=1),
        "envelope_overflows": overflows,
        "all_finite": metrics["all_finite"],
        "status": status,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if status != "HELDOUT_VERIFIED":
        raise SystemExit(status)


if __name__ == "__main__":
    main()
