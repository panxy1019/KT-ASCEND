#!/usr/bin/env python3
"""Classify free-generation first divergences with a frozen margin contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--matched-metrics", type=Path, required=True)
    parser.add_argument("--baseline-generation", type=Path, required=True)
    parser.add_argument("--hybrid-generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    metrics = json.loads(args.matched_metrics.read_text())
    baseline = json.loads(args.baseline_generation.read_text())
    hybrid = json.loads(args.hybrid_generation.read_text())
    metric_rows = {(row["prompt_id"], row["token_index"]): row for row in metrics["rows"]}
    hybrid_rows = {row["prompt_id"]: row for row in hybrid["rows"]}
    threshold = contract["candidate_logit_error_epsilon"] * contract["stable_margin_multiplier"]
    rows = []
    failures = []
    for base_row in baseline["rows"]:
        prompt_id = base_row["prompt_id"]
        base_ids = base_row["repetitions"][0]["output_ids"]
        hybrid_ids = hybrid_rows[prompt_id]["repetitions"][0]["output_ids"]
        first = next((index for index, pair in enumerate(zip(base_ids, hybrid_ids)) if pair[0] != pair[1]), None)
        record = {"prompt_id": prompt_id, "first_divergence_token": first, "status": "EXACT_TRAJECTORY"}
        if first is not None:
            metric = metric_rows[(prompt_id, first)]
            tie_set = [
                token_id
                for token_id, logit in zip(metric["baseline_top16_ids"], metric["baseline_top16_logits"])
                if metric["baseline_top1_logit"] - logit <= threshold + 1e-12
            ]
            near_tie = metric["baseline_margin"] <= threshold
            accepted = near_tie and int(hybrid_ids[first]) in tie_set
            record.update(
                {
                    "baseline_token": int(base_ids[first]),
                    "hybrid_token": int(hybrid_ids[first]),
                    "baseline_margin": metric["baseline_margin"],
                    "classification": "NEAR_TIE" if near_tie else "STABLE",
                    "tie_set": tie_set,
                    "status": "PASS_NEAR_TIE" if accepted else "FAIL",
                }
            )
            if not accepted:
                failures.append(record)
        rows.append(record)
    payload = {
        "schema_version": 1,
        "contract_sha256": contract["sha256"],
        "matched_metrics_sha256": metrics["sha256"],
        "requests": len(rows),
        "first_divergences": sum(row["first_divergence_token"] is not None for row in rows),
        "stable_region_divergences": sum(row.get("classification") == "STABLE" for row in rows),
        "near_tie_divergences": sum(row.get("classification") == "NEAR_TIE" for row in rows),
        "failures": failures,
        "rows": rows,
        "status": "PASS" if not failures else "FAIL",
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if failures:
        raise SystemExit("free-generation contract failure")


if __name__ == "__main__":
    main()
