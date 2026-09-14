#!/usr/bin/env python3
"""Classify free-running first divergences with a frozen pairwise contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def outputs(path: Path) -> tuple[dict, dict[str, list[int]]]:
    payload = json.loads(path.read_text())
    return payload, {row["prompt_id"]: [int(x) for x in row["repetitions"][0]["output_ids"]] for row in payload["rows"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-free", type=Path, required=True)
    parser.add_argument("--hybrid-free", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--corpus-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_payload, baseline = outputs(args.baseline_free)
    hybrid_payload, hybrid = outputs(args.hybrid_free)
    metrics = json.loads(args.metrics.read_text())
    contract = json.loads(args.contract.read_text())
    bound = float(contract["pairwise_margin_bound"])
    metric_rows = {(row["prompt_id"], row["token_index"]): row for row in metrics["rows"]}
    rows = []
    for prompt_id, baseline_ids in baseline.items():
        hybrid_ids = hybrid[prompt_id]
        first = next((i for i, pair in enumerate(zip(baseline_ids, hybrid_ids)) if pair[0] != pair[1]), None)
        if first is None:
            rows.append({"prompt_id": prompt_id, "first_divergence": None, "class": "EXACT", "pass": True})
            continue
        metric = metric_rows[(prompt_id, first)]
        ambiguity_ids = [metric["baseline_top1_id"]]
        ambiguity_ids.extend(
            token for token, margin in zip(metric["candidate_ids"], metric["baseline_pairwise_margins"])
            if margin <= bound + 1e-12
        )
        ambiguity_ids = list(dict.fromkeys(ambiguity_ids))
        chosen = hybrid_ids[first]
        margin_by_id = dict(zip(metric["candidate_ids"], metric["baseline_pairwise_margins"]))
        classification = "PAIRWISE_STABLE" if len(ambiguity_ids) == 1 else "PAIRWISE_AMBIGUOUS"
        passed = classification == "PAIRWISE_AMBIGUOUS" and chosen in ambiguity_ids
        rows.append({
            "prompt_id": prompt_id, "first_divergence": first,
            "baseline_token": baseline_ids[first], "hybrid_token": chosen,
            "baseline_pairwise_margin_to_hybrid": margin_by_id.get(chosen),
            "B_pair": bound, "class": classification, "ambiguity_ids": ambiguity_ids,
            "ambiguity_size": len(ambiguity_ids), "hybrid_token_in_ambiguity_set": chosen in ambiguity_ids,
            "pass": passed,
        })
    passed = all(row["pass"] for row in rows)
    payload = {
        "schema_version": 1, "corpus": args.corpus_label,
        "baseline_free_sha256": baseline_payload["sha256"], "hybrid_free_sha256": hybrid_payload["sha256"],
        "metrics_sha256": metrics["sha256"], "contract_sha256": contract["sha256"],
        "requests": len(rows), "divergences": sum(r["first_divergence"] is not None for r in rows),
        "stable_divergences": sum(r["class"] == "PAIRWISE_STABLE" for r in rows),
        "ambiguous_divergences": sum(r["class"] == "PAIRWISE_AMBIGUOUS" for r in rows),
        "membership_pass": all(r["hybrid_token_in_ambiguity_set"] for r in rows if r["first_divergence"] is not None),
        "rows": rows, "status": "A3_VERIFIED_READY" if passed else "BLOCKED",
    }
    payload["sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit("free-generation pairwise qualification failed")


if __name__ == "__main__":
    main()
