#!/usr/bin/env python3
"""Compare paired quality runs with a deterministic paired bootstrap interval."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    args = parser.parse_args()
    npu = json.loads(args.npu.read_text())
    hybrid = json.loads(args.hybrid.read_text())
    if npu["quality_manifest_sha256"] != hybrid["quality_manifest_sha256"]:
        raise RuntimeError("quality manifest mismatch")
    npu_rows = {(r["benchmark"], r["sample_id"]): r for r in npu["rows"]}
    hybrid_rows = {(r["benchmark"], r["sample_id"]): r for r in hybrid["rows"]}
    if npu_rows.keys() != hybrid_rows.keys():
        raise RuntimeError("paired row mismatch")
    rng = random.Random(0)
    results = {}
    for benchmark in sorted(npu["summaries"]):
        keys = sorted(key for key in npu_rows if key[0] == benchmark)
        paired = [(int(npu_rows[k]["correct"]), int(hybrid_rows[k]["correct"])) for k in keys]
        deltas = []
        for _ in range(args.bootstrap_repeats):
            sample = [paired[rng.randrange(len(paired))] for _ in paired]
            deltas.append(sum(h - n for n, h in sample) / len(sample))
        npu_score = sum(n for n, _ in paired) / len(paired)
        hybrid_score = sum(h for _, h in paired) / len(paired)
        lower, upper = percentile(deltas, 0.025), percentile(deltas, 0.975)
        results[benchmark] = {
            "count": len(paired),
            "npu_score": npu_score,
            "hybrid_score": hybrid_score,
            "absolute_delta": hybrid_score - npu_score,
            "paired_bootstrap_95ci": [lower, upper],
            "npu_correct_hybrid_wrong": sum(n == 1 and h == 0 for n, h in paired),
            "npu_wrong_hybrid_correct": sum(n == 0 and h == 1 for n, h in paired),
            "npu_invalid": sum(npu_rows[k]["invalid"] for k in keys),
            "hybrid_invalid": sum(hybrid_rows[k]["invalid"] for k in keys),
            "no_statistically_meaningful_regression": upper >= 0,
        }
    status = (
        "QUALITY_VERIFIED"
        if all(r["no_statistically_meaningful_regression"] for r in results.values())
        and all(r["hybrid_invalid"] <= r["npu_invalid"] for r in results.values())
        and npu["all_output_tokens_valid"]
        and hybrid["all_output_tokens_valid"]
        else "QUALITY_REGRESSION"
    )
    payload = {
        "schema_version": 1,
        "manifest_sha256": npu["quality_manifest_sha256"],
        "bootstrap_seed": 0,
        "bootstrap_repeats": args.bootstrap_repeats,
        "results": results,
        "status": status,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "QUALITY_VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
