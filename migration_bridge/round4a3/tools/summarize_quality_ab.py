#!/usr/bin/env python3
"""Compare paired All-NPU/Hybrid quality scores with a frozen bootstrap CI."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def percentile(sorted_values: list[float], q: float) -> float:
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap(deltas: list[int], iterations: int = 10000) -> tuple[float, float]:
    rng = random.Random(0)
    count = len(deltas)
    samples = sorted(sum(deltas[rng.randrange(count)] for _ in range(count)) / count for _ in range(iterations))
    return percentile(samples, 0.025), percentile(samples, 0.975)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    hybrid = json.loads(args.hybrid.read_text())
    if baseline["quality_manifest_sha256"] != hybrid["quality_manifest_sha256"]:
        raise RuntimeError("quality manifest mismatch")
    base_rows = {(row["benchmark"], row["sample_id"]): row for row in baseline["rows"]}
    hybrid_rows = {(row["benchmark"], row["sample_id"]): row for row in hybrid["rows"]}
    if base_rows.keys() != hybrid_rows.keys():
        raise RuntimeError("sample mismatch")
    summaries = {}
    for benchmark in sorted({key[0] for key in base_rows}):
        keys = sorted(key for key in base_rows if key[0] == benchmark)
        deltas = [int(hybrid_rows[key]["correct"]) - int(base_rows[key]["correct"]) for key in keys]
        lower, upper = paired_bootstrap(deltas)
        npu_score = sum(base_rows[key]["correct"] for key in keys) / len(keys)
        hybrid_score = sum(hybrid_rows[key]["correct"] for key in keys) / len(keys)
        significant_regression = upper < 0
        summaries[benchmark] = {
            "sample_size": len(keys),
            "npu_score": npu_score,
            "hybrid_score": hybrid_score,
            "absolute_delta": hybrid_score - npu_score,
            "relative_delta": (hybrid_score - npu_score) / npu_score if npu_score else None,
            "paired_bootstrap_95_ci": [lower, upper],
            "bootstrap_iterations": 10000,
            "significant_regression": significant_regression,
            "npu_invalid": sum(base_rows[key]["invalid"] for key in keys),
            "hybrid_invalid": sum(hybrid_rows[key]["invalid"] for key in keys),
        }
    invalid_explosion = any(row["hybrid_invalid"] > row["npu_invalid"] + max(2, row["sample_size"] // 10) for row in summaries.values())
    significant_regression = any(row["significant_regression"] for row in summaries.values())
    payload = {
        "schema_version": 1,
        "baseline_sha256": baseline["sha256"],
        "hybrid_sha256": hybrid["sha256"],
        "method": "paired nonparametric bootstrap, deterministic seed 0",
        "summaries": summaries,
        "invalid_output_explosion": invalid_explosion,
        "status": "QUALITY_REGRESSION" if significant_regression or invalid_explosion else "NO_STATISTICALLY_MEANINGFUL_REGRESSION",
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if payload["status"] != "NO_STATISTICALLY_MEANINGFUL_REGRESSION":
        raise SystemExit(payload["status"])


if __name__ == "__main__":
    main()
