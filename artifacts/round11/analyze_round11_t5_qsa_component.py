#!/usr/bin/env python3
"""Validate and disposition the single Round11 QSA component candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


LAYERS = (3, 23, 47)
PREFILL_CONTEXTS = (32, 64, 128)
NUMERICAL_GATE = 0.001
PREFILL_RATIO_GATE = 0.90
DECODE_RATIO_GATE = 1.05


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {label}")
    return float(value)


def analyze_component(document):
    if (
        document.get("status") != "PASS_ROUND11_T5_QSA_COMPONENT"
        or document.get("candidate") != "QUERY_AWARE_VECTORIZED_SPARSE_GQA"
    ):
        raise ValueError("invalid QSA component identity")
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise ValueError("QSA component rows missing")
    expected = {
        (layer, context, 32)
        for layer in LAYERS for context in PREFILL_CONTEXTS
    } | {(layer, 129, 1) for layer in LAYERS}
    by_key = {}
    for row in rows:
        key = (row.get("layer"), row.get("context_tokens"), row.get("query_tokens"))
        if key not in expected or key in by_key:
            raise ValueError("QSA component row coverage mismatch")
        baseline = _positive(row.get("baseline_median_ms"), "baseline timing")
        candidate = _positive(row.get("candidate_median_ms"), "candidate timing")
        ratio = _positive(row.get("candidate_over_baseline"), "candidate ratio")
        if not math.isclose(candidate / baseline, ratio, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("QSA component timing ratio mismatch")
        relative = row.get("relative_l2")
        if type(relative) not in (int, float) or not math.isfinite(relative) or relative > NUMERICAL_GATE:
            raise ValueError("QSA component numerical gate failed")
        hashes = row.get("candidate_repeat_hashes")
        if not isinstance(hashes, list) or len(hashes) != 3 or len(set(hashes)) != 1:
            raise ValueError("QSA component repeat hash failed")
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError("QSA component row coverage mismatch")
    prefill = {
        str(context): statistics.median(
            by_key[(layer, context, 32)]["candidate_over_baseline"]
            for layer in LAYERS
        )
        for context in PREFILL_CONTEXTS
    }
    decode = statistics.median(
        by_key[(layer, 129, 1)]["candidate_over_baseline"] for layer in LAYERS
    )
    promoted = all(value <= PREFILL_RATIO_GATE for value in prefill.values()) and decode <= DECODE_RATIO_GATE
    return {
        "schema": 1,
        "status": "PASS_ROUND11_T5_QSA_COMPONENT_ANALYSIS",
        "candidate": "QUERY_AWARE_VECTORIZED_SPARSE_GQA",
        "prefill_context_median_ratio": prefill,
        "decode_median_ratio": decode,
        "gates": {
            "relative_l2_max": NUMERICAL_GATE,
            "each_prefill_context_ratio_max": PREFILL_RATIO_GATE,
            "decode_ratio_max": DECODE_RATIO_GATE,
        },
        "disposition": (
            "QSA_COMPONENT_PROMOTE_TO_REQUEST"
            if promoted else "QSA_COMPONENT_ONLY_NO_PROMOTION"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze_component(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["disposition"], flush=True)


if __name__ == "__main__":
    main()
