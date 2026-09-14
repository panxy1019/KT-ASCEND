#!/usr/bin/env python3
"""Compare all-NPU and Hybrid matrices for exact tokens, prefixes and logprobs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def index_cases(data: dict) -> dict[tuple[str, int], dict]:
    return {
        (case["prompt_id"], int(case["max_new_tokens"])): case
        for case in data["cases"]
    }


def prefix_failures(data: dict) -> list[dict]:
    indexed = index_cases(data)
    prompt_ids = sorted({key[0] for key in indexed})
    failures = []
    for prompt_id in prompt_ids:
        longest = max(count for pid, count in indexed if pid == prompt_id)
        full = indexed[(prompt_id, longest)]["output_ids"]
        for pid, count in sorted(indexed):
            if pid == prompt_id and indexed[(pid, count)]["output_ids"] != full[:count]:
                failures.append({"prompt_id": prompt_id, "count": count, "longest": longest})
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--hybrid", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-logprob-diff", default=0.20, type=float)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    hybrid = json.loads(args.hybrid.read_text(encoding="utf-8"))
    if baseline["corpus_sha256"] != hybrid["corpus_sha256"]:
        raise AssertionError("A/B corpus SHA256 differs")
    a_cases = index_cases(baseline)
    b_cases = index_cases(hybrid)
    if set(a_cases) != set(b_cases):
        raise AssertionError("A/B case keys differ")
    token_mismatches = []
    max_diff = 0.0
    max_location = None
    all_finite = True
    for key in sorted(a_cases):
        a_case, b_case = a_cases[key], b_cases[key]
        if a_case["output_ids"] != b_case["output_ids"]:
            token_mismatches.append({"prompt_id": key[0], "max_new_tokens": key[1]})
        for step, (a_value, b_value) in enumerate(zip(a_case["output_logprobs"], b_case["output_logprobs"])):
            all_finite = all_finite and math.isfinite(a_value) and math.isfinite(b_value)
            difference = abs(a_value - b_value)
            if difference > max_diff:
                max_diff = difference
                max_location = {"prompt_id": key[0], "max_new_tokens": key[1], "step": step}
    a_prefix = prefix_failures(baseline)
    b_prefix = prefix_failures(hybrid)
    result = {
        "request_count": len(a_cases),
        "token_exact_count": len(a_cases) - len(token_mismatches),
        "token_mismatches": token_mismatches,
        "baseline_prefix_failures": a_prefix,
        "hybrid_prefix_failures": b_prefix,
        "prefix_deterministic": not a_prefix and not b_prefix,
        "all_finite": all_finite,
        "max_abs_logprob_diff": max_diff,
        "max_abs_logprob_diff_location": max_location,
        "logprob_budget": args.max_logprob_diff,
    }
    result["status"] = (
        "PASS"
        if not token_mismatches and not a_prefix and not b_prefix and all_finite and max_diff <= args.max_logprob_diff
        else "FAIL"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
