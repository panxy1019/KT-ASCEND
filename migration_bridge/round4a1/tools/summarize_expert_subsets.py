#!/usr/bin/env python3
"""Summarize the fixed Round 4A1 15-subset sensitivity campaign."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch


SUBSETS = (
    ("e6", (6,)),
    ("e8", (8,)),
    ("e25", (25,)),
    ("e36", (36,)),
    ("e6_8", (6, 8)),
    ("e6_25", (6, 25)),
    ("e6_36", (6, 36)),
    ("e8_25", (8, 25)),
    ("e8_36", (8, 36)),
    ("e25_36", (25, 36)),
    ("e6_8_25", (6, 8, 25)),
    ("e6_8_36", (6, 8, 36)),
    ("e6_25_36", (6, 25, 36)),
    ("e8_25_36", (8, 25, 36)),
    ("e6_8_25_36", (6, 8, 25, 36)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--baseline-dumps", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def load_passes(path: Path) -> list[dict]:
    payloads = [torch.load(name, map_location="cpu") for name in sorted(glob.glob(str(path / "*.pt")))]
    payloads.sort(key=lambda item: int(item["pass"]))
    return payloads


def main() -> None:
    args = parse_args()
    baseline_matrix = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_case = next(
        case
        for case in baseline_matrix["cases"]
        if case["prompt_id"] == "v_struct_03" and case["max_new_tokens"] == 16
    )
    baseline_ids = baseline_case["output_ids"]
    baseline_logprobs = baseline_case["output_logprobs"]
    baseline_passes = load_passes(args.baseline_dumps)
    rows = []
    for name, experts in SUBSETS:
        response_path = args.workspace / f"round4a1-subset-{name}-response.json"
        dump_dir = args.workspace / f"round4a1-subset-{name}-dumps"
        response = json.loads(response_path.read_text(encoding="utf-8"))
        candidate_ids = response["output_ids"]
        mismatch = first_mismatch(baseline_ids, candidate_ids)
        candidate_logprobs = [
            float(item[0])
            for item in response["meta_info"]["output_token_logprobs"]
        ]
        same_history_steps = len(candidate_ids) if mismatch is None else mismatch + 1
        paired = min(same_history_steps, len(baseline_logprobs), len(candidate_logprobs))
        max_logprob_delta = max(
            (abs(float(baseline_logprobs[i]) - candidate_logprobs[i]) for i in range(paired)),
            default=0.0,
        )
        candidate_passes = load_passes(dump_dir)
        cpu_hits = 0
        sum_sq_difference = 0.0
        sum_sq_reference = 0.0
        routed_max_abs = 0.0
        exact_passes = 0
        compared_passes = 0
        for baseline_payload, candidate_payload in zip(baseline_passes, candidate_passes):
            pass_index = int(baseline_payload["pass"])
            if pass_index > (mismatch if mismatch is not None else len(candidate_ids)):
                break
            topk_ids = candidate_payload["topk_ids"]
            cpu_hits += sum(int((topk_ids == expert).sum()) for expert in experts)
            if not torch.equal(baseline_payload["input"], candidate_payload["hidden_states"]):
                break
            reference = baseline_payload["final"].float()
            actual = candidate_payload["merged_output"].float()
            difference = actual - reference
            sum_sq_difference += float(torch.sum(difference * difference))
            sum_sq_reference += float(torch.sum(reference * reference))
            routed_max_abs = max(routed_max_abs, float(difference.abs().max()))
            exact_passes += int(torch.equal(actual.to(torch.bfloat16), reference.to(torch.bfloat16)))
            compared_passes += 1
        rows.append(
            {
                "name": name,
                "cpu_experts": list(experts),
                "cpu_hit_count": cpu_hits,
                "first_divergent_token": mismatch,
                "baseline_token": baseline_ids[mismatch] if mismatch is not None else None,
                "hybrid_token": candidate_ids[mismatch] if mismatch is not None else None,
                "max_abs_selected_token_logprob_delta_same_history": max_logprob_delta,
                "routed_output_relative_l2_same_history": (
                    (sum_sq_difference / sum_sq_reference) ** 0.5
                    if sum_sq_reference
                    else 0.0
                ),
                "routed_output_max_abs_same_history": routed_max_abs,
                "routed_output_exact_passes": exact_passes,
                "routed_output_compared_passes": compared_passes,
            }
        )
    output = {
        "schema_version": 1,
        "prompt_id": "v_struct_03",
        "max_new_tokens": 16,
        "baseline_output_ids": baseline_ids,
        "subsets": rows,
        "limitations": [
            "API exposes selected-token/top-k logprobs, not the full logits tensor.",
            "Routed-output comparisons stop when Layer 17 inputs cease to be bitwise same-history.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
