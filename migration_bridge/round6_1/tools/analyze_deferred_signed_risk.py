#!/usr/bin/env python3
"""Compute descriptive signed Deferred distortion statistics from frozen logits.

This tool deliberately does not derive, select, or validate an acceptance bound.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


QUANTILES = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99, "max": 1.0}


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in QUANTILES}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {key: float(torch.quantile(tensor, q)) for key, q in QUANTILES.items()}


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denominator == 0:
        return None
    return float(torch.dot(x, y) / denominator)


def margin_bucket(value: float) -> str:
    if value < 0.5:
        return "lt_0.5"
    if value < 2.0:
        return "0.5_to_2"
    if value < 8.0:
        return "2_to_8"
    return "8_plus"


def weight_bucket(value: float) -> str:
    if value == 0:
        return "0"
    if value < 0.1:
        return "0_to_0.1"
    if value < 0.25:
        return "0.1_to_0.25"
    return "0.25_plus"


def summarize(entries: list[dict]) -> dict:
    return {
        "pairs": len(entries),
        "signed_distortion": distribution([e["signed_distortion"] for e in entries]),
        "absolute_distortion": distribution([e["absolute_distortion"] for e in entries]),
        "adverse_contraction": distribution([e["adverse_contraction"] for e in entries]),
        "beneficial_expansion": distribution([e["beneficial_expansion"] for e in entries]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--deferred-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--historical-bound", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    rows_by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in metrics["rows"]:
        rows_by_prompt[row["prompt_id"]].append(row)

    entries = []
    position_summaries = []
    for prompt_id, rows in sorted(rows_by_prompt.items()):
        baseline = torch.load(args.baseline_dir / "logits" / f"{prompt_id}.pt", map_location="cpu", weights_only=True).float()
        deferred = torch.load(args.deferred_dir / "logits" / f"{prompt_id}.pt", map_location="cpu", weights_only=True).float()
        for row in sorted(rows, key=lambda value: value["token_index"]):
            token_index = int(row["token_index"])
            anchor = int(row["baseline_top1_id"])
            base_logits = baseline[token_index]
            deferred_logits = deferred[token_index]
            route_weights = [abs(float(value)) for value in row["cpu_route_weights"]]
            weight_sum = sum(route_weights)
            weight_max = max(route_weights, default=0.0)
            top1_changed = row["deferred_top1_id"] != row["baseline_top1_id"]
            ambiguity_members = {anchor}
            position_entries = []
            for candidate in row["candidate_ids"]:
                candidate = int(candidate)
                m0 = float(base_logits[anchor] - base_logits[candidate])
                m1 = float(deferred_logits[anchor] - deferred_logits[candidate])
                signed = m1 - m0
                if m0 <= args.historical_bound:
                    ambiguity_members.add(candidate)
                entry = {
                    "prompt_id": prompt_id,
                    "category": row["category"],
                    "token_index": token_index,
                    "candidate_id": candidate,
                    "m0": m0,
                    "m1": m1,
                    "signed_distortion": signed,
                    "absolute_distortion": abs(signed),
                    "adverse_contraction": max(0.0, -signed),
                    "beneficial_expansion": max(0.0, signed),
                    "baseline_margin_bucket": margin_bucket(m0),
                    "target_row_deferred_hit": int(row["actual_deferred_route_count"]) > 0,
                    "deferred_route_count": int(row["actual_deferred_route_count"]),
                    "target_row_cpu_route_weight_sum_proxy": weight_sum,
                    "target_row_cpu_route_weight_max_proxy": weight_max,
                    "top1_changed": top1_changed,
                }
                entries.append(entry)
                position_entries.append(entry)
            position_summaries.append(
                {
                    "prompt_id": prompt_id,
                    "token_index": token_index,
                    "top1_changed": top1_changed,
                    "deferred_top1_in_historical_ambiguity_set": int(row["deferred_top1_id"]) in ambiguity_members,
                    "pairwise_abs_max": max((e["absolute_distortion"] for e in position_entries), default=0.0),
                    "adverse_contraction_max": max((e["adverse_contraction"] for e in position_entries), default=0.0),
                    "beneficial_expansion_max": max((e["beneficial_expansion"] for e in position_entries), default=0.0),
                }
            )

    strata: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        strata[f"baseline_margin:{entry['baseline_margin_bucket']}"] .append(entry)
        strata[f"deferred_hit:{int(entry['target_row_deferred_hit'])}"].append(entry)
        count = entry["deferred_route_count"]
        strata[f"deferred_routes:{'0' if count == 0 else '1' if count == 1 else '2plus'}"].append(entry)
        strata[f"cpu_weight_sum_proxy:{weight_bucket(entry['target_row_cpu_route_weight_sum_proxy'])}"].append(entry)
        strata[f"cpu_weight_max_proxy:{weight_bucket(entry['target_row_cpu_route_weight_max_proxy'])}"].append(entry)
        strata[f"family:{entry['category']}"].append(entry)
        lower = entry["token_index"] // 16 * 16
        strata[f"position:{lower}-{lower + 15}"].append(entry)
        strata[f"top1_changed:{int(entry['top1_changed'])}"].append(entry)

    abs_values = [entry["absolute_distortion"] for entry in entries]
    adverse_values = [entry["adverse_contraction"] for entry in entries]
    route_mass_proxy = [entry["target_row_cpu_route_weight_sum_proxy"] for entry in entries]
    margins = [entry["m0"] for entry in entries]
    target = next(
        (
            entry
            for entry in entries
            if entry["prompt_id"] == "hd_en_02"
            and entry["token_index"] == 43
            and entry["candidate_id"] == 16520
        ),
        None,
    )
    result = {
        "schema_version": "round6.1-signed-risk-v1",
        "purpose": "descriptive_mechanism_research_not_qualification",
        "historical_bound_used_only_for_diagnostic_ambiguity_membership": args.historical_bound,
        "position_count": len(position_summaries),
        "pair_count": len(entries),
        "all_pairs": summarize(entries),
        "strata": {key: summarize(value) for key, value in sorted(strata.items())},
        "correlations": {
            "abs_distortion_vs_target_row_cpu_weight_mass_proxy": pearson(abs_values, route_mass_proxy),
            "adverse_contraction_vs_target_row_cpu_weight_mass_proxy": pearson(adverse_values, route_mass_proxy),
            "abs_distortion_vs_baseline_margin": pearson(abs_values, margins),
            "adverse_contraction_vs_baseline_margin": pearson(adverse_values, margins),
        },
        "positions": {
            "top1_changed": sum(int(row["top1_changed"]) for row in position_summaries),
            "historical_ambiguity_membership_failures": sum(
                int(not row["deferred_top1_in_historical_ambiguity_set"])
                for row in position_summaries
            ),
            "adverse_contraction_max": max(row["adverse_contraction_max"] for row in position_summaries),
            "beneficial_expansion_max": max(row["beneficial_expansion_max"] for row in position_summaries),
        },
        "historical_hd_en_02_43_candidate_16520": target,
        "weight_telemetry_limitation": (
            "Frozen QD/HD rows retain target-row CPU-route weights but do not retain "
            "the Deferred-only weight subset. Weight correlations therefore use the "
            "target-row total CPU-route weight mass as a labeled proxy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("position_count", "pair_count", "correlations", "positions", "historical_hd_en_02_43_candidate_16520")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
