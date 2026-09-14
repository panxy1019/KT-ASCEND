#!/usr/bin/env python3
"""Compute pairwise top-order distortion and Round4A4 diagnostic strata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch


QUANTILES = {"min": 0.0, "p50": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99,
             "p99_5": 0.995, "p99_9": 0.999, "max": 1.0}


def distribution(values: list[float]) -> dict:
    if not values:
        return {}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {name: float(torch.quantile(tensor, q)) for name, q in QUANTILES.items()}


def route_coverage(route_dir: Path | None, route_name: str | None, cpu_experts: set[int]) -> dict:
    if route_dir is None or route_name is None:
        return {"cpu_hit_count": 0, "cpu_hit_experts": [], "routing_weights": []}
    match = re.fullmatch(r"layer(?P<layer>\d+)-pass(?P<index>\d+)-pid(?P<pid>\d+)\.pt", route_name)
    if match is None:
        raise RuntimeError(f"unrecognized route dump: {route_name}")
    index = int(match.group("index"))
    names = [route_name]
    if index:
        names.insert(0, f"layer{match.group('layer')}-pass{index - 1:05d}-pid{match.group('pid')}.pt")
    hits, weights = [], []
    for name in names:
        payload = torch.load(route_dir / name, map_location="cpu", weights_only=False)
        ids = payload["topk_ids"]
        route_weights = payload.get("topk_weights")
        for offset, expert in enumerate(ids.flatten().tolist()):
            if int(expert) in cpu_experts:
                hits.append(int(expert))
                if route_weights is not None:
                    weights.append(float(route_weights.flatten()[offset]))
    return {"cpu_hit_count": len(hits), "cpu_hit_experts": sorted(set(hits)), "routing_weights": weights}


def canonical_sha(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--hybrid-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--union", type=Path, required=True)
    parser.add_argument("--kt-dump-dir", type=Path)
    parser.add_argument("--cpu-experts", default="6,8,25,36")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    base_manifest = json.loads((args.baseline_dir / "manifest.json").read_text())
    hybrid_manifest = json.loads((args.hybrid_dir / "manifest.json").read_text())
    union = json.loads(args.union.read_text())
    union_rows = {(r["prompt_id"], r["token_index"]): r for r in union["rows"]}
    hybrid_rows = {(r["prompt_id"], r["token_index"]): r for r in hybrid_manifest["rows"]}
    cpu_experts = {int(x) for x in args.cpu_experts.split(",") if x}
    rows, all_errors = [], []
    strata: dict[str, list[float]] = defaultdict(list)
    for prompt in corpus["prompts"]:
        baseline = torch.load(args.baseline_dir / "logits" / f"{prompt['id']}.pt", weights_only=True).float()
        hybrid = torch.load(args.hybrid_dir / "logits" / f"{prompt['id']}.pt", weights_only=True).float()
        for token_index, (base_logits, hybrid_logits) in enumerate(zip(baseline, hybrid)):
            key = (prompt["id"], token_index)
            candidates = union_rows[key]
            a = candidates["baseline_top1_id"]
            ids = [i for i in candidates["union_ids"] if i != a]
            baseline_margins = [float(base_logits[a] - base_logits[i]) for i in ids]
            hybrid_margins = [float(hybrid_logits[a] - hybrid_logits[i]) for i in ids]
            distortions = [h - b for h, b in zip(hybrid_margins, baseline_margins)]
            errors = [abs(value) for value in distortions]
            coverage = route_coverage(args.kt_dump_dir, hybrid_rows[key].get("route_dump"), cpu_experts)
            probabilities = torch.softmax(base_logits, dim=0)
            entropy = float(-(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum())
            row = {
                **candidates, "category": prompt["category"],
                "candidate_ids": ids, "baseline_pairwise_margins": baseline_margins,
                "hybrid_pairwise_margins": hybrid_margins, "pairwise_distortions": distortions,
                "pairwise_abs_distortions": errors,
                "pairwise_abs_max": max(errors, default=0.0),
                "baseline_entropy": entropy,
                "baseline_top1_probability": float(probabilities[a]),
                "baseline_top2_probability": float(torch.topk(probabilities, 2).values[1]),
                "baseline_candidate_probabilities": [float(probabilities[i]) for i in ids],
                "all_finite": bool(torch.isfinite(base_logits).all() and torch.isfinite(hybrid_logits).all()),
                **coverage,
            }
            rows.append(row)
            all_errors.extend(errors)
            hit_label = "cpu_hit_0" if coverage["cpu_hit_count"] == 0 else ("cpu_hit_1" if coverage["cpu_hit_count"] == 1 else "cpu_hit_2_plus")
            for label in (hit_label, f"family:{prompt['category']}", f"position:{token_index // 16 * 16}-{token_index // 16 * 16 + 15}"):
                strata[label].extend(errors)
            for expert in coverage["cpu_hit_experts"]:
                strata[f"expert:E{expert}"].extend(errors)
    payload = {
        "schema_version": 1, "mechanism": "pairwise_margin_order_stability",
        "corpus_sha256": corpus["sha256"], "baseline_manifest_sha256": base_manifest["sha256"],
        "hybrid_manifest_sha256": hybrid_manifest["sha256"], "candidate_top_k": union["top_k"],
        "position_count": len(rows), "pair_count": len(all_errors), "all_finite": all(r["all_finite"] for r in rows),
        "pairwise_abs_distortion": distribution(all_errors),
        "per_position_max_distortion": distribution([r["pairwise_abs_max"] for r in rows]),
        "strata": {key: {"pair_count": len(values), "distribution": distribution(values)} for key, values in sorted(strata.items())},
        "rows": rows,
    }
    payload["sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
