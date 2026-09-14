#!/usr/bin/env python3
"""Compare matched-history full logits and summarize Round4A3 metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F


QUANTILES = {"min": 0.0, "p50": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99, "p99_5": 0.995, "max": 1.0}


def distribution(values: list[float]) -> dict:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {name: float(torch.quantile(tensor, q)) for name, q in QUANTILES.items()}


def load_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


def route_coverage(route_dir: Path, route_name: str, cpu_experts: set[int]) -> dict:
    match = re.fullmatch(r"layer(?P<layer>\d+)-pass(?P<index>\d+)-pid(?P<pid>\d+)\.pt", route_name)
    if match is None:
        raise RuntimeError(f"unrecognized route dump name: {route_name}")
    index = int(match.group("index"))
    if index == 0:
        raise RuntimeError(f"sampling route has no preceding prefix pass: {route_name}")
    prefix_name = f"layer{match.group('layer')}-pass{index - 1:05d}-pid{match.group('pid')}.pt"
    prefix = torch.load(route_dir / prefix_name, map_location="cpu", weights_only=False)["topk_ids"]
    sampling = torch.load(route_dir / route_name, map_location="cpu", weights_only=False)["topk_ids"]
    prefix_hits = [int(value) for value in prefix.flatten().tolist() if int(value) in cpu_experts]
    sampling_hits = [int(value) for value in sampling[-1].flatten().tolist() if int(value) in cpu_experts]
    all_hits = prefix_hits + sampling_hits
    return {
        "cpu_hit_count": len(all_hits),
        "cpu_hit_experts": sorted(set(all_hits)),
        "prefix_cpu_hit_count": len(prefix_hits),
        "sampling_cpu_hit_count": len(sampling_hits),
        "sampling_cpu_hit_experts": sorted(set(sampling_hits)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--hybrid-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--kt-dump-dir", type=Path)
    parser.add_argument("--cpu-experts", default="6,8,25,36")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_manifest = load_manifest(args.baseline_dir)
    hybrid_manifest = load_manifest(args.hybrid_dir)
    corpus = json.loads(args.corpus.read_text())
    cpu_experts = {int(value) for value in args.cpu_experts.split(",") if value}
    if baseline_manifest["corpus_sha256"] != corpus["sha256"] or hybrid_manifest["corpus_sha256"] != corpus["sha256"]:
        raise RuntimeError("corpus mismatch")
    hybrid_rows = {(row["prompt_id"], row["token_index"]): row for row in hybrid_manifest["rows"]}
    rows = []
    for prompt in corpus["prompts"]:
        baseline = torch.load(args.baseline_dir / "logits" / f"{prompt['id']}.pt", map_location="cpu", weights_only=True).float()
        hybrid = torch.load(args.hybrid_dir / "logits" / f"{prompt['id']}.pt", map_location="cpu", weights_only=True).float()
        if baseline.shape != hybrid.shape:
            raise RuntimeError(f"logit shape mismatch for {prompt['id']}")
        for token_index, (base_logits, hybrid_logits) in enumerate(zip(baseline, hybrid)):
            difference = hybrid_logits - base_logits
            baseline_route = baseline_manifest["rows"][len(rows)]
            route = hybrid_rows[(prompt["id"], token_index)]
            coverage = route_coverage(args.kt_dump_dir, route["route_dump"], cpu_experts) if args.kt_dump_dir else {}
            base_top1 = int(baseline_route["response_token"])
            hybrid_top1 = int(route["response_token"])
            base_order = torch.argsort(base_logits, descending=True, stable=True)
            hybrid_order = torch.argsort(hybrid_logits, descending=True, stable=True)
            base_rest = base_order[base_order != base_top1][:15]
            hybrid_rest = hybrid_order[hybrid_order != hybrid_top1][:15]
            base_ids = torch.cat((torch.tensor([base_top1]), base_rest))
            hybrid_ids = torch.cat((torch.tensor([hybrid_top1]), hybrid_rest))
            base_values = base_logits[base_ids]
            hybrid_values = hybrid_logits[hybrid_ids]
            base_top2 = int(base_ids[1])
            route = hybrid_rows[(prompt["id"], token_index)]
            rows.append(
                {
                    "prompt_id": prompt["id"],
                    "category": prompt["category"],
                    "token_index": token_index,
                    "baseline_token": route["baseline_token"],
                    "baseline_top1_id": base_top1,
                    "baseline_top2_id": base_top2,
                    "baseline_top1_logit": float(base_values[0]),
                    "baseline_top2_logit": float(base_values[1]),
                    "baseline_margin": float(base_values[0] - base_values[1]),
                    "hybrid_top1": hybrid_top1,
                    "hybrid_top1_logit": float(hybrid_values[0]),
                    "hybrid_top2_id": int(hybrid_ids[1]),
                    "hybrid_top2_logit": float(hybrid_values[1]),
                    "hybrid_margin": float(hybrid_values[0] - hybrid_values[1]),
                    "hybrid_logit_for_baseline_top1": float(hybrid_logits[base_top1]),
                    "hybrid_logit_for_baseline_top2": float(hybrid_logits[base_top2]),
                    "max_abs_logit_diff": float(difference.abs().max()),
                    "mean_abs_logit_diff": float(difference.abs().mean()),
                    "relative_l2_logit": float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(base_logits)),
                    "cosine_logit": float(F.cosine_similarity(hybrid_logits, base_logits, dim=0)),
                    "candidate_max_error": float(difference[base_ids].abs().max()),
                    "margin_distortion": float((hybrid_logits[base_top1] - hybrid_logits[base_top2]) - (base_values[0] - base_values[1])),
                    "baseline_top16_ids": [int(value) for value in base_ids],
                    "baseline_top16_logits": [float(value) for value in base_values],
                    "hybrid_top16_ids": [int(value) for value in hybrid_ids],
                    "hybrid_top16_logits": [float(value) for value in hybrid_values],
                    "cpu_hit_count": coverage.get("cpu_hit_count", route["cpu_hit_count"]),
                    "cpu_hit_layers": route["cpu_hit_layers"],
                    "cpu_hit_experts": coverage.get("cpu_hit_experts", route["cpu_hit_experts"]),
                    "prefix_cpu_hit_count": coverage.get("prefix_cpu_hit_count"),
                    "sampling_cpu_hit_count": coverage.get("sampling_cpu_hit_count", route["cpu_hit_count"]),
                    "sampling_cpu_hit_experts": coverage.get("sampling_cpu_hit_experts", route["cpu_hit_experts"]),
                    "all_finite": bool(torch.isfinite(base_logits).all() and torch.isfinite(hybrid_logits).all()),
                }
            )
    metric_names = ("max_abs_logit_diff", "mean_abs_logit_diff", "relative_l2_logit", "candidate_max_error", "margin_distortion")
    summary = {name: distribution([abs(row[name]) for row in rows]) for name in metric_names}
    strata = {}
    for label, predicate in {
        "cpu_hit_0": lambda row: row["cpu_hit_count"] == 0,
        "cpu_hit_1": lambda row: row["cpu_hit_count"] == 1,
        "cpu_hit_2_plus": lambda row: row["cpu_hit_count"] >= 2,
        "E6": lambda row: 6 in row["cpu_hit_experts"],
        "E8": lambda row: 8 in row["cpu_hit_experts"],
        "E25": lambda row: 25 in row["cpu_hit_experts"],
        "E36": lambda row: 36 in row["cpu_hit_experts"],
    }.items():
        selected = [row for row in rows if predicate(row)]
        strata[label] = {
            "count": len(selected),
            "metrics": {name: distribution([abs(row[name]) for row in selected]) for name in metric_names} if selected else {},
        }
    payload = {
        "schema_version": 1,
        "corpus_sha256": corpus["sha256"],
        "baseline_manifest_sha256": baseline_manifest["sha256"],
        "hybrid_manifest_sha256": hybrid_manifest["sha256"],
        "row_count": len(rows),
        "all_finite": all(row["all_finite"] for row in rows),
        "metrics": summary,
        "strata": strata,
        "rows": rows,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
