#!/usr/bin/env python3
"""Summarize frozen-placement CPU route coverage from KT numerical dumps."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    placement = json.loads(args.placement.read_text())
    selected = {
        int(row["layer"]): {int(expert) for expert in row["cpu_experts"]}
        for row in placement["layer_placements"]
    }
    counts = {layer: Counter() for layer in selected}
    dump_counts = Counter()
    all_finite = True
    for path in sorted(args.route_dir.glob("layer*-pass*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        layer = int(payload["layer"])
        if layer not in selected:
            continue
        dump_counts[layer] += 1
        ids = payload["topk_ids"].flatten().tolist()
        weights = payload["topk_weights"]
        all_finite = all_finite and bool(torch.isfinite(weights).all())
        for expert in ids:
            expert = int(expert)
            if expert in selected[layer]:
                counts[layer][expert] += 1
    per_layer = {}
    hit_experts = 0
    for layer, experts in selected.items():
        hits = {str(expert): counts[layer][expert] for expert in sorted(experts)}
        exercised = sum(value > 0 for value in hits.values())
        hit_experts += exercised
        per_layer[str(layer)] = {
            "dump_count": dump_counts[layer],
            "cpu_expert_hits": hits,
            "selected": len(experts),
            "exercised": exercised,
            "total_cpu_hits": sum(hits.values()),
        }
    selected_count = sum(len(value) for value in selected.values())
    payload = {
        "schema_version": 1,
        "placement_sha256": placement["placement_sha256"],
        "selected_layer_count": len(selected),
        "layers_exercised": sum(row["total_cpu_hits"] > 0 for row in per_layer.values()),
        "selected_cpu_expert_count": selected_count,
        "cpu_experts_exercised": hit_experts,
        "expert_coverage_ratio": hit_experts / selected_count,
        "total_cpu_hits": sum(row["total_cpu_hits"] for row in per_layer.values()),
        "routing_weights_finite": all_finite,
        "per_layer": per_layer,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
