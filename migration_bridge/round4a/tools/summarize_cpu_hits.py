#!/usr/bin/env python3
"""Summarize CPU-owned route hits from an SGLang logical_count recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", required=True, type=Path)
    parser.add_argument("--placement", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = torch.load(args.distribution, map_location="cpu", weights_only=True)
    counts = data["logical_count"].sum(dim=0).to(torch.int64)
    placement = json.loads(args.placement.read_text(encoding="utf-8"))
    per_layer = {}
    per_expert = {}
    total = 0
    hit_experts = 0
    selected_experts = 0
    for row in placement["layer_placements"]:
        layer = int(row["layer"])
        layer_total = 0
        layer_hits = {}
        for expert in row["cpu_experts"]:
            expert = int(expert)
            value = int(counts[layer, expert].item())
            layer_hits[str(expert)] = value
            per_expert[f"L{layer}E{expert}"] = value
            layer_total += value
            selected_experts += 1
            hit_experts += int(value > 0)
        per_layer[str(layer)] = layer_total
        total += layer_total
    result = {
        "profile": placement["profile"],
        "placement_sha256": placement["placement_sha256"],
        "distribution_shape": list(data["logical_count"].shape),
        "cpu_route_hits_per_layer": per_layer,
        "cpu_route_hits_per_expert": per_expert,
        "total_cpu_route_hits": total,
        "selected_layer_count": len(per_layer),
        "hit_layer_count": sum(value > 0 for value in per_layer.values()),
        "selected_expert_count": selected_experts,
        "hit_expert_count": hit_experts,
        "expert_coverage": hit_experts / selected_experts if selected_experts else 1.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
