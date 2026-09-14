#!/usr/bin/env python3
"""Create a deterministic KT frequency file with exactly one CPU expert.

The saved ``logical_count`` tensor is consumed by SGLang's KT frequency
placement strategy.  Every routed MoE expert receives count 1 except the
requested CPU expert, which receives count 0.  Dense/non-MoE layers also use
zero; SGLang subsequently forces those layers to the accelerator mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--num-layers", type=int, default=27)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--first-moe-layer", type=int, default=1)
    parser.add_argument("--moe-layer-freq", type=int, default=1)
    parser.add_argument("--cpu-layer", type=int, default=17)
    parser.add_argument("--cpu-expert", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.cpu_layer < args.num_layers):
        raise ValueError("--cpu-layer is outside the model layer range")
    if not (0 <= args.cpu_expert < args.num_experts):
        raise ValueError("--cpu-expert is outside the expert range")
    if (
        args.cpu_layer < args.first_moe_layer
        or args.cpu_layer % args.moe_layer_freq != 0
    ):
        raise ValueError("the requested CPU expert must belong to a MoE layer")

    logical_count = torch.zeros(
        (1, args.num_layers, args.num_experts), dtype=torch.int64
    )
    moe_layers = []
    for layer_id in range(args.first_moe_layer, args.num_layers):
        if layer_id % args.moe_layer_freq == 0:
            logical_count[0, layer_id, :] = 1
            moe_layers.append(layer_id)
    logical_count[0, args.cpu_layer, args.cpu_expert] = 0

    total_moe_experts = len(moe_layers) * args.num_experts
    gpu_experts = int(logical_count.sum().item())
    if gpu_experts != total_moe_experts - 1:
        raise AssertionError(
            f"expected {total_moe_experts - 1} selected experts, got {gpu_experts}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"logical_count": logical_count}, args.output)

    manifest = {
        "format": "sglang_kt_frequency_logical_count",
        "logical_count_shape": list(logical_count.shape),
        "logical_count_dtype": str(logical_count.dtype),
        "num_moe_layers": len(moe_layers),
        "total_moe_experts": total_moe_experts,
        "gpu_experts": gpu_experts,
        "cpu_experts": 1,
        "gpu_experts_ratio": gpu_experts / total_moe_experts,
        "cpu_expert": {"layer": args.cpu_layer, "expert": args.cpu_expert},
        "nonzero_entries": int(torch.count_nonzero(logical_count).item()),
    }
    manifest_path = args.manifest or args.output.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
