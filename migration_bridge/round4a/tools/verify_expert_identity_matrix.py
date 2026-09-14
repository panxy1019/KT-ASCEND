#!/usr/bin/env python3
"""Verify every selected CPU expert against FP32 and BF16 NPU references."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from safetensors import safe_open

from kt_kernel import KTMoEWrapper
from kt_kernel.utils.llamafile import LlamafileMoEWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--placement", required=True, type=Path)
    parser.add_argument("--repeat-experts-per-layer", default=1, type=int)
    parser.add_argument("--cpu-repeats", default=5, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = actual - reference
    norm = float(torch.linalg.vector_norm(reference))
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(difference)) / norm if norm else 0.0,
        "cosine": float(F.cosine_similarity(actual.flatten(), reference.flatten(), dim=0)),
    }


def aggregate(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = round(0.95 * (len(ordered) - 1))
    return {
        "min": min(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": max(ordered),
    }


def main() -> None:
    args = parse_args()
    if args.repeat_experts_per_layer < 0 or args.cpu_repeats <= 0:
        raise ValueError("repeat counts must be valid")
    model_dir = args.model_dir.resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weight_map = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    placement = json.loads(args.placement.read_text(encoding="utf-8"))
    num_experts = int(config["n_routed_experts"])
    top_k = int(config["num_experts_per_tok"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["moe_intermediate_size"])
    rows = []
    LlamafileMoEWrapper._gguf_loader_instance = None
    LlamafileMoEWrapper._gguf_loader_path = None
    for layer_row in placement["layer_placements"]:
        layer = int(layer_row["layer"])
        cpu_experts = [int(expert) for expert in layer_row["cpu_experts"]]
        accelerator_mask = torch.ones(num_experts, dtype=torch.bool)
        accelerator_mask[cpu_experts] = False
        wrapper = KTMoEWrapper(
            layer_idx=layer,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            gpu_experts_mask=accelerator_mask,
            cpuinfer_threads=16,
            threadpool_count=1,
            weight_path=str(args.gguf.resolve()),
            chunked_prefill_size=16,
            max_deferred_experts_per_token=0,
            method="LLAMAFILE",
            numa_nodes=[0],
        )
        mapping = torch.arange(num_experts, dtype=torch.int32)
        wrapper.load_weights(mapping)
        for expert_index, expert in enumerate(cpu_experts):
            keys = {
                projection: f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
                for projection in ("gate", "up", "down")
            }
            shards = {weight_map[key] for key in keys.values()}
            if len(shards) != 1:
                raise ValueError(f"L{layer}E{expert} spans source shards")
            with safe_open(model_dir / next(iter(shards)), framework="pt", device="cpu") as reader:
                weights = {
                    projection: reader.get_tensor(key).clone().contiguous()
                    for projection, key in keys.items()
                }
            torch.manual_seed(20260828 + layer * 100 + expert)
            hidden_cpu = torch.randn(1, hidden_size, dtype=torch.bfloat16)
            ids = torch.full((1, top_k), -1, dtype=torch.int64)
            ids[0, 0] = expert
            route_weights = torch.zeros((1, top_k), dtype=torch.float32)
            route_weights[0, 0] = 1.0
            repeat_count = args.cpu_repeats if expert_index < args.repeat_experts_per_layer else 1
            cpu_outputs = [
                wrapper.forward(hidden_cpu, ids, route_weights).clone()
                for _ in range(repeat_count)
            ]
            cpu_fp32 = cpu_outputs[0].float()
            hashes = {
                hashlib.sha256(output.contiguous().view(torch.uint8).numpy()).hexdigest()
                for output in cpu_outputs
            }
            gate = weights["gate"].to("npu", dtype=torch.bfloat16)
            up = weights["up"].to("npu", dtype=torch.bfloat16)
            down = weights["down"].to("npu", dtype=torch.bfloat16)
            hidden_npu = hidden_cpu.to("npu")
            npu_output = F.linear(F.silu(F.linear(hidden_npu, gate)) * F.linear(hidden_npu, up), down)
            torch.npu.synchronize()
            npu_fp32 = npu_output.cpu().float()
            reference_fp32 = F.linear(
                F.silu(F.linear(hidden_cpu.float(), weights["gate"].float()))
                * F.linear(hidden_cpu.float(), weights["up"].float()),
                weights["down"].float(),
            )
            rounded_reference = reference_fp32.to(torch.bfloat16).float()
            row = {
                "layer": layer,
                "expert": expert,
                "cpu_repeats": repeat_count,
                "cpu_unique_hashes": len(hashes),
                "cpu_deterministic": len(hashes) == 1,
                "finite_cpu": bool(torch.isfinite(cpu_fp32).all()),
                "finite_npu": bool(torch.isfinite(npu_fp32).all()),
                "cpu_vs_bf16_rounded_fp32": metrics(cpu_fp32, rounded_reference),
                "cpu_vs_npu": metrics(cpu_fp32, npu_fp32),
                "source_shard": next(iter(shards)),
                "source_keys": keys,
            }
            row["status"] = (
                "PASS"
                if row["cpu_deterministic"]
                and row["finite_cpu"]
                and row["finite_npu"]
                and row["cpu_vs_bf16_rounded_fp32"]["relative_l2"] <= 5e-4
                and row["cpu_vs_npu"]["relative_l2"] <= 1e-2
                else "FAIL"
            )
            rows.append(row)
            print(
                f"L{layer}E{expert} repeat={repeat_count} "
                f"cpu/fp32={row['cpu_vs_bf16_rounded_fp32']['relative_l2']:.6g} "
                f"cpu/npu={row['cpu_vs_npu']['relative_l2']:.6g} {row['status']}",
                flush=True,
            )
            del gate, up, down, hidden_npu, npu_output
        del wrapper
    fp32_values = [row["cpu_vs_bf16_rounded_fp32"]["relative_l2"] for row in rows]
    npu_values = [row["cpu_vs_npu"]["relative_l2"] for row in rows]
    result = {
        "profile": placement["profile"],
        "placement_sha256": placement["placement_sha256"],
        "expert_count": len(rows),
        "cpu_repeats": args.cpu_repeats,
        "repeat_experts_per_layer": args.repeat_experts_per_layer,
        "cpu_vs_bf16_rounded_fp32_relative_l2": aggregate(fp32_values),
        "cpu_vs_npu_relative_l2": aggregate(npu_values),
        "all_finite": all(row["finite_cpu"] and row["finite_npu"] for row in rows),
        "all_pass": all(row["status"] == "PASS" for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
