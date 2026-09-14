#!/usr/bin/env python3
"""Compare one real DeepSeek-V2 expert through LLAMAFILE CPU and torch NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401 - registers the NPU device module
from safetensors import safe_open

from kt_kernel import KTMoEWrapper
from kt_kernel.utils.llamafile import LlamafileMoEWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--expert", required=True, type=int)
    parser.add_argument("--num-tokens", default=4, type=int)
    parser.add_argument("--cpu-repeats", default=1, type=int)
    parser.add_argument("--cpuinfer-threads", default=16, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.num_tokens <= 0:
        raise ValueError("--num-tokens must be positive")
    if args.cpu_repeats <= 0:
        raise ValueError("--cpu-repeats must be positive")
    if args.cpuinfer_threads <= 0:
        raise ValueError("--cpuinfer-threads must be positive")

    started_at = time.perf_counter()
    stage_started_at = started_at
    timings: dict[str, float] = {}

    def finish_stage(name: str) -> None:
        nonlocal stage_started_at
        now = time.perf_counter()
        timings[name] = now - stage_started_at
        stage_started_at = now
        print(f"[identity] {name}: {timings[name]:.3f}s", flush=True)

    model_dir = args.model_dir.resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    num_experts = int(config["n_routed_experts"])
    top_k = int(config["num_experts_per_tok"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["moe_intermediate_size"])

    keys = {
        name: f"model.layers.{args.layer}.mlp.experts.{args.expert}.{name}_proj.weight"
        for name in ("gate", "up", "down")
    }
    shards = {index[key] for key in keys.values()}
    if len(shards) != 1:
        raise ValueError(f"selected expert spans shards: {sorted(shards)}")
    with safe_open(model_dir / next(iter(shards)), framework="pt", device="cpu") as reader:
        # Materialize the selected tensors instead of retaining safetensors'
        # file-backed views.  Direct torch_npu H2D from those mmap views is
        # pathologically slow on the A3 container filesystem.
        weights = {
            name: reader.get_tensor(key).clone().contiguous()
            for name, key in keys.items()
        }
    finish_stage("load_selected_safetensors")

    accelerator_mask = torch.ones(num_experts, dtype=torch.bool)
    accelerator_mask[args.expert] = False
    LlamafileMoEWrapper._gguf_loader_instance = None
    LlamafileMoEWrapper._gguf_loader_path = None
    wrapper = KTMoEWrapper(
        layer_idx=args.layer,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        gpu_experts_mask=accelerator_mask,
        cpuinfer_threads=args.cpuinfer_threads,
        threadpool_count=1,
        weight_path=str(args.gguf.resolve()),
        chunked_prefill_size=16,
        max_deferred_experts_per_token=0,
        method="LLAMAFILE",
        numa_nodes=[0],
    )
    finish_stage("construct_cpu_wrapper")
    mapping = torch.arange(num_experts, dtype=torch.int32)
    wrapper.load_weights(mapping)
    finish_stage("load_cpu_expert_weights")

    torch.manual_seed(20260827 + args.layer * 100 + args.expert)
    hidden_cpu = torch.randn(args.num_tokens, hidden_size, dtype=torch.bfloat16)
    expert_ids = torch.full((args.num_tokens, top_k), -1, dtype=torch.int64)
    expert_ids[:, 0] = args.expert
    routing_weights = torch.zeros((args.num_tokens, top_k), dtype=torch.float32)
    routing_weights[:, 0] = 1.0
    cpu_outputs = [
        wrapper.forward(hidden_cpu, expert_ids, routing_weights).clone()
        for _ in range(args.cpu_repeats)
    ]
    cpu_output = cpu_outputs[0].float()
    cpu_repeat_hashes = [
        hashlib.sha256(output.contiguous().view(torch.uint8).numpy()).hexdigest()
        for output in cpu_outputs
    ]
    cpu_repeat_max_abs = max(
        float((output.float() - cpu_output).abs().max()) for output in cpu_outputs
    )
    finish_stage("cpu_expert_forward")

    hidden_npu = hidden_cpu.to("npu")
    finish_stage("copy_hidden_to_npu")
    gate = weights["gate"].to("npu", dtype=torch.bfloat16)
    finish_stage("copy_gate_to_npu")
    up = weights["up"].to("npu", dtype=torch.bfloat16)
    finish_stage("copy_up_to_npu")
    down = weights["down"].to("npu", dtype=torch.bfloat16)
    finish_stage("copy_down_to_npu")
    gate_output = F.linear(hidden_npu, gate)
    finish_stage("npu_gate_linear_submit")
    up_output = F.linear(hidden_npu, up)
    finish_stage("npu_up_linear_submit")
    npu_output = F.linear(F.silu(gate_output) * up_output, down)
    finish_stage("npu_down_linear_submit")
    torch.npu.synchronize()
    npu_output_cpu = npu_output.cpu().float()
    finish_stage("npu_synchronize_and_copy_output")

    reference_fp32 = F.linear(
        F.silu(F.linear(hidden_cpu.float(), weights["gate"].float()))
        * F.linear(hidden_cpu.float(), weights["up"].float()),
        weights["down"].float(),
    )
    reference_bf16 = reference_fp32.to(torch.bfloat16).float()
    finish_stage("cpu_fp32_reference")

    def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
        difference = actual - reference
        reference_norm = float(torch.linalg.vector_norm(reference))
        return {
            "max_abs": float(difference.abs().max()),
            "mean_abs": float(difference.abs().mean()),
            "relative_l2": (
                float(torch.linalg.vector_norm(difference)) / reference_norm
                if reference_norm
                else 0.0
            ),
            "cosine": float(
                F.cosine_similarity(actual.flatten(), reference.flatten(), dim=0)
            ),
        }

    metrics = error_metrics(cpu_output, npu_output_cpu)
    reference_metrics = {
        "cpu_vs_fp32": error_metrics(cpu_output, reference_fp32),
        "npu_vs_fp32": error_metrics(npu_output_cpu, reference_fp32),
        "cpu_vs_bf16_rounded_fp32": error_metrics(cpu_output, reference_bf16),
        "npu_vs_bf16_rounded_fp32": error_metrics(npu_output_cpu, reference_bf16),
    }
    result = {
        "layer": args.layer,
        "logical_expert_id": args.expert,
        "physical_expert_id": args.expert,
        "num_tokens": args.num_tokens,
        "cpu_repeats": args.cpu_repeats,
        "cpuinfer_threads": args.cpuinfer_threads,
        "cpu_repeat_unique_hashes": sorted(set(cpu_repeat_hashes)),
        "cpu_repeat_max_abs": cpu_repeat_max_abs,
        "source_keys": keys,
        "source_shard": next(iter(shards)),
        "gate_shape": list(weights["gate"].shape),
        "up_shape": list(weights["up"].shape),
        "down_shape": list(weights["down"].shape),
        "source_dtype": str(weights["gate"].dtype),
        "cpu_gguf_dtype": "float32",
        "npu_compute_dtype": str(npu_output.dtype),
        "metrics": metrics,
        "reference_metrics": reference_metrics,
        "finite_cpu": bool(torch.isfinite(cpu_output).all()),
        "finite_npu": bool(torch.isfinite(npu_output_cpu).all()),
        "timings_seconds": timings,
        "total_seconds": time.perf_counter() - started_at,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
