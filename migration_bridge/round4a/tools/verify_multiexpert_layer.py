#!/usr/bin/env python3
"""Verify real same-layer multi-CPU-expert routing through SGLang KTEP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


ROUND3_TOOLS = Path(__file__).resolve().parents[2] / "round3" / "tools"
sys.path.insert(0, str(ROUND3_TOOLS))
from verify_real_single_layer_hybrid import (  # noqa: E402
    _Recorder,
    _SelectedNPUExpertMethod,
    _load_tensors,
    _metrics,
    _npu_reference,
    _tensor_sha256,
)

from kt_kernel import KTMoEWrapper  # noqa: E402
from kt_kernel.utils.llamafile import LlamafileMoEWrapper  # noqa: E402
from sglang.srt.eplb.expert_distribution import (  # noqa: E402
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.layers.moe.kt_ep_wrapper import (  # noqa: E402
    KTEPWrapperMethod,
    SharedStagingBuffer,
)
from sglang.srt.layers.moe.token_dispatcher.standard import (  # noqa: E402
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput  # noqa: E402


def parse_ids(value: str) -> list[int]:
    ids = [int(item) for item in value.split(",") if item]
    if len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("expert IDs must be unique")
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--layer", default=17, type=int)
    parser.add_argument("--cpu-experts", default="6,8,25,36", type=parse_ids)
    parser.add_argument("--npu-experts", default="0,12,30,51,55,63", type=parse_ids)
    parser.add_argument("--lifecycle-forwards", default=1, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.cpu_experts) != 4 or len(args.npu_experts) < 6:
        raise ValueError("frozen P1 fixture requires four CPU and at least six NPU experts")
    if args.lifecycle_forwards <= 0:
        raise ValueError("--lifecycle-forwards must be positive")
    model_dir = args.model_dir.resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weight_map = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    num_experts = int(config["n_routed_experts"])
    top_k = int(config["num_experts_per_tok"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["moe_intermediate_size"])
    routed_scaling_factor = float(config["routed_scaling_factor"])
    if top_k != 6:
        raise ValueError("controlled P1 cases require DeepSeek-V2-Lite top-6")
    if set(args.cpu_experts) & set(args.npu_experts):
        raise ValueError("CPU and NPU fixtures overlap")

    exercised = sorted(set(args.cpu_experts + args.npu_experts))
    expert_keys = {
        expert: {
            projection: f"model.layers.{args.layer}.mlp.experts.{expert}.{projection}_proj.weight"
            for projection in ("gate", "up", "down")
        }
        for expert in exercised
    }
    shared_keys = {
        projection: f"model.layers.{args.layer}.mlp.shared_experts.{projection}_proj.weight"
        for projection in ("gate", "up", "down")
    }
    keys = [key for projections in expert_keys.values() for key in projections.values()]
    keys.extend(shared_keys.values())
    tensors = _load_tensors(model_dir, weight_map, keys)
    weights = {
        expert: {projection: tensors[key] for projection, key in projections.items()}
        for expert, projections in expert_keys.items()
    }

    accelerator_mask = torch.ones(num_experts, dtype=torch.bool)
    accelerator_mask[args.cpu_experts] = False
    LlamafileMoEWrapper._gguf_loader_instance = None
    LlamafileMoEWrapper._gguf_loader_path = None
    cpu_wrapper = KTMoEWrapper(
        layer_idx=args.layer,
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
    cpu_wrapper.load_weights(torch.arange(num_experts, dtype=torch.int32))
    logical_to_local = torch.full((num_experts,), -1, dtype=torch.int32)
    accelerator_ids = torch.where(accelerator_mask)[0]
    logical_to_local[accelerator_ids] = torch.arange(accelerator_ids.numel(), dtype=torch.int32)

    method = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
    method.tp_rank = 0
    method.wrapper = cpu_wrapper
    method.kt_expert_lora_enabled = False
    method.num_gpu_experts = num_experts - len(args.cpu_experts)
    method.gpu_prefill_token_threshold = 0
    method.gpu_experts_mask = accelerator_mask
    method.gpu_experts_mask_cuda = accelerator_mask.to("npu")
    method.logical_to_gpu_index_cuda = logical_to_local.to("npu")
    method.gpu_method = _SelectedNPUExpertMethod(
        {int(logical_to_local[expert]): weights[expert] for expert in args.npu_experts}
    )
    method.kt_config = SimpleNamespace(layer_idx=args.layer)
    method.moe_runner_config = SimpleNamespace(activation="silu")
    device_module = torch.get_device_module(torch.device("npu"))
    method._cpu_stream = device_module.Stream(device=torch.device("npu"))
    method._sync_done_event = device_module.Event()
    method._shared_staging_buffer = SharedStagingBuffer(
        max_tokens=8,
        hidden_size=hidden_size,
        dtype=torch.bfloat16,
        device=torch.device("npu"),
    )

    c = args.cpu_experts
    n = args.npu_experts
    case_names = ["c0_npu_only", "c1_one_cpu", "c2_two_cpu", "c3_cpu_positions", "c4_multi_token"]
    ids_cpu = torch.tensor(
        [
            n[:6],
            [c[0], *n[:5]],
            [c[0], c[1], *n[:4]],
            [n[0], c[2], n[1], c[3], n[2], n[3]],
            [c[3], n[4], c[0], n[5], c[1], c[2]],
        ],
        dtype=torch.int64,
    )
    routing_cpu = torch.tensor(
        [
            [0.25, 0.20, 0.18, 0.15, 0.12, 0.10],
            [0.22, 0.21, 0.18, 0.16, 0.13, 0.10],
            [0.24, 0.20, 0.18, 0.15, 0.13, 0.10],
            [0.23, 0.21, 0.18, 0.16, 0.12, 0.10],
            [0.24, 0.20, 0.18, 0.15, 0.13, 0.10],
        ],
        dtype=torch.float32,
    )
    torch.manual_seed(20260828 + args.layer)
    hidden_cpu = torch.randn(len(case_names), hidden_size, dtype=torch.bfloat16)
    hidden_npu = hidden_cpu.to("npu")
    ids_npu = ids_cpu.to("npu")
    routing_npu = routing_cpu.to("npu")
    dispatch = StandardDispatchOutput(
        hidden_states=hidden_npu,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=routing_npu,
            topk_ids=ids_npu,
            router_logits=torch.empty(len(case_names), num_experts, dtype=torch.float32, device="npu"),
        ),
    )

    previous_recorder = get_global_expert_distribution_recorder()
    set_global_expert_distribution_recorder(_Recorder())
    output_hashes = []
    try:
        method.apply(SimpleNamespace(), dispatch)
        device_module.synchronize()
        for _ in range(args.lifecycle_forwards):
            routed_hybrid = method.apply(SimpleNamespace(), dispatch).hidden_states
            device_module.synchronize()
            output_hashes.append(_tensor_sha256(routed_hybrid))
    finally:
        set_global_expert_distribution_recorder(previous_recorder)

    cpu_ids = ids_cpu.clone()
    cpu_weights = routing_cpu.clone()
    cpu_owned = ~accelerator_mask[cpu_ids]
    cpu_ids[~cpu_owned] = -1
    cpu_weights[~cpu_owned] = 0
    cpu_contribution = cpu_wrapper.forward(hidden_cpu, cpu_ids, cpu_weights).to("npu")
    npu_contribution = _npu_reference(
        hidden_npu, ids_npu, routing_npu, weights, args.npu_experts
    )
    expected = cpu_contribution + npu_contribution
    all_npu = _npu_reference(hidden_npu, ids_npu, routing_npu, weights, exercised)

    shared = {
        projection: tensors[key].to("npu", dtype=torch.bfloat16).contiguous()
        for projection, key in shared_keys.items()
    }
    shared_output = F.linear(
        F.silu(F.linear(hidden_npu, shared["gate"])) * F.linear(hidden_npu, shared["up"]),
        shared["down"],
    )
    final_hybrid = routed_hybrid.mul(routed_scaling_factor) + shared_output
    final_expected = expected.mul(routed_scaling_factor) + shared_output
    device_module.synchronize()
    case_metrics = {
        name: {
            "cpu_owned_routes": int((~accelerator_mask[ids_cpu[index]]).sum().item()),
            "ktep_vs_explicit": _metrics(routed_hybrid[index], expected[index]),
            "hybrid_vs_all_npu": _metrics(routed_hybrid[index], all_npu[index]),
        }
        for index, name in enumerate(case_names)
    }
    result = {
        "layer": args.layer,
        "cpu_experts": args.cpu_experts,
        "npu_experts_exercised": args.npu_experts,
        "physical_npu_expert_count": method.num_gpu_experts,
        "logical_to_local_dtype": str(logical_to_local.dtype),
        "logical_to_local": logical_to_local.tolist(),
        "cases": case_metrics,
        "lifecycle_forwards": args.lifecycle_forwards,
        "lifecycle_unique_output_hashes": sorted(set(output_hashes)),
        "execution_mode": "sequential" if os.environ.get("SGLANG_KT_HYBRID_NO_CPU_STREAM") == "1" else "overlap",
        "output_sha256": {
            "routed": _tensor_sha256(routed_hybrid),
            "final": _tensor_sha256(final_hybrid),
        },
        "metrics": {
            "ktep_vs_explicit": _metrics(routed_hybrid, expected),
            "final_shared_and_scaling_once": _metrics(final_hybrid, final_expected),
            "cpu_not_hit_vs_all_npu": _metrics(routed_hybrid[0], all_npu[0]),
        },
        "routing_weight_owner": "CPU and NPU providers exactly once before additive merge",
        "routed_scaling_owner": "outer DeepSeek MoE exactly once",
        "shared_expert_owner": "outer DeepSeek MoE exactly once",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
