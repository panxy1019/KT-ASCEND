#!/usr/bin/env python3
"""Verify one real DeepSeek-V2-Lite layer through the SGLang KTEP path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401 - registers the NPU device module
from safetensors import safe_open

from kt_kernel import KTMoEWrapper
from kt_kernel.utils.llamafile import LlamafileMoEWrapper
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTEPWrapperMethod,
    SharedStagingBuffer,
)
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput


class _Recorder:
    def on_gpu_expert_mask(self, _layer_idx, _mask):
        pass


class _SelectedNPUExpertMethod:
    """Real BF16 NPU experts indexed by KTEP's accelerator-local IDs."""

    def __init__(self, weights_by_local_id: dict[int, dict[str, torch.Tensor]]):
        self.weights = {
            local_id: {
                name: tensor.clone().to("npu", dtype=torch.bfloat16).contiguous()
                for name, tensor in projections.items()
            }
            for local_id, projections in weights_by_local_id.items()
        }

    def apply(self, _layer, dispatch_output):
        hidden_states = dispatch_output.hidden_states
        topk_weights, local_expert_ids, _ = dispatch_output.topk_output
        output = torch.zeros_like(hidden_states, dtype=torch.float32)
        for local_id, projections in self.weights.items():
            routes = torch.nonzero(local_expert_ids == local_id, as_tuple=False)
            if routes.numel() == 0:
                continue
            token_indices = routes[:, 0]
            route_indices = routes[:, 1]
            selected_hidden = hidden_states.index_select(0, token_indices)
            gate = F.linear(selected_hidden, projections["gate"])
            up = F.linear(selected_hidden, projections["up"])
            expert_output = F.linear(F.silu(gate) * up, projections["down"])
            weights = topk_weights[token_indices, route_indices].unsqueeze(1)
            output.index_add_(0, token_indices, expert_output.float() * weights)
        return StandardCombineInput(hidden_states=output.to(hidden_states.dtype))


def _load_tensors(
    model_dir: Path, weight_map: dict[str, str], keys: list[str]
) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(weight_map[key], []).append(key)
    tensors: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as reader:
            for key in shard_keys:
                # Never retain file-backed safetensors views for torch_npu H2D.
                tensors[key] = reader.get_tensor(key).clone().contiguous()
    return tensors


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_fp32 = actual.float().cpu()
    reference_fp32 = reference.float().cpu()
    difference = actual_fp32 - reference_fp32
    reference_norm = float(torch.linalg.vector_norm(reference_fp32))
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "relative_l2": (
            float(torch.linalg.vector_norm(difference)) / reference_norm
            if reference_norm
            else 0.0
        ),
        "cosine": float(
            F.cosine_similarity(actual_fp32.flatten(), reference_fp32.flatten(), dim=0)
        ),
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _npu_reference(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    weights_by_logical_id: dict[int, dict[str, torch.Tensor]],
    owned: list[int],
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    resident = {
        logical_id: {
            name: tensor.to("npu", dtype=torch.bfloat16).contiguous()
            for name, tensor in weights_by_logical_id[logical_id].items()
        }
        for logical_id in owned
    }
    for logical_id, projections in resident.items():
        routes = torch.nonzero(expert_ids == logical_id, as_tuple=False)
        if routes.numel() == 0:
            continue
        token_indices = routes[:, 0]
        route_indices = routes[:, 1]
        selected_hidden = hidden_states.index_select(0, token_indices)
        gate = F.linear(selected_hidden, projections["gate"])
        up = F.linear(selected_hidden, projections["up"])
        expert_output = F.linear(F.silu(gate) * up, projections["down"])
        weights = routing_weights[token_indices, route_indices].unsqueeze(1)
        output.index_add_(0, token_indices, expert_output.float() * weights)
    return output.to(hidden_states.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--layer", default=17, type=int)
    parser.add_argument("--cpu-expert", default=6, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

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
    n_shared_experts = int(config["n_shared_experts"])
    if top_k != 6 or num_experts != 64:
        raise ValueError("this frozen DeepSeek-V2-Lite replay expects 64 experts/top-6")

    # Expert 6 is the captured high-frequency CPU expert.  The other IDs are
    # real accelerator-owned experts seen in the same layer's router statistics.
    npu_experts = [25, 36, 0, 51, 12, 30]
    exercised_experts = [args.cpu_expert, *npu_experts]
    expert_keys = {
        logical_id: {
            name: (
                f"model.layers.{args.layer}.mlp.experts.{logical_id}."
                f"{name}_proj.weight"
            )
            for name in ("gate", "up", "down")
        }
        for logical_id in exercised_experts
    }
    shared_keys = {
        name: f"model.layers.{args.layer}.mlp.shared_experts.{name}_proj.weight"
        for name in ("gate", "up", "down")
    }
    flat_keys = [key for projections in expert_keys.values() for key in projections.values()]
    flat_keys.extend(shared_keys.values())
    tensors = _load_tensors(model_dir, weight_map, flat_keys)
    weights_by_logical_id = {
        logical_id: {
            name: tensors[key] for name, key in projections.items()
        }
        for logical_id, projections in expert_keys.items()
    }

    accelerator_mask = torch.ones(num_experts, dtype=torch.bool)
    accelerator_mask[args.cpu_expert] = False
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
    logical_to_local[accelerator_ids] = torch.arange(
        accelerator_ids.numel(), dtype=torch.int32
    )
    selected_weights_by_local = {
        int(logical_to_local[logical_id]): weights_by_logical_id[logical_id]
        for logical_id in npu_experts
    }

    method = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
    method.tp_rank = 0
    method.wrapper = cpu_wrapper
    method.kt_expert_lora_enabled = False
    method.num_gpu_experts = num_experts - 1
    method.gpu_prefill_token_threshold = 0
    method.gpu_experts_mask = accelerator_mask
    method.gpu_experts_mask_cuda = accelerator_mask.to("npu")
    method.logical_to_gpu_index_cuda = logical_to_local.to("npu")
    method.gpu_method = _SelectedNPUExpertMethod(selected_weights_by_local)
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
    main_stream_handle = int(device_module.current_stream().npu_stream)
    cpu_stream_handle = int(method._cpu_stream.npu_stream)

    torch.manual_seed(20260827 + args.layer)
    hidden_cpu = torch.randn(3, hidden_size, dtype=torch.bfloat16)
    hidden_npu = hidden_cpu.to("npu")
    ids_cpu = torch.tensor(
        [
            [25, 36, 0, 51, 12, 30],
            [args.cpu_expert, 25, 36, 0, 51, 12],
            [25, args.cpu_expert, 36, 0, 51, 12],
        ],
        dtype=torch.int64,
    )
    route_weights_cpu = torch.tensor(
        [
            [0.25, 0.20, 0.18, 0.15, 0.12, 0.10],
            [0.25, 0.20, 0.18, 0.15, 0.12, 0.10],
            [0.20, 0.25, 0.18, 0.15, 0.12, 0.10],
        ],
        dtype=torch.float32,
    )
    ids_npu = ids_cpu.to("npu")
    route_weights_npu = route_weights_cpu.to("npu")
    staging_data_ptr = method._shared_staging_buffer.get_slice(3).data_ptr()
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_npu,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=route_weights_npu,
            topk_ids=ids_npu,
            router_logits=torch.empty(3, num_experts, dtype=torch.float32, device="npu"),
        ),
    )

    previous_recorder = get_global_expert_distribution_recorder()
    set_global_expert_distribution_recorder(_Recorder())
    try:
        # Match the server lifecycle: one warmup invocation, then measured replay.
        method.apply(SimpleNamespace(), dispatch_output)
        device_module.synchronize()
        routed_hybrid = method.apply(SimpleNamespace(), dispatch_output).hidden_states
        device_module.synchronize()
    finally:
        set_global_expert_distribution_recorder(previous_recorder)

    cpu_ids = ids_cpu.clone()
    cpu_ids[cpu_ids != args.cpu_expert] = -1
    cpu_weights = route_weights_cpu.clone()
    cpu_weights[cpu_ids < 0] = 0
    cpu_contribution = cpu_wrapper.forward(
        hidden_cpu, cpu_ids, cpu_weights, main_stream_handle
    ).to("npu")
    npu_contribution = _npu_reference(
        hidden_npu,
        ids_npu,
        route_weights_npu,
        weights_by_logical_id,
        npu_experts,
    )
    expected_hybrid = npu_contribution + cpu_contribution
    all_npu_routed = _npu_reference(
        hidden_npu,
        ids_npu,
        route_weights_npu,
        weights_by_logical_id,
        exercised_experts,
    )

    shared = {
        name: tensors[key].to("npu", dtype=torch.bfloat16).contiguous()
        for name, key in shared_keys.items()
    }
    shared_output = F.linear(
        F.silu(F.linear(hidden_npu, shared["gate"]))
        * F.linear(hidden_npu, shared["up"]),
        shared["down"],
    )
    final_hybrid = routed_hybrid.mul(routed_scaling_factor) + shared_output
    final_all_npu = all_npu_routed.mul(routed_scaling_factor) + shared_output
    device_module.synchronize()

    result = {
        "layer": args.layer,
        "cpu_expert": args.cpu_expert,
        "npu_experts_exercised": npu_experts,
        "placement": "one CPU routed expert; all other logical experts NPU-owned",
        "cases": {
            "cpu_not_hit_row": 0,
            "mixed_row": 1,
            "reversed_mixed_row": 2,
        },
        "routed_scaling_factor": routed_scaling_factor,
        "routed_scaling_owner": "DeepseekV2MoE outer forward after KTEP routed output",
        "shared_expert_count": n_shared_experts,
        "shared_expert_owner": "DeepseekV2MoE outer forward; not KTEPWrapperMethod",
        "routing_weight_owner": "CPU and NPU expert providers; applied exactly once before additive merge",
        "execution": {
            "mode": (
                "sequential"
                if os.environ.get("SGLANG_KT_HYBRID_NO_CPU_STREAM") == "1"
                else "overlap"
            ),
            "main_stream_native_handle": main_stream_handle,
            "cpu_stream_native_handle": cpu_stream_handle,
            "distinct_streams": main_stream_handle != cpu_stream_handle,
            "input_data_ptr": hidden_npu.data_ptr(),
            "staging_data_ptr": staging_data_ptr,
            "distinct_staging_buffer": hidden_npu.data_ptr() != staging_data_ptr,
        },
        "output_sha256": {
            "routed_hybrid": _tensor_sha256(routed_hybrid),
            "final_hybrid": _tensor_sha256(final_hybrid),
        },
        "metrics": {
            "ktep_vs_explicit_cpu_plus_npu": _metrics(routed_hybrid, expected_hybrid),
            "cpu_not_hit_vs_all_npu": _metrics(routed_hybrid[0], all_npu_routed[0]),
            "mixed_vs_all_npu": _metrics(routed_hybrid[1:], all_npu_routed[1:]),
            "final_with_shared_once_vs_all_npu": _metrics(final_hybrid, final_all_npu),
        },
        "cpu_hit_routes": int((ids_cpu == args.cpu_expert).sum()),
        "finite_routed_hybrid": bool(torch.isfinite(routed_hybrid).all()),
        "finite_final_hybrid": bool(torch.isfinite(final_hybrid).all()),
    }

    contract = result["metrics"]["ktep_vs_explicit_cpu_plus_npu"]
    cpu_not_hit = result["metrics"]["cpu_not_hit_vs_all_npu"]
    mixed = result["metrics"]["mixed_vs_all_npu"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    assert result["cpu_hit_routes"] == 2
    assert result["finite_routed_hybrid"] and result["finite_final_hybrid"]
    assert result["execution"]["distinct_streams"]
    assert result["execution"]["distinct_staging_buffer"]
    assert contract["relative_l2"] <= 2e-2
    assert cpu_not_hit["relative_l2"] <= 1e-6
    assert mixed["relative_l2"] <= 2e-2


if __name__ == "__main__":
    main()
