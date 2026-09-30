"""Real-image native Ascend resident-partial MoE probe.

The probe deliberately invokes the frozen ``AscendUnquantizedFusedMoEMethod``
with its unmodified selector and communication implementation.  The input
remains in the logical-512 domain while the owner contains only physical Q32
weights.  It therefore answers the Round18 P1 question before any CPU bridge
or full-model work is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from round16.b.q32_hybrid_contract import HIDDEN_SIZE, validate_v2_receipt
from round16_integration.b_q32_hybrid_live_probe import (
    _enter_bounded_forward_context,
    _hash_tensor,
    _relative_l2,
)
from round16_integration.b_q32_loader_live_factory import create_q32_loader_binding
from round16_integration.b_q32_loader_probe import Q32V2FactoryContext
from round16_integration.p1_allgather_q32_probe import (
    load_snapshot_manifest,
    load_source_preflight_token,
)
from round18.native_partial_contract import (
    LOGICAL_EXPERTS,
    PHYSICAL_NPU_SLOTS,
    NativePartialCase,
    frozen_native_partial_cases,
    validate_native_partial_receipt,
)
from round18.native_runner_seam import execution_map_from_owner


R18_NATIVE_PARTIAL_PROBE_API = "round18:native-resident-partial-device-probe:v1"
MAX_RELATIVE_L2 = 0.01


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_enable(environment: Mapping[str, str]) -> None:
    if environment.get("ROUND18_ENABLE_NATIVE_PARTIAL") != "1":
        raise RuntimeError("ROUND18_ENABLE_NATIVE_PARTIAL=1 is required")


def _router_logits(case: NativePartialCase, torch_module: Any) -> Any:
    rows = torch_module.full(
        (len(case.rows), LOGICAL_EXPERTS),
        -40.0,
        dtype=torch_module.float32,
        device="npu:0",
    )
    for row_index, ids in enumerate(case.rows):
        if case.tie_topk:
            values = [20.0] * len(ids)
        else:
            values = [20.0 - index for index in range(len(ids))]
        rows[row_index, torch_module.tensor(ids, dtype=torch_module.int64, device="npu:0")] = (
            torch_module.tensor(values, dtype=torch_module.float32, device="npu:0")
        )
    return rows


def _load_bf16_reference(
    *, model_root: Path, layer_id: int, resident_ids: tuple[int, ...]
) -> dict[int, tuple[Any, Any, Any]]:
    from safetensors import safe_open

    index = json.loads((model_root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    base = f"model.language_model.layers.{layer_id}.mlp.experts."
    gate_key = base + "gate_up_proj"
    down_key = base + "down_proj"
    result: dict[int, tuple[Any, Any, Any]] = {}
    with safe_open(str(model_root / weight_map[gate_key]), framework="pt", device="cpu") as gate_file, safe_open(
        str(model_root / weight_map[down_key]), framework="pt", device="cpu"
    ) as down_file:
        gate_slice = gate_file.get_slice(gate_key)
        down_slice = down_file.get_slice(down_key)
        for expert_id in resident_ids:
            gate_up = gate_slice[expert_id]
            gate, up = gate_up.float().chunk(2, dim=0)
            down = down_slice[expert_id].float()
            result[expert_id] = (
                gate.contiguous(),
                up.contiguous(),
                down.contiguous(),
            )
    return result


def _resident_reference(
    hidden_cpu: Any,
    topk_ids_cpu: Any,
    topk_weights_cpu: Any,
    weights: Mapping[int, tuple[Any, Any, Any]],
) -> Any:
    import torch

    output = torch.zeros(hidden_cpu.shape[0], HIDDEN_SIZE, dtype=torch.float32)
    for row in range(hidden_cpu.shape[0]):
        source = hidden_cpu[row : row + 1].float()
        for column in range(topk_ids_cpu.shape[1]):
            expert_id = int(topk_ids_cpu[row, column])
            if expert_id >= PHYSICAL_NPU_SLOTS:
                continue
            gate, up, down = weights[expert_id]
            gate_out = (source @ gate.T).bfloat16().float()
            up_out = (source @ up.T).bfloat16().float()
            intermediate = (torch.nn.functional.silu(gate_out) * up_out).bfloat16().float()
            expert_out = (intermediate @ down.T).bfloat16().float()
            output[row].add_(
                expert_out[0] * float(topk_weights_cpu[row, column])
            )
    return output.bfloat16()


def _run_case(
    *,
    case: NativePartialCase,
    owner: Any,
    comm: Any,
    expert_map: Any,
    reference_weights: Mapping[int, tuple[Any, Any, Any]],
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    import torch
    import vllm_ascend.ops.fused_moe.fused_moe as fused_moe_module

    torch.manual_seed(seed)
    hidden_cpu = torch.randn(len(case.rows), HIDDEN_SIZE, dtype=torch.float32).bfloat16()
    hidden = hidden_cpu.to("npu:0")
    router_logits = _router_logits(case, torch)
    hashes: list[str] = []
    rows: list[dict[str, Any]] = []
    actual_id_sets: list[list[int]] | None = None

    for repeat in range(repeats):
        captured: list[tuple[Any, Any]] = []
        original_selector = fused_moe_module.select_experts

        def observed_selector(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
            weights, ids = original_selector(*args, **kwargs)
            captured.append((weights.detach().clone(), ids.detach().clone()))
            return weights, ids

        fused_moe_module.select_experts = observed_selector
        try:
            prepared = comm.prepare(
                hidden_states=hidden,
                router_logits=router_logits,
                replace_allreduce=False,
                enable_shared_expert_dp=False,
            )
            fused_result = owner.quant_method.apply(
                layer=owner,
                x=prepared.hidden_states,
                router_logits=prepared.router_logits,
                top_k=10,
                renormalize=True,
                use_grouped_topk=False,
                num_experts=LOGICAL_EXPERTS,
                expert_map=expert_map,
                routed_scaling_factor=1.0,
                activation="silu",
            )
            if not hasattr(fused_result, "routed_out"):
                raise TypeError("native quant method did not return FusedExpertsResult")
            actual = comm.finalize(
                hidden_states=fused_result.routed_out,
                reduce_results=True,
                padded_hidden_states_shape=prepared.padded_hidden_states_shape,
            )
            torch.npu.synchronize()
        finally:
            fused_moe_module.select_experts = original_selector

        if len(captured) != 1:
            raise AssertionError(
                f"native quant method selected experts {len(captured)} times"
            )
        topk_weights, topk_ids = captured[0]
        ids_cpu = topk_ids.cpu()
        weights_cpu = topk_weights.float().cpu()
        expected_sets = [set(route) for route in case.rows]
        observed_sets = [set(int(value) for value in row) for row in ids_cpu.tolist()]
        if observed_sets != expected_sets:
            raise AssertionError(
                f"native selector changed frozen {case.name} routes: {observed_sets}"
            )
        if actual_id_sets is None:
            actual_id_sets = [sorted(values) for values in observed_sets]
        expected = _resident_reference(
            hidden_cpu, ids_cpu, weights_cpu, reference_weights
        )
        actual_cpu = actual.detach().cpu()
        error = _relative_l2(actual_cpu, expected)
        if error > MAX_RELATIVE_L2:
            raise AssertionError(
                f"native resident partial {case.name} relative L2 {error} exceeds {MAX_RELATIVE_L2}"
            )
        if not bool(torch.isfinite(actual_cpu).all()):
            raise AssertionError("native resident partial is not finite")
        digest = _hash_tensor(actual_cpu)
        hashes.append(digest)
        rows.append(
            {
                "repeat": repeat,
                "relative_l2": error,
                "hash": digest,
                "selector_calls": len(captured),
            }
        )

    if len(set(hashes)) != 1:
        raise AssertionError(f"native partial {case.name} is not repeat-exact")
    return {
        "case": case.name,
        "token_rows": len(case.rows),
        "requested_ids": [list(row) for row in case.rows],
        "actual_id_sets": actual_id_sets,
        "resident_occurrences": sum(
            expert_id < PHYSICAL_NPU_SLOTS
            for route in case.rows
            for expert_id in route
        ),
        "nonresident_occurrences": sum(
            expert_id >= PHYSICAL_NPU_SLOTS
            for route in case.rows
            for expert_id in route
        ),
        "topk_ties": case.tie_topk,
        "passed": True,
        "repeat_exact": True,
        "hash": hashes[0],
        "rows": rows,
    }


def run_device_probe(parsed: argparse.Namespace) -> dict[str, Any]:
    source_token = load_source_preflight_token(parsed.source_token)
    snapshot = load_snapshot_manifest(parsed.snapshot_manifest)
    v2_receipt, v2_sha256 = validate_v2_receipt(parsed.v2_receipt)
    if tuple(v2_receipt["resident_logical_ids"]) != snapshot.resident_logical_ids:
        raise ValueError("Round18 V2 receipt and S0 resident maps differ")
    if parsed.layer != 0:
        raise ValueError("the first native partial probe is restricted to layer0")

    import torch
    import torch_npu
    # Full vLLM workers import the compiled extension during worker startup.
    # This bounded component process bypasses that startup path, so register
    # the exact frozen custom-op namespace explicitly before invoking the
    # original Ascend selector.  Falling back to Python Top-K here would no
    # longer answer the P1 native-selector question.
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
    from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl

    del torch_npu
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(0.10, 0)
    torch.npu.config.allow_internal_format = True
    if not hasattr(torch.ops._C_ascend, "moe_gating_top_k"):
        raise RuntimeError("frozen Ascend moe_gating_top_k op was not registered")

    context = Q32V2FactoryContext(
        source_token=source_token,
        snapshot=snapshot,
        layer_id=parsed.layer,
        checkpoint_root=parsed.model_root,
        v1_receipt_sha256=str(v2_receipt["v1_receipt_sha256"]),
    )
    binding = create_q32_loader_binding(context)
    config_scope = None
    forward_scope = None
    prior_extra: dict[str, Any] = {}
    distributed_initialized = False
    model_parallel_initialized = False
    started = time.monotonic()
    try:
        vllm_config = VllmConfig(additional_config={"weight_nz_mode": 2})
        config_scope = set_current_vllm_config(vllm_config)
        config_scope.__enter__()
        ascend_config = init_ascend_config(vllm_config)
        if getattr(ascend_config, "weight_nz_mode", None) != 2:
            raise RuntimeError("Round18 did not restore BF16 NZ mode 2")
        # Full workers create EP even for the TP1/EP1 case.  The native
        # AllGather token dispatcher calls get_ep_group() unconditionally, so
        # the component probe must reproduce that official worker lifecycle
        # rather than installing a fake group object.
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29918",
            backend="hccl",
        )
        distributed_initialized = True
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            prefill_context_model_parallel_size=1,
            decode_context_model_parallel_size=1,
            backend="hccl",
        )
        model_parallel_initialized = True
        forward_scope, _forward_state = _enter_bounded_forward_context()

        owner = binding.owner
        comm = AllGatherCommImpl(owner.moe_config)
        for name in (
            "moe_comm_method",
            "moe_comm_type",
            "flash_comm_v1_enabled",
            "in_profile_run",
        ):
            prior_extra[name] = getattr(_EXTRA_CTX, name, None)
        _EXTRA_CTX.moe_comm_method = comm
        _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
        _EXTRA_CTX.flash_comm_v1_enabled = False
        _EXTRA_CTX.in_profile_run = False

        expert_map = execution_map_from_owner(
            owner, owner.round16_selective_plan
        )
        if str(expert_map.device).split(":")[0] != "npu":
            raise RuntimeError("native owner expert map is not on NPU")
        if type(owner.quant_method).__name__ != "AscendUnquantizedFusedMoEMethod":
            raise RuntimeError("native owner quant method identity changed")

        resident_ids = tuple(
            sorted(
                {
                    expert_id
                    for case in frozen_native_partial_cases()
                    for route in case.rows
                    for expert_id in route
                    if expert_id < PHYSICAL_NPU_SLOTS
                }
            )
        )
        references = _load_bf16_reference(
            model_root=parsed.model_root,
            layer_id=parsed.layer,
            resident_ids=resident_ids,
        )
        torch.npu.synchronize()
        allocator_steady = int(torch.npu.memory_allocated())
        torch.npu.reset_peak_memory_stats()
        results = [
            _run_case(
                case=case,
                owner=owner,
                comm=comm,
                expert_map=expert_map,
                reference_weights=references,
                seed=18000 + index,
                repeats=3,
            )
            for index, case in enumerate(frozen_native_partial_cases())
        ]
        # Twenty alternating calls establish bounded same-shape reuse without
        # introducing a CPU task or shared/finalize contribution.
        expected_hashes = {row["case"]: row["hash"] for row in results}
        lifecycle = []
        cases = frozen_native_partial_cases()
        for iteration in range(parsed.lifecycle_rounds):
            case = cases[iteration % len(cases)]
            row = _run_case(
                case=case,
                owner=owner,
                comm=comm,
                expert_map=expert_map,
                reference_weights=references,
                seed=18000 + (iteration % len(cases)),
                repeats=1,
            )
            if row["hash"] != expected_hashes[case.name]:
                raise AssertionError("native partial lifecycle changed a qualified hash")
            lifecycle.append(row["hash"])
        torch.npu.synchronize()
        method_path = Path(inspect.getsourcefile(type(owner.quant_method)) or "")
        receipt = {
            "schema": 1,
            "api": R18_NATIVE_PARTIAL_PROBE_API,
            "status": "PASS_NATIVE_RESIDENT_PARTIAL",
            "device_executed": True,
            "native_runner_preserved": True,
            "runner_device_execution": "NOT_RUN_P2_FULL_LAYER",
            "native_quant_method": type(owner.quant_method).__name__,
            "logical_experts": LOGICAL_EXPERTS,
            "physical_npu_slots": PHYSICAL_NPU_SLOTS,
            "resident_logical_ids": list(snapshot.resident_logical_ids),
            "selector_calls_per_execution": 1,
            "selector_semantics": "ORIGINAL_ASCEND_TOP10_ONCE_512_LOGICAL",
            "route_weight_application": "EXACTLY_ONCE",
            "shared_finalize_residual_in_partial": False,
            "expert_map_nonresident_sentinel": -1,
            "expert_map_semantics": "NATIVE_KERNEL_SKIPS_NONRESIDENT_OCCURRENCES",
            "method_source": str(method_path),
            "method_source_sha256": _sha256(method_path),
            "v2_receipt_sha256": v2_sha256,
            "cases": results,
            "lifecycle_rounds": parsed.lifecycle_rounds,
            "lifecycle_hash_count": len(lifecycle),
            "allocator_steady_bytes": allocator_steady,
            "allocator_peak_bytes": int(torch.npu.max_memory_allocated()),
            "relative_l2_limit": MAX_RELATIVE_L2,
            "elapsed_seconds": time.monotonic() - started,
            "cpu_complement": "NOT_RUN_P2",
            "full_decoder_layer": "NOT_RUN_P2",
            "full_model": "NOT_RUN_P2",
        }
        validate_native_partial_receipt(receipt)
        return receipt
    finally:
        for name, value in prior_extra.items():
            setattr(_EXTRA_CTX, name, value)
        if forward_scope is not None:
            forward_scope.__exit__(None, None, None)
        if model_parallel_initialized:
            destroy_model_parallel()
        if distributed_initialized:
            destroy_distributed_environment()
        if config_scope is not None:
            config_scope.__exit__(None, None, None)
        binding.cleanup()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-device", action="store_true")
    parser.add_argument("--source-token", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--v2-receipt", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--lifecycle-rounds", type=int, default=20)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    if not parsed.enable_device:
        raise ValueError("Round18 native partial probe requires --enable-device")
    _require_enable(os.environ)
    if parsed.lifecycle_rounds != 20:
        raise ValueError("Round18 native partial probe requires 20 lifecycle rounds")
    result = run_device_probe(parsed)
    _write_json(parsed.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("R18_NATIVE_PARTIAL_PROBE_API", "main", "run_device_probe")
