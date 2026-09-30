"""Real-image Round18 P2 native resident + KT CPU complement component.

The probe uses the exact frozen Ascend quant method and AllGather prepare /
finalize implementation.  ``NativeHybridQuantMethodAdapter`` observes the
single original Top-K call, submits the complement to the existing
C128/MXFP4 wrapper, and adds it once to the native physical-Q32 partial.  No
custom MoE runner is constructed by this component.
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

from round16.b.q32_hybrid_contract import (
    HIDDEN_SIZE,
    validate_layer_bank_catalog,
    validate_v2_receipt,
)
from round16_integration.b_q32_hybrid_live_probe import (
    _load_reference_weights,
    _reference_partials,
)
from round16_integration.b_q32_loader_live_factory import create_q32_loader_binding
from round16_integration.b_q32_loader_probe import Q32V2FactoryContext
from round16_integration.p1_allgather_q32_probe import (
    KTDeviceWeightedPartial,
    load_snapshot_manifest,
    load_source_preflight_token,
)
from round18.native_hybrid_quant_method import NativeHybridQuantMethodAdapter
from round18.native_partial_contract import frozen_native_partial_cases
from round18.native_runner_seam import execution_map_from_owner
from round18_integration.native_partial_probe import (
    _enter_bounded_forward_context,
    _hash_tensor,
    _relative_l2,
    _router_logits,
)


R18_NATIVE_HYBRID_COMPONENT_API = (
    "round18:native-ascend-runner-kt-complement-component:v1"
)
MAX_RELATIVE_L2 = 0.01


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _require_enable(environment: Mapping[str, str]) -> None:
    if environment.get("ROUND18_ENABLE_NATIVE_HYBRID") != "1":
        raise RuntimeError("ROUND18_ENABLE_NATIVE_HYBRID=1 is required")


def _hybrid_reference(
    hidden_cpu: Any,
    topk_ids_cpu: Any,
    topk_weights_cpu: Any,
    weights: Any,
) -> Any:
    import torch

    rows = []
    for row in range(int(hidden_cpu.shape[0])):
        _cpu, _npu, merged = _reference_partials(
            hidden_cpu[row : row + 1],
            tuple(int(value) for value in topk_ids_cpu[row].tolist()),
            topk_weights_cpu[row : row + 1].float(),
            weights,
        )
        rows.append(merged)
    return torch.cat(rows, dim=0)


def _run_case(
    *,
    case: Any,
    adapter: NativeHybridQuantMethodAdapter,
    owner: Any,
    comm: Any,
    reference_weights: Any,
    capture: dict[str, Any],
    torch_module: Any,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    torch_module.manual_seed(seed)
    hidden_cpu = torch_module.randn(
        len(case.rows), HIDDEN_SIZE, dtype=torch_module.float32
    ).bfloat16()
    hidden = hidden_cpu.to("npu:0")
    router_logits = _router_logits(case, torch_module)
    hashes: list[str] = []
    rows: list[dict[str, Any]] = []
    observed_sets: list[list[int]] | None = None

    for repeat in range(repeats):
        capture.clear()
        capture["current"] = True
        capture["tag"] = (case.name, repeat)
        prepared = comm.prepare(
            hidden_states=hidden,
            router_logits=router_logits,
            replace_allreduce=False,
            enable_shared_expert_dp=False,
        )
        fused_result = adapter.apply(
            layer=owner,
            x=prepared.hidden_states,
            router_logits=prepared.router_logits,
            top_k=10,
            renormalize=True,
            use_grouped_topk=False,
            num_experts=512,
            expert_map=execution_map_from_owner(
                owner, owner.round16_selective_plan
            ),
            routed_scaling_factor=1.0,
            activation="silu",
        )
        actual = comm.finalize(
            hidden_states=fused_result.routed_out,
            reduce_results=True,
            padded_hidden_states_shape=prepared.padded_hidden_states_shape,
        )
        torch_module.npu.synchronize()

        ids = capture.get("topk_ids")
        weights = capture.get("topk_weights")
        if ids is None or weights is None:
            raise RuntimeError("P2 CPU submission did not preserve selected routes")
        ids_cpu = ids.cpu()
        weights_cpu = weights.float().cpu()
        current_sets = [
            sorted(int(value) for value in route) for route in ids_cpu.tolist()
        ]
        expected_sets = [sorted(route) for route in case.rows]
        if current_sets != expected_sets:
            raise AssertionError(
                f"native selector changed frozen {case.name} routes: {current_sets}"
            )
        if observed_sets is None:
            observed_sets = current_sets
        expected = _hybrid_reference(
            hidden_cpu, ids_cpu, weights_cpu, reference_weights
        )
        actual_cpu = actual.detach().cpu()
        error = _relative_l2(actual_cpu, expected)
        if error > MAX_RELATIVE_L2:
            raise AssertionError(
                f"native Hybrid {case.name} relative L2 {error} exceeds {MAX_RELATIVE_L2}"
            )
        if not bool(torch_module.isfinite(actual_cpu).all()):
            raise AssertionError("native Hybrid output is not finite")
        digest = _hash_tensor(actual_cpu)
        hashes.append(digest)
        rows.append(
            {
                "repeat": repeat,
                "relative_l2": error,
                "hash": digest,
                "work_identity": list(capture["identity_key"]),
            }
        )

    if len(set(hashes)) != 1:
        raise AssertionError(f"native Hybrid {case.name} is not repeat-exact")
    return {
        "case": case.name,
        "token_rows": len(case.rows),
        "resident_occurrences": sum(
            value < 32 for route in case.rows for value in route
        ),
        "cpu_occurrences": sum(
            value >= 32 for route in case.rows for value in route
        ),
        "actual_id_sets": observed_sets,
        "repeat_exact": True,
        "hash": hashes[0],
        "rows": rows,
    }


def run_device_probe(parsed: argparse.Namespace) -> dict[str, Any]:
    source_token = load_source_preflight_token(parsed.source_token)
    snapshot = load_snapshot_manifest(parsed.snapshot_manifest)
    v2_receipt, v2_sha256 = validate_v2_receipt(parsed.v2_receipt)
    bank_identity = validate_layer_bank_catalog(
        parsed.bank_catalog, parsed.bank, layer_id=parsed.layer
    )
    if tuple(v2_receipt["resident_logical_ids"]) != snapshot.resident_logical_ids:
        raise ValueError("P2 V2 receipt and S0 resident maps differ")
    if parsed.layer != 0:
        raise ValueError("the first native Hybrid probe is restricted to layer0")

    import torch
    import torch_npu
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    import vllm_ascend.ops.fused_moe.fused_moe as fused_moe_module
    from kt_kernel import get_current_device_stream_handle, kt_kernel_ext
    from kt_kernel.experts_base import BaseMoEWrapper
    from kt_kernel.utils.llamafile import LlamafileMoEWrapper
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
    if not bool(kt_kernel_ext.llamafile_block32_tail_supported()):
        raise RuntimeError("P2 KT native lacks MXFP4 block32 tail support")
    if not bool(kt_kernel_ext.llamafile_bf16_f32_tail_supported()):
        raise RuntimeError("P2 KT native lacks BF16/F32 tail support")

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
    wrapper = None
    distributed_initialized = False
    model_parallel_initialized = False
    prior_extra: dict[str, Any] = {}
    started = time.monotonic()
    try:
        vllm_config = VllmConfig(additional_config={"weight_nz_mode": 2})
        config_scope = set_current_vllm_config(vllm_config)
        config_scope.__enter__()
        ascend_config = init_ascend_config(vllm_config)
        if getattr(ascend_config, "weight_nz_mode", None) != 2:
            raise RuntimeError("P2 did not restore BF16 NZ mode 2")
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29919",
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

        mask = torch.zeros(512, dtype=torch.bool, device="cpu")
        for logical_id in snapshot.resident_logical_ids:
            mask[logical_id] = True
        BaseMoEWrapper.set_capture_batch_sizes([1, 3])
        with torch.device("cpu"):
            wrapper = LlamafileMoEWrapper(
                layer_idx=0,
                num_experts=512,
                num_experts_per_tok=10,
                hidden_size=HIDDEN_SIZE,
                moe_intermediate_size=640,
                gpu_experts_mask=mask,
                cpuinfer_threads=128,
                threadpool_count=4,
                weight_path=str(parsed.bank),
                chunked_prefill_size=32,
                cpu_save=False,
                max_deferred_experts_per_token=0,
                method="LLAMAFILE",
                numa_nodes=list(parsed.numa_nodes),
            )
        wrapper.load_weights(torch.arange(512, dtype=torch.int32, device="cpu"))
        stream_handle = int(get_current_device_stream_handle("npu"))
        capture: dict[str, Any] = {"current": True}

        def identity_provider(**kwargs: Any) -> tuple[Any, ...]:
            tag = capture.get("tag", ("unknown", -1))
            return (
                "round18-p2",
                0,
                kwargs["layer_name"],
                int(kwargs["generation"]),
                str(tag[0]),
                int(tag[1]),
            )

        def cpu_submit(**kwargs: Any) -> dict[str, Any]:
            capture["topk_ids"] = kwargs["topk_ids"].detach().clone()
            capture["topk_weights"] = kwargs["topk_weights"].detach().clone()
            capture["identity_key"] = tuple(kwargs["identity_key"])
            wrapper.submit_forward(
                kwargs["hidden_states"],
                kwargs["topk_ids"],
                kwargs["topk_weights"],
                stream_handle,
            )
            return {
                "identity_key": tuple(kwargs["identity_key"]),
                "hidden_states": kwargs["hidden_states"],
            }

        def cpu_sync(**kwargs: Any) -> KTDeviceWeightedPartial:
            pending = kwargs["pending"]
            if tuple(kwargs["identity_key"]) != pending["identity_key"]:
                raise RuntimeError("P2 CPU completion identity changed")
            output = wrapper.sync_forward(
                kwargs["reference_tensor"], stream_handle
            ).to(dtype=kwargs["reference_tensor"].dtype)
            return KTDeviceWeightedPartial(output)

        def cpu_abort(**kwargs: Any) -> None:
            pending = kwargs["pending"]
            reference = kwargs.get("reference_tensor")
            if reference is None:
                reference = pending["hidden_states"]
            wrapper.sync_forward(reference, stream_handle)

        adapter = NativeHybridQuantMethodAdapter(
            delegate=owner.quant_method,
            selector_module=fused_moe_module,
            layer_name="model.layers.0.mlp.experts",
            identity_provider=identity_provider,
            cpu_submit=cpu_submit,
            cpu_sync=cpu_sync,
            cpu_abort=cpu_abort,
            identity_is_current=lambda _identity: bool(capture["current"]),
        )
        all_ids = tuple(
            sorted(
                {
                    expert_id
                    for case in frozen_native_partial_cases()
                    for route in case.rows
                    for expert_id in route
                }
            )
        )
        references = _load_reference_weights(
            model_root=parsed.model_root,
            bank_path=parsed.bank,
            logical_ids=all_ids,
            layer_id=0,
        )
        torch.npu.synchronize()
        allocator_steady = int(torch.npu.memory_allocated())
        torch.npu.reset_peak_memory_stats()
        cases = frozen_native_partial_cases()
        results = [
            _run_case(
                case=case,
                adapter=adapter,
                owner=owner,
                comm=comm,
                reference_weights=references,
                capture=capture,
                torch_module=torch,
                seed=18200 + index,
                repeats=3,
            )
            for index, case in enumerate(cases)
        ]
        expected_hashes = {row["case"]: row["hash"] for row in results}
        lifecycle = []
        for iteration in range(parsed.lifecycle_rounds):
            case = cases[iteration % len(cases)]
            row = _run_case(
                case=case,
                adapter=adapter,
                owner=owner,
                comm=comm,
                reference_weights=references,
                capture=capture,
                torch_module=torch,
                seed=18200 + (iteration % len(cases)),
                repeats=1,
            )
            if row["hash"] != expected_hashes[case.name]:
                raise AssertionError("P2 lifecycle changed a qualified hash")
            lifecycle.append(row["hash"])
        torch.npu.synchronize()
        method_path = Path(inspect.getsourcefile(type(adapter.delegate)) or "")
        return {
            "schema": 1,
            "api": R18_NATIVE_HYBRID_COMPONENT_API,
            "status": "PASS_NATIVE_HYBRID_EAGER_COMPONENT",
            "device_executed": True,
            "native_runner_replaced": False,
            "native_quant_method": type(adapter.delegate).__name__,
            "native_method_source": str(method_path),
            "native_method_sha256": hashlib.sha256(method_path.read_bytes()).hexdigest(),
            "logical_experts": 512,
            "physical_npu_slots": 32,
            "top_k": 10,
            "selector_calls_per_execution": 1,
            "route_weight_application": "EXACTLY_ONCE_PER_BRANCH",
            "routed_scaling_in_adapter": False,
            "shared_finalize_residual_in_adapter": False,
            "cpu_provider": "LlamafileMoEWrapper/MXFP4",
            "cpu_workers": 128,
            "threadpool_count": 4,
            "numa_nodes": list(parsed.numa_nodes),
            "bank_identity": bank_identity,
            "v2_receipt_sha256": v2_sha256,
            "cases": results,
            "lifecycle_rounds": parsed.lifecycle_rounds,
            "lifecycle_hash_count": len(lifecycle),
            "allocator_steady_bytes": allocator_steady,
            "allocator_peak_bytes": int(torch.npu.max_memory_allocated()),
            "relative_l2_limit": MAX_RELATIVE_L2,
            "elapsed_seconds": time.monotonic() - started,
            "full_decoder_layer": "NOT_RUN_P2_NEXT",
            "full_model": "NOT_RUN_P2_NEXT",
            "graph_mode": "NONE_COMPONENT",
        }
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
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--numa-nodes", type=int, nargs=4, default=(2, 3, 4, 7))
    parser.add_argument("--lifecycle-rounds", type=int, default=20)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    if not parsed.enable_device:
        raise ValueError("Round18 native Hybrid probe requires --enable-device")
    _require_enable(os.environ)
    parsed.numa_nodes = tuple(parsed.numa_nodes)
    if parsed.numa_nodes != (2, 3, 4, 7):
        raise ValueError("P2 requires frozen C128 NUMA nodes 2/3/4/7")
    if parsed.lifecycle_rounds != 20:
        raise ValueError("P2 requires exactly 20 lifecycle rounds")
    result = run_device_probe(parsed)
    _write_json(parsed.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("R18_NATIVE_HYBRID_COMPONENT_API", "main", "run_device_probe")
