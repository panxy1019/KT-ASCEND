"""Real-image Round16-B V3 logical512/physical-Q32 Hybrid MoE probe.

This is a bounded one-layer component entrypoint.  It consumes the verified V2
constructor/loader identity, rebuilds one physical-Q32 owner in the exact
vLLM-Ascend image, binds the exact TP1 ALLGATHER resident provider, loads the
existing C128/MXFP4 layer bank, and executes all-NPU, all-CPU, and mixed Top-10
occurrence cases.  It does not construct a full model or enable Graph/EPLB.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from round16.b.allgather_q32_device_adapter import (
    ALLGATHER_BUILD_FUSED_INPUT_API,
    ALLGATHER_STREAM_IDENTITY_API,
    AllGatherQ32DeviceTensorAdapter,
    AllGatherQ32LayerBinding,
    AllGatherQ32TensorBinding,
    DeviceSelectedTopK,
)
from round16.b.q32_hybrid_contract import (
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    frozen_route_cases,
    validate_layer_bank_catalog,
    validate_v2_receipt,
)
from round16.b.source_bound_hybrid_runner import SOURCE_BOUND_MOE_COMM_CONTEXT_API
from round16_integration.b_q32_loader_live_factory import create_q32_loader_binding
from round16_integration.b_q32_loader_probe import Q32V2FactoryContext
from round16_integration.p1_allgather_q32_probe import (
    DevicePhysicalTopKPayloadBuilder,
    TorchNpuTensorOps,
    load_snapshot_manifest,
    load_source_preflight_token,
)


Q32_V3_PROBE_API = "round16-integration:q32-hybrid-live-probe:v1"
MAX_RELATIVE_L2 = 0.01


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _require_device_enable(environment: Mapping[str, str]) -> None:
    if environment.get("ROUND16_B_ENABLE_Q32_V3") != "1":
        raise RuntimeError("ROUND16_B_ENABLE_Q32_V3=1 is required")


def _hash_tensor(value: Any) -> str:
    # Reinterpret through uint8 without relying on a dtype-specific NumPy path.
    raw = value.detach().contiguous().cpu().view(__import__("torch").uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _relative_l2(actual: Any, expected: Any) -> float:
    torch = __import__("torch")
    numerator = (actual.float() - expected.float()).norm()
    denominator = expected.float().norm().clamp_min(1e-30)
    return float((numerator / denominator).item())


@dataclass
class _ReferenceWeights:
    bf16: dict[int, tuple[Any, Any, Any]]
    mxfp4: dict[int, tuple[Any, Any, Any]]
    q8_activation: Any


def _load_reference_weights(
    *,
    model_root: Path,
    bank_path: Path,
    logical_ids: tuple[int, ...],
    layer_id: int,
) -> _ReferenceWeights:
    import gguf
    import torch
    from gguf.quants import dequantize
    from safetensors import safe_open
    from kt_kernel import kt_kernel_ext

    index = json.loads((model_root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    base = f"model.language_model.layers.{layer_id}.mlp.experts."
    gate_key = base + "gate_up_proj"
    down_key = base + "down_proj"
    gate_path = model_root / weight_map[gate_key]
    down_path = model_root / weight_map[down_key]
    bf16: dict[int, tuple[Any, Any, Any]] = {}
    resident_ids = tuple(value for value in logical_ids if value < 32)
    with safe_open(str(gate_path), framework="pt", device="cpu") as gate_file, safe_open(
        str(down_path), framework="pt", device="cpu"
    ) as down_file:
        gate_slice = gate_file.get_slice(gate_key)
        down_slice = down_file.get_slice(down_key)
        for expert_id in resident_ids:
            gate_up = gate_slice[expert_id]
            gate, up = gate_up.float().chunk(2, dim=0)
            down = down_slice[expert_id].float()
            bf16[expert_id] = (gate.contiguous(), up.contiguous(), down.contiguous())

    reader = gguf.GGUFReader(str(bank_path))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    mxfp4: dict[int, tuple[Any, Any, Any]] = {}
    cpu_ids = tuple(value for value in logical_ids if value >= 32)
    for expert_id in cpu_ids:
        pieces = []
        for projection in ("gate", "up", "down"):
            tensor = tensors[f"blk.{layer_id}.ffn_{projection}_exps.weight"]
            if int(tensor.tensor_type) != 39:
                raise ValueError("V3 CPU reference source is not MXFP4")
            raw = tensor.data.reshape(512, -1)[expert_id]
            shape = (2560, 640) if projection == "down" else (640, 2560)
            value = torch.from_numpy(
                dequantize(raw, tensor.tensor_type).copy()
            ).reshape(shape)
            pieces.append(value.contiguous())
        mxfp4[expert_id] = tuple(pieces)  # type: ignore[assignment]

    library = ctypes.CDLL(kt_kernel_ext.__file__)
    quantize = library.quantize_row_q8_0
    dequantize_q8 = library.dequantize_row_q8_0
    for function in (quantize, dequantize_q8):
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

    def q8_activation(value: Any) -> Any:
        source = value.float().contiguous()
        if source.numel() % 32:
            raise ValueError("Q8 activation reference requires a block32 tensor")
        packed = torch.empty(source.numel() // 32 * 34, dtype=torch.uint8)
        result = torch.empty_like(source)
        quantize(source.data_ptr(), packed.data_ptr(), source.numel())
        dequantize_q8(packed.data_ptr(), result.data_ptr(), source.numel())
        return result

    return _ReferenceWeights(bf16=bf16, mxfp4=mxfp4, q8_activation=q8_activation)


def _reference_partials(
    hidden_cpu: Any,
    route_ids: tuple[int, ...],
    route_weights_cpu: Any,
    weights: _ReferenceWeights,
) -> tuple[Any, Any, Any]:
    import torch

    cpu = torch.zeros(hidden_cpu.shape[0], HIDDEN_SIZE, dtype=torch.float32)
    npu = torch.zeros_like(cpu)
    for column, expert_id in enumerate(route_ids):
        is_cpu = expert_id >= 32
        gate, up, down = (weights.mxfp4 if is_cpu else weights.bf16)[expert_id]
        source = weights.q8_activation(hidden_cpu) if is_cpu else hidden_cpu.float()
        gate_out = (source @ gate.T).bfloat16().float()
        up_out = (source @ up.T).bfloat16().float()
        intermediate = (torch.nn.functional.silu(gate_out) * up_out).bfloat16().float()
        if is_cpu:
            intermediate = weights.q8_activation(intermediate)
        expert_out = intermediate @ down.T
        if not is_cpu:
            expert_out = expert_out.bfloat16().float()
        weighted = expert_out * route_weights_cpu[:, column, None].float()
        (cpu if is_cpu else npu).add_(weighted)
    # The live bridge casts the C++ FP32 CPU partial to the native BF16
    # boundary before the one visible merge.  Match that exact boundary here.
    cpu_bf16 = cpu.bfloat16()
    npu_bf16 = npu.bfloat16()
    merged = (cpu_bf16 + npu_bf16).bfloat16()
    return cpu_bf16, npu_bf16, merged


def _build_live_layer(
    *,
    v2_binding: Any,
    source_token: Any,
    torch_module: Any,
) -> tuple[Any, Any, Any, Any]:
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
    from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl
    from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
    from vllm_ascend.quantization.quant_type import QuantType

    owner = v2_binding.owner
    comm = AllGatherCommImpl(owner.moe_config)
    prior = {
        name: getattr(_EXTRA_CTX, name, None)
        for name in ("moe_comm_method", "moe_comm_type", "flash_comm_v1_enabled", "in_profile_run")
    }
    _EXTRA_CTX.moe_comm_method = comm
    _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
    _EXTRA_CTX.flash_comm_v1_enabled = False
    _EXTRA_CTX.in_profile_run = False
    stream = torch_module.npu.current_stream()

    def comm_provider() -> Any:
        return _EXTRA_CTX.moe_comm_method

    comm_provider.round16_api_identity = SOURCE_BOUND_MOE_COMM_CONTEXT_API

    def stream_provider() -> Any:
        return stream

    stream_provider.round16_api_identity = ALLGATHER_STREAM_IDENTITY_API

    def fused_input_builder(**kwargs: Any) -> Any:
        return build_fused_experts_input(**kwargs)

    fused_input_builder.round16_api_identity = ALLGATHER_BUILD_FUSED_INPUT_API

    layer = AllGatherQ32LayerBinding(
        layer_name="model.layers.0.mlp.experts",
        plan=owner.round16_selective_plan,
        routed_owner=owner,
        comm_method=comm,
        comm_method_provider=comm_provider,
        stream_identity_provider=stream_provider,
        build_fused_experts_input=fused_input_builder,
        source_preflight_token=source_token,
        quant_none=QuantType.NONE,
        activation="silu",
    )
    payload = DevicePhysicalTopKPayloadBuilder(
        owner.round16_selective_plan,
        TorchNpuTensorOps(torch_module),
    )
    adapter = AllGatherQ32DeviceTensorAdapter(
        AllGatherQ32TensorBinding(layer, payload)
    )

    def restore_context() -> None:
        for name, value in prior.items():
            setattr(_EXTRA_CTX, name, value)

    return layer, adapter, stream, restore_context


def _enter_bounded_forward_context() -> tuple[Any, Any]:
    """Enter the real vLLM forward-context storage used by Ascend's proxy.

    V3 is a one-layer component probe rather than a model-runner invocation, so
    there is no ambient scheduler-created forward context.  Construct the
    smallest upstream ``ForwardContext`` directly and scope it with upstream's
    ``override_forward_context``.  The Ascend ``_EXTRA_CTX`` proxy can then
    retain the exact TP1 AllGather method selected by this probe.  The caller
    must exit the returned context manager before releasing the owner.
    """

    from vllm.forward_context import (
        ForwardContext,
        get_forward_context,
        override_forward_context,
    )

    state = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
    )
    scope = override_forward_context(state)
    scope.__enter__()
    if get_forward_context() is not state:
        scope.__exit__(None, None, None)
        raise RuntimeError("V3 failed to install the bounded vLLM forward context")
    return scope, state


def _run_case(
    *,
    case: Any,
    adapter: Any,
    layer: Any,
    stream_handle: int,
    wrapper: Any,
    reference_weights: _ReferenceWeights,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    import torch
    from round16_integration.p1_allgather_q32_probe import DeviceSelectedTopK

    torch.manual_seed(seed)
    hidden_cpu = torch.randn(1, HIDDEN_SIZE, dtype=torch.float32).bfloat16()
    route_weights_cpu = torch.softmax(torch.randn(1, 10), dim=-1).float()
    hidden = hidden_cpu.to("npu:0")
    ids = torch.tensor(case.logical_ids, dtype=torch.int32, device="npu:0").view(1, 10)
    route_weights = route_weights_cpu.to("npu:0")
    router_logits = torch.zeros((1, 512), dtype=torch.float32, device="npu:0")
    expected_cpu, expected_npu, expected_merged = _reference_partials(
        hidden_cpu,
        case.logical_ids,
        route_weights_cpu,
        reference_weights,
    )
    rows: list[dict[str, Any]] = []
    hashes: list[str] = []
    for repeat in range(repeats):
        prepared = adapter.prepare(hidden, router_logits, layer.layer_name)
        selected = DeviceSelectedTopK(ids, route_weights, prepared.valid_rows)
        pending = False
        try:
            wrapper.submit_forward(
                prepared.selector_hidden_states,
                selected.topk_ids,
                selected.topk_weights,
                stream_handle,
            )
            pending = True
            native = adapter.execute(
                prepared,
                selected,
                prepared.selector_hidden_states,
                ("v3", case.name, repeat),
            )
            cpu_fp32 = wrapper.sync_forward(
                prepared.selector_hidden_states,
                stream_handle,
            )
            pending = False
            cpu_partial = cpu_fp32.to(dtype=native.output.dtype)
            merged = native.output + cpu_partial
            torch.npu.synchronize()
        except BaseException:
            if pending:
                wrapper.sync_forward(prepared.selector_hidden_states, stream_handle)
            try:
                prepared.cleanup(prepared)
            except BaseException:
                pass
            raise
        actual_cpu = cpu_partial.detach().cpu()
        actual_npu = native.output.detach().cpu()
        actual_merged = merged.detach().cpu()
        errors = {
            "cpu_partial": _relative_l2(actual_cpu, expected_cpu),
            "npu_partial": _relative_l2(actual_npu, expected_npu),
            "merged": _relative_l2(actual_merged, expected_merged),
        }
        if any(value > MAX_RELATIVE_L2 for value in errors.values()):
            raise AssertionError(f"V3 {case.name} numerical error: {errors}")
        finite = bool(torch.isfinite(actual_merged).all())
        if not finite:
            raise AssertionError("V3 merged output is not finite")
        digest = _hash_tensor(actual_merged)
        hashes.append(digest)
        rows.append({"repeat": repeat, "errors": errors, "hash": digest, "finite": finite})
    if len(set(hashes)) != 1:
        raise AssertionError(f"V3 {case.name} output hash is not repeat-exact")
    return {
        "case": case.name,
        "logical_ids": list(case.logical_ids),
        "resident_occurrences": sum(value < 32 for value in case.logical_ids),
        "cpu_occurrences": sum(value >= 32 for value in case.logical_ids),
        "rows": rows,
        "repeat_exact": True,
        "hash": hashes[0],
    }


def run_device_probe(parsed: argparse.Namespace) -> dict[str, Any]:
    source_token = load_source_preflight_token(parsed.source_token)
    snapshot = load_snapshot_manifest(parsed.snapshot_manifest)
    v2_receipt, v2_sha256 = validate_v2_receipt(parsed.v2_receipt)
    bank_identity = validate_layer_bank_catalog(
        parsed.bank_catalog,
        parsed.bank,
        layer_id=parsed.layer,
    )
    if tuple(v2_receipt["resident_logical_ids"]) != snapshot.resident_logical_ids:
        raise ValueError("V3 V2 receipt and S0 snapshot resident maps differ")
    if parsed.layer != 0:
        raise ValueError("the first V3 live probe is restricted to layer0")

    import torch
    import torch_npu
    from kt_kernel import get_current_device_stream_handle, kt_kernel_ext
    from kt_kernel.experts_base import BaseMoEWrapper
    from kt_kernel.utils.llamafile import LlamafileMoEWrapper
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_ascend.ascend_config import init_ascend_config

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(0.10, 0)
    torch.npu.config.allow_internal_format = True
    if not bool(kt_kernel_ext.llamafile_block32_tail_supported()):
        raise RuntimeError("V3 KT native lacks MXFP4 block32 tail support")
    if not bool(kt_kernel_ext.llamafile_bf16_f32_tail_supported()):
        raise RuntimeError("V3 KT native lacks BF16/F32 tail support")

    v2_context = Q32V2FactoryContext(
        source_token=source_token,
        snapshot=snapshot,
        layer_id=parsed.layer,
        checkpoint_root=parsed.model_root,
        v1_receipt_sha256=str(v2_receipt["v1_receipt_sha256"]),
    )
    v2_binding = create_q32_loader_binding(v2_context)
    restore_context = None
    config_context = None
    forward_context_scope = None
    wrapper = None
    start = time.monotonic()
    try:
        vllm_config = VllmConfig(additional_config={"weight_nz_mode": 2})
        config_context = set_current_vllm_config(vllm_config)
        config_context.__enter__()
        ascend_config = init_ascend_config(vllm_config)
        if getattr(ascend_config, "weight_nz_mode", None) != 2:
            raise RuntimeError("V3 did not restore frozen BF16 NZ mode 2")
        forward_context_scope, _forward_context_state = (
            _enter_bounded_forward_context()
        )
        layer, adapter, _stream, restore_context = _build_live_layer(
            v2_binding=v2_binding,
            source_token=source_token,
            torch_module=torch,
        )
        mask = torch.zeros(512, dtype=torch.bool, device="cpu")
        mask[:32] = True
        BaseMoEWrapper.set_capture_batch_sizes([1])
        with torch.device("cpu"):
            wrapper = LlamafileMoEWrapper(
                layer_idx=0,
                num_experts=512,
                num_experts_per_tok=10,
                hidden_size=HIDDEN_SIZE,
                moe_intermediate_size=INTERMEDIATE_SIZE,
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
        # The public KT stream helper deliberately requires the framework
        # device type; do not infer CUDA/NPU from an ambient default.
        stream_handle = int(get_current_device_stream_handle("npu"))
        cases = frozen_route_cases()
        all_ids = tuple(sorted({value for case in cases for value in case.logical_ids}))
        references = _load_reference_weights(
            model_root=parsed.model_root,
            bank_path=parsed.bank,
            logical_ids=all_ids,
            layer_id=0,
        )
        torch.npu.synchronize()
        allocator_steady = int(torch.npu.memory_allocated())
        torch.npu.reset_peak_memory_stats()
        results = [
            _run_case(
                case=case,
                adapter=adapter,
                layer=layer,
                stream_handle=stream_handle,
                wrapper=wrapper,
                reference_weights=references,
                seed=16000 + index,
                repeats=3,
            )
            for index, case in enumerate(cases)
        ]

        # Lifecycle proof: one drained-and-discarded mixed result followed by
        # twenty same-shape alternating calls.  Each visible call must match the
        # already-qualified case hash; the discarded result is never published.
        mixed = cases[2]
        _run_case(
            case=mixed,
            adapter=adapter,
            layer=layer,
            stream_handle=stream_handle,
            wrapper=wrapper,
            reference_weights=references,
            seed=16002,
            repeats=1,
        )
        lifecycle_hashes: list[str] = []
        expected_by_case = {row["case"]: row["hash"] for row in results}
        for iteration in range(parsed.lifecycle_rounds):
            case = cases[iteration % len(cases)]
            row = _run_case(
                case=case,
                adapter=adapter,
                layer=layer,
                stream_handle=stream_handle,
                wrapper=wrapper,
                reference_weights=references,
                seed=16000 + (iteration % len(cases)),
                repeats=1,
            )
            if row["hash"] != expected_by_case[case.name]:
                raise AssertionError("V3 lifecycle reuse changed a qualified case hash")
            lifecycle_hashes.append(row["hash"])
        torch.npu.synchronize()
        allocator_peak = int(torch.npu.max_memory_allocated())
        return {
            "schema": 1,
            "status": "REAL_HYBRID_MOE_VERIFIED",
            "stage": "V3",
            "device_executed": True,
            "logical_experts": 512,
            "physical_npu_slots": 32,
            "top_k": 10,
            "resident_logical_ids": list(snapshot.resident_logical_ids),
            "v2_receipt_sha256": v2_sha256,
            "bank_identity": bank_identity,
            "kt_native_sha256": hashlib.sha256(Path(kt_kernel_ext.__file__).read_bytes()).hexdigest(),
            "cpu_workers": 128,
            "threadpool_count": 4,
            "numa_nodes": list(parsed.numa_nodes),
            "npu_provider": "AllGatherCommImpl/AscendUnquantizedFusedMoEMethod",
            "npu_weight_format": 29,
            "cpu_provider": "LlamafileMoEWrapper/MXFP4",
            "selector_semantics": "PRESELECTED_TOP10_ONCE_NO_RETOPK_NO_RENORMALIZE",
            "forward_context": "BOUNDED_UPSTREAM_VLLM_FORWARD_CONTEXT_COMPONENT_SCOPE",
            "route_weight_application": "ONCE_PER_OCCURRENCE_BEFORE_SINGLE_LAYER_MERGE",
            "routed_scaling_applied": False,
            "cases": results,
            "drained_discarded_results": 1,
            "lifecycle_rounds": parsed.lifecycle_rounds,
            "lifecycle_hash_count": len(lifecycle_hashes),
            "late_result_published": False,
            "allocator_steady_bytes": allocator_steady,
            "allocator_peak_bytes": allocator_peak,
            "allocator_peak_delta_bytes": max(0, allocator_peak - allocator_steady),
            "relative_l2_limit": MAX_RELATIVE_L2,
            "elapsed_seconds": time.monotonic() - start,
            "all_layer_residency": "NOT_RUN_V4",
            "full_model": "NOT_RUN_V5",
        }
    finally:
        if restore_context is not None:
            restore_context()
        if forward_context_scope is not None:
            forward_context_scope.__exit__(None, None, None)
        if config_context is not None:
            config_context.__exit__(None, None, None)
        v2_binding.cleanup()


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
        raise ValueError("V3 requires --enable-device")
    _require_device_enable(os.environ)
    parsed.numa_nodes = tuple(parsed.numa_nodes)
    if parsed.numa_nodes != (2, 3, 4, 7):
        raise ValueError("V3 requires the frozen C128 NUMA nodes 2/3/4/7")
    if parsed.lifecycle_rounds != 20:
        raise ValueError("V3 requires exactly 20 lifecycle reuse rounds")
    result = run_device_probe(parsed)
    _write_json(parsed.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("Q32_V3_PROBE_API", "main", "run_device_probe")
