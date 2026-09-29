"""Bounded TP1/EP1 ALLGATHER physical-Q32 device probe.

Importing this module imports no torch, torch_npu, vLLM, or vLLM-Ascend.  The
CLI validates the frozen source token and S0 snapshot before it imports an
explicit runtime binding factory.  Device execution remains default OFF.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from round16.b.allgather_q32_device_adapter import (
    ALLGATHER_PHYSICAL_TENSOR_PAYLOAD_API,
    AllGatherQ32DeviceTensorAdapter,
    AllGatherQ32LayerBinding,
    AllGatherQ32TensorBinding,
    DevicePhysicalTopKPayload,
    DeviceSelectedTopK,
    DeviceWeightedPartial,
)
from round16.b.native_source_preflight import (
    DEFAULT_MODEL_CONFIG_EXPECTATION,
    DEFAULT_SOURCE_EXPECTATIONS,
    PreflightResult,
    SourcePreflightToken,
    validate_exact_source_preflight_token,
)
from round16.b.selective_loader import SelectiveLoadPlan


P1_PROBE_API = "round16-integration:p1-allgather-q32-probe:v1"
P1_TENSOR_OPS_API = "round16-integration:p1-npu-tensor-ops:v1"
P1_TENSOR_MERGE_API = "round16-integration:p1-tensor-weighted-merge:v1"
P1_SELECTOR_API = "round16-integration:p1-selector-once:v1"
P1_KT_SUBMIT_API = "round16-integration:p1-kt-tensor-submit:v1"
P1_KT_SYNC_API = "round16-integration:p1-kt-tensor-sync:v1"
P1_KT_ABORT_API = "round16-integration:p1-kt-tensor-abort-drain:v1"
P1_RUNTIME_FACTORY_API = "round16-integration:p1-runtime-binding-factory:v1"

EXPECTED_Q32_PHYSICAL_SHA256 = (
    "fdc73639b5d3b0f97db2e10c689b4b3a802240d72e50d3bc00aeee402d9f1872"
)
EXPECTED_EVIDENCE_ARCHIVE_SHA256 = (
    "14ad7d864258b41dc347f3b756654791fe356eb45275567381e3bd962ad82d36"
)


def _require_callable(value: Any, identity: str, name: str) -> None:
    if not callable(value):
        raise TypeError(f"{name} must be callable")
    if getattr(value, "round16_api_identity", None) != identity:
        raise ValueError(f"{name} API identity is not verified")


@dataclass(frozen=True)
class _TensorMetadata:
    shape: tuple[int, ...]
    dtype: str
    device: str


def _tensor_metadata(value: Any, name: str) -> _TensorMetadata:
    try:
        shape = tuple(value.shape)
    except (AttributeError, TypeError) as error:
        raise TypeError(f"{name} must expose tensor shape metadata") from error
    if not shape or any(type(item) is not int or item < 0 for item in shape):
        raise ValueError(f"{name} has invalid tensor shape metadata")
    dtype = str(getattr(value, "dtype", "")).lower()
    device_value = getattr(value, "device", None)
    device_type = str(getattr(device_value, "type", "")).lower()
    device = str(device_value).lower()
    if not dtype:
        raise TypeError(f"{name} must expose dtype metadata")
    if device_type != "npu" or not device.startswith("npu"):
        raise ValueError(f"{name} must remain on NPU")
    return _TensorMetadata(shape, dtype, device)


def _require_bfloat16(metadata: _TensorMetadata, name: str) -> None:
    if metadata.dtype not in ("bfloat16", "bf16", "torch.bfloat16"):
        raise ValueError(f"{name} must use bfloat16")


class TorchNpuTensorOps:
    """Tensor-only implementation with an injected torch-compatible module.

    No method calls ``item``, ``tolist``, Python iteration on a tensor, or a
    synchronization API.  The module is injected only after the CLI's static
    gates pass, so importing the probe remains device-runtime free.
    """

    round16_api_identity = P1_TENSOR_OPS_API

    def __init__(self, torch_module: Any) -> None:
        required = (
            "as_tensor",
            "clamp",
            "take",
            "ge",
            "lt",
            "logical_and",
            "arange",
            "full_like",
            "zeros_like",
            "where",
            "add",
            "int32",
            "long",
        )
        missing = tuple(name for name in required if not hasattr(torch_module, name))
        if missing:
            raise TypeError(f"torch tensor ops are incomplete: {missing}")
        self._torch = torch_module

    def _cast(self, value: Any, dtype: Any) -> Any:
        injected = getattr(self._torch, "cast", None)
        if callable(injected):
            return injected(value, dtype)
        return value.to(dtype=dtype)

    def _reshape_column(self, value: Any, rows: int) -> Any:
        injected = getattr(self._torch, "reshape", None)
        if callable(injected):
            return injected(value, (rows, 1))
        return value.reshape(rows, 1)

    def build_mapping(self, values: tuple[int, ...], reference: Any) -> Any:
        return self._torch.as_tensor(
            values,
            dtype=self._torch.int32,
            device=reference.device,
        )

    def physicalize(
        self,
        mapping: Any,
        topk_ids: Any,
        topk_weights: Any,
        *,
        valid_prefix_rows: int,
        safe_physical_id: int,
    ) -> tuple[Any, Any]:
        rows = topk_ids.shape[0]
        bounded = self._torch.clamp(topk_ids, 0, 511)
        bounded_long = self._cast(bounded, self._torch.long)
        physical = self._torch.take(mapping, bounded_long)
        logical_valid = self._torch.logical_and(
            self._torch.ge(topk_ids, 0),
            self._torch.lt(topk_ids, 512),
        )
        resident = self._torch.logical_and(
            logical_valid,
            self._torch.ge(physical, 0),
        )
        row_indices = self._torch.arange(rows, device=topk_ids.device)
        row_indices = self._reshape_column(row_indices, rows)
        valid_rows = self._torch.lt(row_indices, valid_prefix_rows)
        keep = self._torch.logical_and(resident, valid_rows)
        safe_ids = self._torch.full_like(topk_ids, safe_physical_id)
        physical_ids = self._torch.where(
            keep,
            self._cast(physical, self._torch.int32),
            safe_ids,
        )
        physical_weights = self._torch.where(
            keep,
            topk_weights,
            self._torch.zeros_like(topk_weights),
        )
        return physical_ids, physical_weights

    def add(self, left: Any, right: Any) -> Any:
        return self._torch.add(left, right)


class DevicePhysicalTopKPayloadBuilder:
    """Map logical Top-K tensors to physical Q32 entirely with tensor ops."""

    round16_api_identity = ALLGATHER_PHYSICAL_TENSOR_PAYLOAD_API

    def __init__(self, plan: SelectiveLoadPlan, tensor_ops: TorchNpuTensorOps) -> None:
        if not isinstance(plan, SelectiveLoadPlan):
            raise TypeError("physical payload builder requires SelectiveLoadPlan")
        if plan.logical_expert_count != 512 or plan.physical_expert_count != 32:
            raise ValueError("physical payload builder requires logical512/Q32")
        if not isinstance(tensor_ops, TorchNpuTensorOps):
            raise TypeError("physical payload builder requires TorchNpuTensorOps")
        self.plan = plan
        self.tensor_ops = tensor_ops
        self._mapping_by_device: dict[str, Any] = {}

    @staticmethod
    def _valid_prefix(valid_rows: tuple[bool, ...]) -> int:
        frozen = tuple(valid_rows)
        if any(type(value) is not bool for value in frozen):
            raise ValueError("valid_rows entries must be exact booleans")
        try:
            first_invalid = frozen.index(False)
        except ValueError:
            return len(frozen)
        if any(frozen[first_invalid:]):
            raise ValueError("prepared padding rows must form one trailing suffix")
        return first_invalid

    def __call__(
        self,
        *,
        topk_ids: Any,
        topk_weights: Any,
        resident_logical_ids: Sequence[int],
        logical_to_physical: Sequence[int],
        valid_rows: tuple[bool, ...],
        safe_physical_id: int,
        identity_key: tuple[Any, ...],
    ) -> DevicePhysicalTopKPayload:
        if tuple(resident_logical_ids) != self.plan.resident_logical_ids:
            raise ValueError("resident IDs differ from the frozen Q32 plan")
        if tuple(logical_to_physical) != self.plan.logical_to_physical:
            raise ValueError("logical mapping differs from the frozen Q32 plan")
        if type(safe_physical_id) is not int or not (0 <= safe_physical_id < 32):
            raise ValueError("safe physical ID must be an exact Q32 slot")
        if type(identity_key) is not tuple or not identity_key:
            raise ValueError("payload identity must be a non-empty tuple")
        ids_meta = _tensor_metadata(topk_ids, "logical topk_ids")
        weights_meta = _tensor_metadata(topk_weights, "logical topk_weights")
        if ids_meta.shape != weights_meta.shape or len(ids_meta.shape) != 2:
            raise ValueError("logical Top-K ID/weight layouts differ")
        if ids_meta.shape[1] != 10 or ids_meta.shape[0] != len(valid_rows):
            raise ValueError("logical Top-K tensors must have [prepared_T,10] layout")
        if ids_meta.dtype not in ("int32", "torch.int32"):
            raise ValueError("logical topk_ids must use int32")
        _require_bfloat16(weights_meta, "logical topk_weights")
        if ids_meta.device != weights_meta.device:
            raise ValueError("logical Top-K tensors must share one NPU")
        mapping = self._mapping_by_device.get(ids_meta.device)
        if mapping is None:
            mapping = self.tensor_ops.build_mapping(
                self.plan.logical_to_physical, topk_ids
            )
            mapping_meta = _tensor_metadata(mapping, "logical-to-physical map")
            if mapping_meta.shape != (512,) or mapping_meta.dtype not in (
                "int32",
                "torch.int32",
            ):
                raise ValueError("device logical map must be int32[512]")
            if mapping_meta.device != ids_meta.device:
                raise ValueError("device logical map is on the wrong NPU")
            self._mapping_by_device[ids_meta.device] = mapping
        physical_ids, physical_weights = self.tensor_ops.physicalize(
            mapping,
            topk_ids,
            topk_weights,
            valid_prefix_rows=self._valid_prefix(tuple(valid_rows)),
            safe_physical_id=safe_physical_id,
        )
        return DevicePhysicalTopKPayload(physical_ids, physical_weights)


@dataclass(frozen=True)
class KTDeviceWeightedPartial:
    output: Any
    route_weights_applied: bool = True
    routed_scaling_applied: bool = False

    def __post_init__(self) -> None:
        if self.route_weights_applied is not True:
            raise ValueError("KT partial must contain original Top-K weights")
        if self.routed_scaling_applied is not False:
            raise ValueError("KT partial must exclude routed scaling")


@dataclass(frozen=True)
class CombinedDeviceWeightedPartial:
    output: Any
    route_weights_applied: bool = True
    routed_scaling_applied: bool = False

    def __post_init__(self) -> None:
        if self.route_weights_applied is not True:
            raise ValueError("combined partial must contain Top-K weights")
        if self.routed_scaling_applied is not False:
            raise ValueError("combined partial must exclude routed scaling")


class TensorWeightedAggregateMerger:
    round16_api_identity = P1_TENSOR_MERGE_API

    def __init__(self, tensor_ops: TorchNpuTensorOps) -> None:
        if not isinstance(tensor_ops, TorchNpuTensorOps):
            raise TypeError("tensor merger requires TorchNpuTensorOps")
        self.tensor_ops = tensor_ops

    def merge(
        self,
        native: DeviceWeightedPartial,
        kt: KTDeviceWeightedPartial,
    ) -> CombinedDeviceWeightedPartial:
        if not isinstance(native, DeviceWeightedPartial):
            raise TypeError("native partial must be DeviceWeightedPartial")
        if not isinstance(kt, KTDeviceWeightedPartial):
            raise TypeError("KT partial must be KTDeviceWeightedPartial")
        native_meta = _tensor_metadata(native.output, "native weighted partial")
        kt_meta = _tensor_metadata(kt.output, "KT weighted partial")
        _require_bfloat16(native_meta, "native weighted partial")
        _require_bfloat16(kt_meta, "KT weighted partial")
        if native_meta != kt_meta:
            raise ValueError("native and KT weighted partial layouts differ")
        output = self.tensor_ops.add(native.output, kt.output)
        if _tensor_metadata(output, "combined weighted partial") != native_meta:
            raise ValueError("combined weighted partial layout changed")
        return CombinedDeviceWeightedPartial(output)


@dataclass(frozen=True)
class P1Snapshot:
    manifest_sha256: str
    q32_physical_sha256: str
    evidence_archive_sha256: str
    layer_count: int
    resident_logical_ids: tuple[int, ...]
    logical_to_physical: tuple[int, ...]
    w13_shape: tuple[int, ...]
    w2_shape: tuple[int, ...]
    dtype: str
    owner_path_template: str
    q32_receipt_path: Path


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def load_snapshot_manifest(path: str | Path) -> P1Snapshot:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise ValueError("P1 snapshot manifest schema is invalid")
    evidence = payload.get("s0_evidence")
    layer = payload.get("one_layer_fixture")
    if not isinstance(evidence, dict) or not isinstance(layer, dict):
        raise ValueError("P1 snapshot manifest sections are missing")
    if evidence.get("q32_physical_sha256") != EXPECTED_Q32_PHYSICAL_SHA256:
        raise ValueError("S0 Q32 receipt identity is not verified")
    if evidence.get("evidence_archive_sha256") != EXPECTED_EVIDENCE_ARCHIVE_SHA256:
        raise ValueError("S0 evidence archive identity is not verified")
    binding = payload.get("binding_sha256")
    unsigned = dict(payload)
    unsigned.pop("binding_sha256", None)
    if binding != _canonical_digest(unsigned):
        raise ValueError("P1 snapshot manifest binding is invalid")
    receipt_name = evidence.get("q32_physical_file")
    if type(receipt_name) is not str or not receipt_name:
        raise ValueError("S0 Q32 receipt file is missing from the manifest")
    receipt_relative = Path(receipt_name)
    if receipt_relative.is_absolute() or ".." in receipt_relative.parts:
        raise ValueError("S0 Q32 receipt path must remain manifest-relative")
    receipt_path = (manifest_path.parent / receipt_relative).resolve()
    if receipt_path.parent != manifest_path.parent:
        raise ValueError("S0 Q32 receipt path escapes the overlay")
    if not receipt_path.is_file() or receipt_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("S0 Q32 receipt is missing or exceeds the bounded limit")
    receipt_bytes = receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != EXPECTED_Q32_PHYSICAL_SHA256:
        raise ValueError("S0 Q32 receipt file SHA-256 is not verified")
    receipt = json.loads(receipt_bytes)
    residents = tuple(layer.get("resident_logical_ids", ()))
    mapping = tuple(layer.get("logical_to_physical", ()))
    if residents != tuple(range(32)):
        raise ValueError("P1 snapshot requires the authoritative S0 Q32 residents")
    if len(mapping) != 512 or mapping[:32] != tuple(range(32)) or any(
        value != -1 for value in mapping[32:]
    ):
        raise ValueError("P1 snapshot logical-to-physical map is invalid")
    w13_shape = tuple(layer.get("w13_shape", ()))
    w2_shape = tuple(layer.get("w2_shape", ()))
    if w13_shape != (32, 2560, 1280) or w2_shape != (32, 640, 2560):
        raise ValueError("P1 one-layer owner shapes differ from S0 receipt")
    if layer.get("dtype") not in ("bfloat16", "torch.bfloat16"):
        raise ValueError("P1 one-layer owner must use bfloat16")
    if payload.get("layer_count") != 48:
        raise ValueError("P1 snapshot must describe all 48 source layers")
    if (
        receipt.get("schema") != 1
        or receipt.get("status") != "RECORDED_Q32_PHYSICAL_NOT_F7_QUALIFIED"
        or receipt.get("summary", {}).get("layers") != 48
        or receipt.get("summary", {}).get("resident_capacity") != 32
    ):
        raise ValueError("S0 Q32 receipt header is invalid")
    receipt_layers = receipt.get("layers")
    receipt_maps = receipt.get("maps")
    receipt_devices = receipt.get("devices")
    if not all(
        isinstance(value, list) and len(value) == 48
        for value in (receipt_layers, receipt_maps, receipt_devices)
    ):
        raise ValueError("S0 Q32 receipt must contain 48 complete layer records")
    expected_residents = list(range(32))
    expected_mapping = list(mapping)
    for index, (record, map_record, device_record) in enumerate(
        zip(receipt_layers, receipt_maps, receipt_devices)
    ):
        tensors = record.get("tensors", {}) if isinstance(record, dict) else {}
        if (
            record.get("layer") != index
            or record.get("resident_capacity") != 32
            or tensors.get("w13_weight", {}).get("shape") != [32, 2560, 1280]
            or tensors.get("w2_weight", {}).get("shape") != [32, 640, 2560]
            or tensors.get("w13_weight", {}).get("dtype") != "torch.bfloat16"
            or tensors.get("w2_weight", {}).get("dtype") != "torch.bfloat16"
            or map_record.get("layer") != index
            or map_record.get("gpu_index_to_logical") != expected_residents
            or map_record.get("selected_logical_experts") != expected_residents
            or map_record.get("logical_to_gpu_index") != expected_mapping
            or device_record.get("layer") != index
            or device_record.get("w13_weight") != "npu"
            or device_record.get("w2_weight") != "npu"
        ):
            raise ValueError(f"S0 Q32 receipt layer {index} is inconsistent")
    owner_path = layer.get("live_owner_path_template")
    if type(owner_path) is not str or "{layer}" not in owner_path:
        raise ValueError("P1 live owner path template is invalid")
    return P1Snapshot(
        manifest_sha256=_canonical_digest(payload),
        q32_physical_sha256=EXPECTED_Q32_PHYSICAL_SHA256,
        evidence_archive_sha256=EXPECTED_EVIDENCE_ARCHIVE_SHA256,
        layer_count=48,
        resident_logical_ids=residents,
        logical_to_physical=mapping,
        w13_shape=w13_shape,
        w2_shape=w2_shape,
        dtype="bfloat16",
        owner_path_template=owner_path,
        q32_receipt_path=receipt_path,
    )


def load_source_preflight_token(path: str | Path) -> SourcePreflightToken:
    payload = json.loads(Path(path).read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "PASS_EXACT_SOURCE_AND_API"
        or payload.get("passed") is not True
    ):
        raise ValueError("P1 requires a passing exact-source preflight")
    observed_value = payload.get("observed_sha256")
    checks_value = payload.get("api_checks")
    if not isinstance(observed_value, dict) or not isinstance(checks_value, list):
        raise ValueError("source preflight receipt is malformed")
    expected_observed = tuple(
        (item.name, item.sha256) for item in DEFAULT_SOURCE_EXPECTATIONS
    ) + (("model_config", DEFAULT_MODEL_CONFIG_EXPECTATION.sha256),)
    expected_checks = tuple(
        item.name for item in DEFAULT_SOURCE_EXPECTATIONS
    ) + ("model_config",)
    observed = tuple((name, observed_value.get(name)) for name, _ in expected_observed)
    checks = tuple(checks_value)
    result = PreflightResult(
        status=payload["status"],
        passed=True,
        observed_sha256=observed,
        api_checks=checks,
    )
    token = result.verification_token()
    if observed != expected_observed or checks != expected_checks:
        raise ValueError("source preflight receipt differs from the frozen identity")
    if not validate_exact_source_preflight_token(token):
        raise ValueError("source preflight token binding is invalid")
    return token


def require_device_enable(environment: Mapping[str, str]) -> None:
    if environment.get("ROUND16_B_ENABLE_P1") != "1":
        raise RuntimeError("ROUND16_B_ENABLE_P1=1 is required for device execution")


@dataclass(frozen=True)
class P1FactoryContext:
    source_token: SourcePreflightToken
    snapshot: P1Snapshot
    layer: int
    tokens: int


@dataclass(frozen=True)
class P1RuntimeBinding:
    layer_binding: AllGatherQ32LayerBinding
    tensor_ops: TorchNpuTensorOps
    hidden_states: Any
    router_logits: Any
    selector_once: Callable[..., DeviceSelectedTopK]
    kt_submit: Callable[..., Any]
    kt_sync: Callable[..., KTDeviceWeightedPartial]
    kt_abort_drain: Callable[..., None]

    def __post_init__(self) -> None:
        if not isinstance(self.layer_binding, AllGatherQ32LayerBinding):
            raise TypeError("runtime layer binding is invalid")
        if not isinstance(self.tensor_ops, TorchNpuTensorOps):
            raise TypeError("runtime tensor ops are invalid")
        _require_callable(self.selector_once, P1_SELECTOR_API, "selector-once")
        _require_callable(self.kt_submit, P1_KT_SUBMIT_API, "KT submit")
        _require_callable(self.kt_sync, P1_KT_SYNC_API, "KT sync")
        _require_callable(self.kt_abort_drain, P1_KT_ABORT_API, "KT abort/drain")


def validate_live_owner(
    layer_binding: AllGatherQ32LayerBinding,
    snapshot: P1Snapshot,
    layer: int,
) -> None:
    """Validate the already-loaded one-layer owner before any probe side effect."""

    if not isinstance(layer_binding, AllGatherQ32LayerBinding):
        raise TypeError("live owner gate requires AllGatherQ32LayerBinding")
    if not isinstance(snapshot, P1Snapshot):
        raise TypeError("live owner gate requires P1Snapshot")
    if type(layer) is not int or not (0 <= layer < snapshot.layer_count):
        raise ValueError("live owner layer index is invalid")
    expected_suffix = f"layers.{layer}.mlp.experts"
    if not layer_binding.layer_name.endswith(expected_suffix):
        raise ValueError("live owner path differs from the requested layer")
    plan = layer_binding.plan
    if (
        plan.resident_logical_ids != snapshot.resident_logical_ids
        or plan.logical_to_physical != snapshot.logical_to_physical
    ):
        raise ValueError("live owner Q32 plan differs from the S0 snapshot")
    owner = layer_binding.routed_owner
    if getattr(owner, "round16_selective_plan", None) is not plan:
        raise ValueError("live owner does not retain the identical Q32 plan")
    w13 = getattr(owner, "w13_weight", None)
    w2 = getattr(owner, "w2_weight", None)
    if w13 is w2:
        raise ValueError("live owner w13/w2 allocations must be distinct")
    w13_meta = _tensor_metadata(w13, "live owner w13_weight")
    w2_meta = _tensor_metadata(w2, "live owner w2_weight")
    _require_bfloat16(w13_meta, "live owner w13_weight")
    _require_bfloat16(w2_meta, "live owner w2_weight")
    if w13_meta.shape != snapshot.w13_shape:
        raise ValueError("live owner w13 shape differs from the S0 receipt")
    if w2_meta.shape != snapshot.w2_shape:
        raise ValueError("live owner w2 shape differs from the S0 receipt")
    if w13_meta.device != w2_meta.device:
        raise ValueError("live owner weights must share one NPU")
    manager = getattr(owner, "expert_map_manager", None)
    if (
        type(getattr(manager, "global_num_experts", None)) is not int
        or manager.global_num_experts != 512
        or type(getattr(manager, "local_num_experts", None)) is not int
        or manager.local_num_experts != 32
    ):
        raise ValueError("live owner manager dimensions must be logical512/Q32")
    mapper = getattr(manager, "map_global_to_local", None)
    if not callable(mapper):
        raise TypeError("live owner manager mapper is unavailable")
    for logical_id, expected in enumerate(plan.logical_to_physical):
        observed = mapper(logical_id)
        if type(observed) is not int or observed != expected:
            raise ValueError("live owner manager mapping differs from the Q32 plan")
    if layer_binding.comm_method_provider() is not layer_binding.comm_method:
        raise ValueError("live current comm identity differs from the layer binding")
    if layer_binding.stream_identity_provider() is None:
        raise ValueError("live current stream identity is unavailable")


def _load_factory(specification: str) -> Callable[[P1FactoryContext], P1RuntimeBinding]:
    if type(specification) is not str or specification.count(":") != 1:
        raise ValueError("binding factory must use module:callable syntax")
    module_name, attribute = specification.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("binding factory must use module:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    _require_callable(factory, P1_RUNTIME_FACTORY_API, "runtime binding factory")
    return factory


def run_device_probe(
    binding: P1RuntimeBinding,
    context: P1FactoryContext,
) -> dict[str, Any]:
    if binding.layer_binding.source_preflight_token != context.source_token:
        raise ValueError("runtime binding did not consume the exact source token")
    validate_live_owner(binding.layer_binding, context.snapshot, context.layer)
    plan = binding.layer_binding.plan
    if (
        plan.resident_logical_ids != context.snapshot.resident_logical_ids
        or plan.logical_to_physical != context.snapshot.logical_to_physical
    ):
        raise ValueError("runtime Q32 plan differs from the S0 snapshot")
    payload_builder = DevicePhysicalTopKPayloadBuilder(plan, binding.tensor_ops)
    adapter = AllGatherQ32DeviceTensorAdapter(
        AllGatherQ32TensorBinding(binding.layer_binding, payload_builder)
    )
    prepared = adapter.prepare(
        binding.hidden_states,
        binding.router_logits,
        binding.layer_binding.layer_name,
    )
    try:
        selected = binding.selector_once(
            hidden_states=prepared.selector_hidden_states,
            router_logits=prepared.selector_router_logits,
            valid_rows=prepared.valid_rows,
            layer_name=binding.layer_binding.layer_name,
        )
        if not isinstance(selected, DeviceSelectedTopK):
            raise TypeError("selector must return DeviceSelectedTopK")
    except BaseException:
        prepared.cleanup(prepared)
        raise
    stream_identity = binding.layer_binding.stream_identity_provider()
    identity_key = ("p1", context.layer, context.tokens)
    try:
        pending = binding.kt_submit(
            hidden_states=prepared.selector_hidden_states,
            topk_ids=selected.topk_ids,
            topk_weights=selected.topk_weights,
            stream_identity=stream_identity,
            identity_key=identity_key,
        )
    except BaseException:
        prepared.cleanup(prepared)
        raise
    native: DeviceWeightedPartial | None = None
    pending_active = True
    try:
        native = adapter.execute(
            prepared,
            selected,
            prepared.selector_hidden_states,
            identity_key,
        )
        kt = binding.kt_sync(
            pending=pending,
            reference_tensor=native.output,
            stream_identity=stream_identity,
            identity_key=identity_key,
        )
        pending_active = False
        merged = TensorWeightedAggregateMerger(binding.tensor_ops).merge(native, kt)
    except BaseException as primary:
        if pending_active:
            try:
                binding.kt_abort_drain(
                    pending=pending,
                    reference_tensor=None if native is None else native.output,
                    stream_identity=stream_identity,
                    identity_key=identity_key,
                )
            except BaseException as cleanup_error:
                primary.add_note(f"KT abort/drain also failed: {cleanup_error!r}")
        raise
    output_meta = _tensor_metadata(merged.output, "merged P1 output")
    return {
        "status": "PASS_P1_ONE_LAYER_DEVICE_CONTRACT",
        "device_executed": True,
        "layer": context.layer,
        "tokens": context.tokens,
        "output_shape": list(output_meta.shape),
        "output_dtype": output_meta.dtype,
        "output_device": output_meta.device,
        "route_weights_applied": True,
        "routed_scaling_applied": False,
        "source_checks": 18,
        "snapshot_sha256": context.snapshot.manifest_sha256,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--enable-device", action="store_true")
    parser.add_argument("--source-token", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binding-factory")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--physical-experts", type=int, default=32)
    parser.add_argument("--logical-experts", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quant-type", default="NONE")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    exact = (
        parsed.tp_size == 1
        and parsed.ep_size == 1
        and parsed.physical_experts == 32
        and parsed.logical_experts == 512
        and parsed.top_k == 10
        and parsed.max_model_len >= 256
        and parsed.dtype == "bfloat16"
        and parsed.quant_type == "NONE"
    )
    if not exact:
        raise ValueError("P1 is restricted to TP1/EP1 BF16 physical-Q32 Top-10")
    if type(parsed.layer) is not int or not (0 <= parsed.layer < 48):
        raise ValueError("P1 layer must be an exact index in [0,47]")
    if type(parsed.tokens) is not int or not (1 <= parsed.tokens <= 128):
        raise ValueError("P1 token count must be in [1,128]")
    source_token = load_source_preflight_token(parsed.source_token)
    snapshot = load_snapshot_manifest(parsed.snapshot_manifest)
    context = P1FactoryContext(source_token, snapshot, parsed.layer, parsed.tokens)
    if parsed.validate_only:
        _write_json(
            parsed.output,
            {
                "status": "READY_FOR_DEVICE_RUNTIME_BINDING",
                "device_executed": False,
                "live_eligible": False,
                "source_checks": 18,
                "snapshot_sha256": snapshot.manifest_sha256,
                "required_runtime_factory_api": P1_RUNTIME_FACTORY_API,
                "one_layer_weight_source": snapshot.owner_path_template.format(
                    layer=parsed.layer
                ),
            },
        )
        return 0
    require_device_enable(os.environ)
    if not parsed.enforce_eager:
        raise ValueError("P1 device execution requires --enforce-eager")
    if not parsed.binding_factory:
        raise ValueError("P1 device execution requires --binding-factory")
    factory = _load_factory(parsed.binding_factory)
    binding = factory(context)
    if not isinstance(binding, P1RuntimeBinding):
        raise TypeError("runtime binding factory returned the wrong type")
    _write_json(parsed.output, run_device_probe(binding, context))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CombinedDeviceWeightedPartial",
    "DevicePhysicalTopKPayloadBuilder",
    "KTDeviceWeightedPartial",
    "P1FactoryContext",
    "P1RuntimeBinding",
    "P1Snapshot",
    "P1_PROBE_API",
    "P1_RUNTIME_FACTORY_API",
    "P1_SELECTOR_API",
    "P1_KT_ABORT_API",
    "P1_KT_SUBMIT_API",
    "P1_KT_SYNC_API",
    "P1_TENSOR_MERGE_API",
    "P1_TENSOR_OPS_API",
    "TensorWeightedAggregateMerger",
    "TorchNpuTensorOps",
    "load_snapshot_manifest",
    "load_source_preflight_token",
    "main",
    "require_device_enable",
    "run_device_probe",
    "validate_live_owner",
)
