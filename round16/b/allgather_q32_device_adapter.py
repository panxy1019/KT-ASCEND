"""Default-OFF source-bound TP1/EP1 ALLGATHER physical-Q32 adapter.

The module imports no torch, vLLM, or vLLM-Ascend runtime.  A prepared worker
injects the exact donor objects only after the frozen source preflight passes.
This adapter remains live-ineligible until the same sequence passes a bounded
device test in the frozen image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from threading import Lock
from typing import Any, Callable

from .native_source_preflight import (
    SourcePreflightToken,
    validate_exact_source_preflight_token,
)
from .routing_bridge import HybridRoutePlan
from .selective_loader import SelectiveLoadPlan
from .source_bound_hybrid_runner import (
    NativeMoeCommPrepared,
    NativeWeightedPartial,
    SOURCE_BOUND_MOE_COMM_CLEANUP_API,
    SOURCE_BOUND_MOE_COMM_CONTEXT_API,
    SOURCE_BOUND_MOE_COMM_PREPARE_API,
    SOURCE_BOUND_NATIVE_WEIGHTED_PARTIAL_API,
)


ALLGATHER_BUILD_FUSED_INPUT_API = (
    "vllm-ascend@550ab66:build_fused_experts_input:exact-source-v1"
)
ALLGATHER_STREAM_IDENTITY_API = (
    "round16-b:allgather-current-stream-identity:v1"
)
ALLGATHER_Q32_DEVICE_ADAPTER_API = (
    "round16-b:tp1-ep1-allgather-bf16-physical-q32:device-tensor-v1"
)
ALLGATHER_Q32_OFFLINE_FAKE_API = (
    "round16-b:tp1-ep1-allgather-bf16-physical-q32:offline-fake-v1"
)
ALLGATHER_PHYSICAL_TENSOR_PAYLOAD_API = (
    "round16-b:allgather-device-physical-topk-payload:v1"
)
ALLGATHER_DEVICE_WEIGHTED_PARTIAL_API = (
    "round16-b:allgather-device-weighted-aggregate-partial:v1"
)


def _require_callback(value: Any, identity: str, name: str) -> None:
    if not callable(value):
        raise TypeError(f"{name} must be callable")
    if getattr(value, "round16_api_identity", None) != identity:
        raise ValueError(f"{name} API identity is not verified")


def _optional_finite_float(value: Any, default: float, name: str) -> float:
    """Normalize optional Ascend owner scalars without changing their default.

    The frozen BF16 owner exposes the optional SwiGLU tuning fields as
    ``None`` when no override was configured.  The Ascend fused-input schema
    still requires concrete floats, so the adapter supplies the schema's
    existing defaults.  Non-finite configured values continue to fail closed.
    """

    result = default if value is None else float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class AllGatherRuntimeContract:
    tp_size: int = 1
    ep_size: int = 1
    eager: bool = True
    comm_type: str = "ALLGATHER"
    dtype: str = "bfloat16"
    quant_type: str = "NONE"
    expert_map: None = None
    log2phy: None = None

    def __post_init__(self) -> None:
        if type(self.tp_size) is not int or self.tp_size != 1:
            raise ValueError("first ALLGATHER adapter requires TP1")
        if type(self.ep_size) is not int or self.ep_size != 1:
            raise ValueError("first ALLGATHER adapter requires EP1")
        if self.eager is not True:
            raise ValueError("first ALLGATHER adapter requires eager execution")
        if self.comm_type != "ALLGATHER":
            raise ValueError("first adapter supports only ALLGATHER")
        if self.dtype not in ("bfloat16", "bf16", "torch.bfloat16"):
            raise ValueError("first ALLGATHER adapter requires bfloat16")
        if self.quant_type != "NONE":
            raise ValueError("first ALLGATHER adapter requires QuantType.NONE")
        if self.expert_map is not None or self.log2phy is not None:
            raise ValueError("ALLGATHER physical-Q32 requires no expert map or log2phy")


def _validate_weight(owner: Any, name: str, physical_count: int) -> None:
    weight = getattr(owner, name, None)
    try:
        shape = tuple(weight.shape)
    except (AttributeError, TypeError) as error:
        raise ValueError(f"{name} shape is unavailable") from error
    if (
        not shape
        or any(type(value) is not int for value in shape)
        or shape[0] != physical_count
    ):
        raise ValueError(f"{name} must have physical-Q32 leading dimension")
    if str(getattr(weight, "dtype", "")).lower() not in (
        "bfloat16",
        "bf16",
        "torch.bfloat16",
    ):
        raise ValueError(f"{name} must use bfloat16")


@dataclass(frozen=True)
class AllGatherQ32LayerBinding:
    layer_name: str
    plan: SelectiveLoadPlan
    routed_owner: Any
    comm_method: Any
    comm_method_provider: Callable[[], Any]
    stream_identity_provider: Callable[[], Any]
    build_fused_experts_input: Callable[..., Any]
    source_preflight_token: SourcePreflightToken
    runtime: AllGatherRuntimeContract = field(
        default_factory=AllGatherRuntimeContract
    )
    quant_none: Any = "NONE"
    activation: str = "silu"

    def __post_init__(self) -> None:
        if type(self.layer_name) is not str or not self.layer_name:
            raise ValueError("layer_name must be a non-empty string")
        if not isinstance(self.plan, SelectiveLoadPlan):
            raise TypeError("ALLGATHER binding requires SelectiveLoadPlan")
        if self.plan.logical_expert_count != 512 or self.plan.physical_expert_count != 32:
            raise ValueError("ALLGATHER binding requires logical512/physical32")
        if getattr(self.routed_owner, "round16_selective_plan", None) is not self.plan:
            raise ValueError("routed owner must retain the identical Q32 plan")
        if type(self.comm_method).__name__ != "AllGatherCommImpl":
            raise ValueError("exact AllGatherCommImpl object is required")
        for name in ("prepare", "fused_experts", "finalize"):
            if not callable(getattr(self.comm_method, name, None)):
                raise TypeError(f"AllGatherCommImpl lacks {name}")
        _require_callback(
            self.comm_method_provider,
            SOURCE_BOUND_MOE_COMM_CONTEXT_API,
            "current comm provider",
        )
        _require_callback(
            self.stream_identity_provider,
            ALLGATHER_STREAM_IDENTITY_API,
            "current stream provider",
        )
        _require_callback(
            self.build_fused_experts_input,
            ALLGATHER_BUILD_FUSED_INPUT_API,
            "fused input builder",
        )
        if self.comm_method_provider() is not self.comm_method:
            raise ValueError("current comm object identity differs from binding")
        config = getattr(self.comm_method, "moe_config", None)
        if (
            type(getattr(config, "ep_size", None)) is not int
            or config.ep_size != 1
            or type(getattr(config, "tp_size", None)) is not int
            or config.tp_size != 1
        ):
            raise ValueError("AllGatherCommImpl runtime config must be TP1/EP1")
        if not isinstance(self.runtime, AllGatherRuntimeContract):
            raise TypeError("runtime contract is invalid")
        if not validate_exact_source_preflight_token(self.source_preflight_token):
            raise ValueError("exact source preflight token is required")
        _validate_weight(self.routed_owner, "w13_weight", 32)
        _validate_weight(self.routed_owner, "w2_weight", 32)
        if self.routed_owner.w13_weight is self.routed_owner.w2_weight:
            raise ValueError("w13/w2 physical allocations must be distinct")
        if type(self.activation) is not str or not self.activation:
            raise ValueError("activation must be a non-empty string")


@dataclass(frozen=True)
class _PreparedState:
    owner: Any
    donor_prepare_output: Any
    stream_identity: Any
    sequence: int


class AllGatherQ32OfflineFakeAdapter:
    """Tuple-only fake used to test source order and routing ownership offline."""

    api_identity = ALLGATHER_Q32_OFFLINE_FAKE_API
    live_eligible = False

    def __init__(self, binding: AllGatherQ32LayerBinding) -> None:
        if not isinstance(binding, AllGatherQ32LayerBinding):
            raise TypeError("binding must be AllGatherQ32LayerBinding")
        self.binding = binding
        self._lock = Lock()
        self._active: NativeMoeCommPrepared | None = None
        self._sequence = 0

    def prepare(
        self,
        hidden_states: Any,
        router_logits: Any,
        layer_name: str,
    ) -> NativeMoeCommPrepared:
        if layer_name != self.binding.layer_name:
            raise ValueError("prepared layer identity differs from binding")
        if self.binding.comm_method_provider() is not self.binding.comm_method:
            raise RuntimeError("current comm object identity changed before prepare")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("ALLGATHER adapter already has an active prepare")
        stream_identity = self.binding.stream_identity_provider()
        if stream_identity is None:
            raise RuntimeError("current stream identity is unavailable")
        prepare_output = self.binding.comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=False,
            enable_shared_expert_dp=False,
            quant_type=self.binding.quant_none,
        )
        for name in (
            "hidden_states",
            "router_logits",
            "mc2_mask",
            "padded_hidden_states_shape",
            "pertoken_scale",
        ):
            if not hasattr(prepare_output, name):
                raise TypeError(f"donor prepare output lacks {name}")
        if prepare_output.mc2_mask is not None:
            raise ValueError("EP1 ALLGATHER must not produce an MC2 mask")
        if prepare_output.pertoken_scale is not None:
            raise ValueError("BF16 QuantType.NONE must not produce per-token scale")
        try:
            input_rows = len(hidden_states)
            prepared_rows = len(prepare_output.hidden_states)
            router_rows = len(prepare_output.router_logits)
        except TypeError as error:
            raise ValueError("prepared hidden/router inputs must expose row count") from error
        if prepared_rows < input_rows or router_rows != prepared_rows:
            raise ValueError("prepared hidden/router row layout is inconsistent")
        valid_rows = (True,) * input_rows + (False,) * (prepared_rows - input_rows)
        with self._lock:
            if self._active is not None:
                raise RuntimeError("ALLGATHER adapter prepare raced with another request")
            self._sequence += 1
            state = _PreparedState(
                self,
                prepare_output,
                stream_identity,
                self._sequence,
            )
            prepared = NativeMoeCommPrepared(
                comm_method=self.binding.comm_method,
                selector_hidden_states=prepare_output.hidden_states,
                selector_router_logits=prepare_output.router_logits,
                valid_rows=valid_rows,
                opaque_state=state,
                comm_type="ALLGATHER",
                ep_size=1,
                tp_size=1,
                quant_type="NONE",
                expert_map=None,
                log2phy=None,
                physical_expert_count=32,
                cleanup=self.cleanup,
            )
            self._active = prepared
        return prepared

    def _require_active(self, prepared: NativeMoeCommPrepared) -> _PreparedState:
        if not isinstance(prepared, NativeMoeCommPrepared):
            raise TypeError("execute requires NativeMoeCommPrepared")
        state = prepared.opaque_state
        if not isinstance(state, _PreparedState) or state.owner is not self:
            raise RuntimeError("prepared state belongs to a different adapter")
        with self._lock:
            if self._active is not prepared:
                raise RuntimeError("prepared state is stale or already consumed")
        return state

    def _retire(self, prepared: NativeMoeCommPrepared) -> None:
        with self._lock:
            if self._active is prepared:
                self._active = None

    def cleanup(self, prepared: NativeMoeCommPrepared) -> None:
        state = prepared.opaque_state
        if not isinstance(state, _PreparedState) or state.owner is not self:
            raise RuntimeError("cleanup prepared state belongs to another adapter")
        self._retire(prepared)

    def execute(
        self,
        prepared: NativeMoeCommPrepared,
        route_plan: HybridRoutePlan,
        hidden_states: Any,
        identity_key: tuple[Any, ...],
    ) -> NativeWeightedPartial:
        state = self._require_active(prepared)
        try:
            if not isinstance(route_plan, HybridRoutePlan):
                raise TypeError("execute requires HybridRoutePlan")
            if hidden_states is not prepared.selector_hidden_states:
                raise RuntimeError("native branch did not receive prepared hidden states")
            if type(identity_key) is not tuple or not identity_key:
                raise ValueError("native execution identity must be a non-empty tuple")
            if self.binding.comm_method_provider() is not prepared.comm_method:
                raise RuntimeError("comm object identity changed before fused execution")
            if self.binding.stream_identity_provider() is not state.stream_identity:
                raise RuntimeError("stream identity changed before fused execution")
            if route_plan.routes.token_count != len(prepared.valid_rows):
                raise ValueError("route rows differ from prepared valid-row layout")

            topk_ids = [
                [0] * len(row) for row in route_plan.routes.topk_ids
            ]
            topk_weights = [
                [0.0] * len(row) for row in route_plan.routes.topk_weights
            ]
            for route in route_plan.npu_routes:
                if route.npu_slot is None or not (0 <= route.npu_slot < 32):
                    raise ValueError("resident route has invalid physical Q32 slot")
                topk_ids[route.token_index][route.topk_index] = route.npu_slot
                topk_weights[route.token_index][route.topk_index] = route.weight
            physical_ids = tuple(tuple(row) for row in topk_ids)
            physical_weights = tuple(tuple(row) for row in topk_weights)

            owner = self.binding.routed_owner
            prepare_output = state.donor_prepare_output
            fused_input = self.binding.build_fused_experts_input(
                hidden_states=hidden_states,
                topk_weights=physical_weights,
                topk_ids=physical_ids,
                w1=owner.w13_weight,
                w2=owner.w2_weight,
                w1_bias=getattr(owner, "w13_bias", None),
                w2_bias=getattr(owner, "w2_bias", None),
                quant_type=self.binding.quant_none,
                dynamic_eplb=False,
                expert_map=None,
                global_redundant_expert_num=0,
                mc2_mask=None,
                apply_router_weight_on_input=False,
                log2phy=None,
                pertoken_scale=None,
                activation=self.binding.activation,
                w1_scale=None,
                w2_scale=None,
                w1_scale_bias=None,
                w2_scale_bias=None,
                swiglu_limit=_optional_finite_float(
                    getattr(owner, "swiglu_limit", None), 0.0, "swiglu_limit"
                ),
                swiglu_alpha=_optional_finite_float(
                    getattr(owner, "swiglu_alpha", None), 1.0, "swiglu_alpha"
                ),
                swiglu_beta=_optional_finite_float(
                    getattr(owner, "swiglu_beta", None), 0.0, "swiglu_beta"
                ),
                lora_context=getattr(owner, "_ascend_moe_lora_context", None),
            )
            fused_result = self.binding.comm_method.fused_experts(
                fused_experts_input=fused_input
            )
            if not hasattr(fused_result, "routed_out"):
                raise TypeError("fused_experts result lacks routed_out")
            finalized = self.binding.comm_method.finalize(
                hidden_states=fused_result.routed_out,
                reduce_results=True,
                padded_hidden_states_shape=(
                    prepare_output.padded_hidden_states_shape
                ),
            )
            try:
                rows = tuple(
                    tuple(float(value) for value in row) for row in finalized
                )
            except (TypeError, ValueError) as error:
                raise ValueError("finalized ALLGATHER output must be numeric [T,H]") from error
            if len(rows) != route_plan.routes.token_count:
                raise ValueError("finalized ALLGATHER row count differs from routes")
            if any(not isfinite(value) for row in rows for value in row):
                raise ValueError("finalized ALLGATHER output must be finite")
            return NativeWeightedPartial(rows)
        finally:
            self._retire(prepared)


AllGatherQ32OfflineFakeAdapter.prepare.round16_api_identity = (
    SOURCE_BOUND_MOE_COMM_PREPARE_API
)
AllGatherQ32OfflineFakeAdapter.execute.round16_api_identity = (
    SOURCE_BOUND_NATIVE_WEIGHTED_PARTIAL_API
)
AllGatherQ32OfflineFakeAdapter.cleanup.round16_api_identity = (
    SOURCE_BOUND_MOE_COMM_CLEANUP_API
)


@dataclass(frozen=True)
class DeviceSelectedTopK:
    """Opaque device Top-K tensors selected exactly once upstream."""

    topk_ids: Any
    topk_weights: Any
    valid_rows: tuple[bool, ...]

    def __post_init__(self) -> None:
        try:
            valid_rows = tuple(self.valid_rows)
        except TypeError as error:
            raise ValueError("valid_rows must be a sequence") from error
        if any(type(value) is not bool for value in valid_rows):
            raise ValueError("valid_rows entries must be exact booleans")
        object.__setattr__(self, "valid_rows", valid_rows)


@dataclass(frozen=True)
class DevicePhysicalTopKPayload:
    """Device-side physical-Q32 payload produced by a verified callback.

    The markers are part of the callback contract because this torch-free
    module must not inspect tensor values or copy them to the host.
    """

    topk_ids: Any
    topk_weights: Any
    resident_ids_are_physical: bool = True
    cpu_routes_zero_weight: bool = True
    padding_zero_weight: bool = True
    safe_placeholder_id: int = 0

    def __post_init__(self) -> None:
        if self.resident_ids_are_physical is not True:
            raise ValueError("resident routes must carry physical Q32 IDs")
        if self.cpu_routes_zero_weight is not True:
            raise ValueError("CPU routes must have zero native route weight")
        if self.padding_zero_weight is not True:
            raise ValueError("padding routes must have zero native route weight")
        if type(self.safe_placeholder_id) is not int or not (
            0 <= self.safe_placeholder_id < 32
        ):
            raise ValueError("safe placeholder must be a physical Q32 ID")


@dataclass(frozen=True)
class DeviceWeightedPartial:
    """Opaque `[T,H]` device tensor with route weights already applied."""

    output: Any
    route_weights_applied: bool = True
    routed_scaling_applied: bool = False
    prepare_applied: bool = True
    finalize_applied: bool = True

    def __post_init__(self) -> None:
        if self.route_weights_applied is not True:
            raise ValueError("native aggregate must apply Top-K route weights")
        if self.routed_scaling_applied is not False:
            raise ValueError("routed scaling remains owned by upstream MoERunner")
        if self.prepare_applied is not True or self.finalize_applied is not True:
            raise ValueError("native aggregate must retain prepare/finalize envelope")


@dataclass(frozen=True)
class AllGatherQ32TensorBinding:
    layer: AllGatherQ32LayerBinding
    build_physical_topk_payload: Callable[..., DevicePhysicalTopKPayload]

    def __post_init__(self) -> None:
        if not isinstance(self.layer, AllGatherQ32LayerBinding):
            raise TypeError("layer must be AllGatherQ32LayerBinding")
        _require_callback(
            self.build_physical_topk_payload,
            ALLGATHER_PHYSICAL_TENSOR_PAYLOAD_API,
            "device physical Top-K payload builder",
        )


@dataclass(frozen=True)
class _TensorMetadata:
    shape: tuple[int, ...]
    dtype: str
    device: str


def _tensor_metadata(value: Any, name: str) -> _TensorMetadata:
    """Read tensor metadata without indexing, iteration, scalar extraction, or sync."""

    try:
        shape = tuple(value.shape)
    except (AttributeError, TypeError) as error:
        raise TypeError(f"{name} must expose tensor shape metadata") from error
    if not shape or any(type(dimension) is not int or dimension < 0 for dimension in shape):
        raise ValueError(f"{name} shape must contain exact non-negative integers")
    dtype = str(getattr(value, "dtype", "")).lower()
    if not dtype:
        raise TypeError(f"{name} must expose tensor dtype metadata")
    device_value = getattr(value, "device", None)
    device_type = str(getattr(device_value, "type", "")).lower()
    device = str(device_value).lower()
    if device_type != "npu" or not device.startswith("npu"):
        raise ValueError(f"{name} must remain on an NPU device")
    return _TensorMetadata(shape, dtype, device)


def _require_dtype(metadata: _TensorMetadata, accepted: tuple[str, ...], name: str) -> None:
    if metadata.dtype not in accepted:
        required = (
            "float32 or bfloat16"
            if "float32" in accepted and "bfloat16" in accepted
            else ("bfloat16" if "bfloat16" in accepted else accepted[0])
        )
        raise ValueError(f"{name} must use {required}")


@dataclass(frozen=True)
class _TensorPreparedState:
    owner: Any
    donor_prepare_output: Any
    stream_identity: Any
    sequence: int
    input_rows: int
    prepared_rows: int
    hidden_width: int
    device: str


class AllGatherQ32DeviceTensorAdapter:
    """Source-bound live tensor contract; device eligibility remains OFF.

    This adapter deliberately does not satisfy the tuple-based
    ``SOURCE_BOUND_NATIVE_WEIGHTED_PARTIAL_API``.  A device runner must bind a
    tensor-preserving KT/native merge before it may consume this result.
    """

    api_identity = ALLGATHER_Q32_DEVICE_ADAPTER_API
    live_eligible = False
    device_probe_eligible = False

    def __init__(self, binding: AllGatherQ32TensorBinding) -> None:
        if not isinstance(binding, AllGatherQ32TensorBinding):
            raise TypeError("binding must be AllGatherQ32TensorBinding")
        self.binding = binding
        self._lock = Lock()
        self._active: NativeMoeCommPrepared | None = None
        self._sequence = 0

    @property
    def layer(self) -> AllGatherQ32LayerBinding:
        return self.binding.layer

    def prepare(
        self,
        hidden_states: Any,
        router_logits: Any,
        layer_name: str,
    ) -> NativeMoeCommPrepared:
        if layer_name != self.layer.layer_name:
            raise ValueError("prepared layer identity differs from binding")
        hidden_meta = _tensor_metadata(hidden_states, "hidden_states")
        router_meta = _tensor_metadata(router_logits, "router_logits")
        _require_dtype(
            hidden_meta, ("bfloat16", "bf16", "torch.bfloat16"), "hidden_states"
        )
        if len(hidden_meta.shape) != 2:
            raise ValueError("hidden_states must be a rank-2 [T,H] tensor")
        if len(router_meta.shape) != 2 or router_meta.shape[1] != 512:
            raise ValueError("router_logits must be a rank-2 [T,512] tensor")
        if router_meta.shape[0] != hidden_meta.shape[0]:
            raise ValueError("hidden/router input row counts differ")
        if router_meta.device != hidden_meta.device:
            raise ValueError("hidden/router inputs must share one NPU device")
        if self.layer.comm_method_provider() is not self.layer.comm_method:
            raise RuntimeError("current comm object identity changed before prepare")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("ALLGATHER adapter already has an active prepare")
        stream_identity = self.layer.stream_identity_provider()
        if stream_identity is None:
            raise RuntimeError("current stream identity is unavailable")
        prepare_output = self.layer.comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=False,
            enable_shared_expert_dp=False,
            quant_type=self.layer.quant_none,
        )
        for name in (
            "hidden_states",
            "router_logits",
            "mc2_mask",
            "padded_hidden_states_shape",
            "pertoken_scale",
        ):
            if not hasattr(prepare_output, name):
                raise TypeError(f"donor prepare output lacks {name}")
        if prepare_output.mc2_mask is not None:
            raise ValueError("EP1 ALLGATHER must not produce an MC2 mask")
        if prepare_output.pertoken_scale is not None:
            raise ValueError("BF16 QuantType.NONE must not produce per-token scale")
        prepared_hidden_meta = _tensor_metadata(
            prepare_output.hidden_states, "prepared hidden_states"
        )
        prepared_router_meta = _tensor_metadata(
            prepare_output.router_logits, "prepared router_logits"
        )
        _require_dtype(
            prepared_hidden_meta,
            ("bfloat16", "bf16", "torch.bfloat16"),
            "prepared hidden_states",
        )
        if (
            len(prepared_hidden_meta.shape) != 2
            or prepared_hidden_meta.shape[1] != hidden_meta.shape[1]
            or prepared_hidden_meta.shape[0] < hidden_meta.shape[0]
        ):
            raise ValueError("prepared hidden layout is inconsistent")
        if prepared_router_meta.shape != (
            prepared_hidden_meta.shape[0],
            512,
        ):
            raise ValueError("prepared router layout is inconsistent")
        if (
            prepared_hidden_meta.device != hidden_meta.device
            or prepared_router_meta.device != hidden_meta.device
        ):
            raise ValueError("prepared tensors moved to a different NPU device")
        input_rows = hidden_meta.shape[0]
        prepared_rows = prepared_hidden_meta.shape[0]
        valid_rows = (True,) * input_rows + (False,) * (prepared_rows - input_rows)
        with self._lock:
            if self._active is not None:
                raise RuntimeError("ALLGATHER adapter prepare raced with another request")
            self._sequence += 1
            state = _TensorPreparedState(
                self,
                prepare_output,
                stream_identity,
                self._sequence,
                input_rows,
                prepared_rows,
                hidden_meta.shape[1],
                hidden_meta.device,
            )
            prepared = NativeMoeCommPrepared(
                comm_method=self.layer.comm_method,
                selector_hidden_states=prepare_output.hidden_states,
                selector_router_logits=prepare_output.router_logits,
                valid_rows=valid_rows,
                opaque_state=state,
                comm_type="ALLGATHER",
                ep_size=1,
                tp_size=1,
                quant_type="NONE",
                expert_map=None,
                log2phy=None,
                physical_expert_count=32,
                cleanup=self.cleanup,
            )
            self._active = prepared
        return prepared

    def _require_active(self, prepared: NativeMoeCommPrepared) -> _TensorPreparedState:
        if not isinstance(prepared, NativeMoeCommPrepared):
            raise TypeError("execute requires NativeMoeCommPrepared")
        state = prepared.opaque_state
        if not isinstance(state, _TensorPreparedState) or state.owner is not self:
            raise RuntimeError("prepared state belongs to a different adapter")
        with self._lock:
            if self._active is not prepared:
                raise RuntimeError("prepared state is stale or already consumed")
        return state

    def _retire(self, prepared: NativeMoeCommPrepared) -> None:
        with self._lock:
            if self._active is prepared:
                self._active = None

    def cleanup(self, prepared: NativeMoeCommPrepared) -> None:
        state = prepared.opaque_state
        if not isinstance(state, _TensorPreparedState) or state.owner is not self:
            raise RuntimeError("cleanup prepared state belongs to another adapter")
        self._retire(prepared)

    def execute(
        self,
        prepared: NativeMoeCommPrepared,
        selected_topk: DeviceSelectedTopK,
        hidden_states: Any,
        identity_key: tuple[Any, ...],
    ) -> DeviceWeightedPartial:
        state = self._require_active(prepared)
        try:
            if not isinstance(selected_topk, DeviceSelectedTopK):
                raise TypeError("execute requires DeviceSelectedTopK")
            if hidden_states is not prepared.selector_hidden_states:
                raise RuntimeError("native branch did not receive prepared hidden states")
            if type(identity_key) is not tuple or not identity_key:
                raise ValueError("native execution identity must be a non-empty tuple")
            if self.layer.comm_method_provider() is not prepared.comm_method:
                raise RuntimeError("comm object identity changed before fused execution")
            if self.layer.stream_identity_provider() is not state.stream_identity:
                raise RuntimeError("stream identity changed before fused execution")
            if selected_topk.valid_rows != prepared.valid_rows:
                raise ValueError("selected Top-K valid rows differ from prepare")
            ids_meta = _tensor_metadata(selected_topk.topk_ids, "selected topk_ids")
            weights_meta = _tensor_metadata(
                selected_topk.topk_weights, "selected topk_weights"
            )
            if ids_meta.shape != (state.prepared_rows, 10):
                raise ValueError("selected topk_ids must have shape [prepared_T,10]")
            if weights_meta.shape != ids_meta.shape:
                raise ValueError("selected Top-K ID/weight shapes differ")
            _require_dtype(ids_meta, ("int32", "torch.int32"), "selected topk_ids")
            _require_dtype(
                weights_meta,
                (
                    "float32",
                    "torch.float32",
                    "bfloat16",
                    "bf16",
                    "torch.bfloat16",
                ),
                "selected topk_weights",
            )
            if ids_meta.device != state.device or weights_meta.device != state.device:
                raise ValueError("selected Top-K tensors must remain on the prepared NPU")
            physical = self.binding.build_physical_topk_payload(
                topk_ids=selected_topk.topk_ids,
                topk_weights=selected_topk.topk_weights,
                resident_logical_ids=self.layer.plan.resident_logical_ids,
                logical_to_physical=self.layer.plan.logical_to_physical,
                valid_rows=prepared.valid_rows,
                safe_physical_id=0,
                identity_key=identity_key,
            )
            if not isinstance(physical, DevicePhysicalTopKPayload):
                raise TypeError(
                    "device payload builder must return DevicePhysicalTopKPayload"
                )
            physical_ids_meta = _tensor_metadata(
                physical.topk_ids, "physical topk_ids"
            )
            physical_weights_meta = _tensor_metadata(
                physical.topk_weights, "physical topk_weights"
            )
            if physical_ids_meta.shape != ids_meta.shape:
                raise ValueError("physical topk_ids shape differs from selected Top-K")
            if physical_weights_meta.shape != ids_meta.shape:
                raise ValueError("physical topk_weights shape differs from selected Top-K")
            _require_dtype(
                physical_ids_meta, ("int32", "torch.int32"), "physical topk_ids"
            )
            _require_dtype(
                physical_weights_meta,
                (
                    "float32",
                    "torch.float32",
                    "bfloat16",
                    "bf16",
                    "torch.bfloat16",
                ),
                "physical topk_weights",
            )
            if (
                physical_ids_meta.device != state.device
                or physical_weights_meta.device != state.device
            ):
                raise ValueError("physical Top-K payload moved to a different NPU")
            owner = self.layer.routed_owner
            prepare_output = state.donor_prepare_output
            fused_input = self.layer.build_fused_experts_input(
                hidden_states=hidden_states,
                topk_weights=physical.topk_weights,
                topk_ids=physical.topk_ids,
                w1=owner.w13_weight,
                w2=owner.w2_weight,
                w1_bias=getattr(owner, "w13_bias", None),
                w2_bias=getattr(owner, "w2_bias", None),
                quant_type=self.layer.quant_none,
                dynamic_eplb=False,
                expert_map=None,
                global_redundant_expert_num=0,
                mc2_mask=None,
                apply_router_weight_on_input=False,
                log2phy=None,
                pertoken_scale=None,
                activation=self.layer.activation,
                w1_scale=None,
                w2_scale=None,
                w1_scale_bias=None,
                w2_scale_bias=None,
                swiglu_limit=_optional_finite_float(
                    getattr(owner, "swiglu_limit", None), 0.0, "swiglu_limit"
                ),
                swiglu_alpha=_optional_finite_float(
                    getattr(owner, "swiglu_alpha", None), 1.0, "swiglu_alpha"
                ),
                swiglu_beta=_optional_finite_float(
                    getattr(owner, "swiglu_beta", None), 0.0, "swiglu_beta"
                ),
                lora_context=getattr(owner, "_ascend_moe_lora_context", None),
            )
            fused_result = self.layer.comm_method.fused_experts(
                fused_experts_input=fused_input
            )
            if not hasattr(fused_result, "routed_out"):
                raise TypeError("fused_experts result lacks routed_out")
            routed_meta = _tensor_metadata(
                fused_result.routed_out, "fused routed_out"
            )
            if routed_meta.shape != (state.prepared_rows, state.hidden_width):
                raise ValueError("fused routed_out must have shape [prepared_T,H]")
            _require_dtype(
                routed_meta,
                ("bfloat16", "bf16", "torch.bfloat16"),
                "fused routed_out",
            )
            if routed_meta.device != state.device:
                raise ValueError("fused routed_out moved to a different NPU")
            finalized = self.layer.comm_method.finalize(
                hidden_states=fused_result.routed_out,
                reduce_results=True,
                padded_hidden_states_shape=prepare_output.padded_hidden_states_shape,
            )
            finalized_meta = _tensor_metadata(finalized, "finalized routed_out")
            if finalized_meta.shape != (state.input_rows, state.hidden_width):
                raise ValueError("finalized routed_out must have shape [input_T,H]")
            _require_dtype(
                finalized_meta,
                ("bfloat16", "bf16", "torch.bfloat16"),
                "finalized routed_out",
            )
            if finalized_meta.device != state.device:
                raise ValueError("finalized routed_out moved to a different NPU")
            return DeviceWeightedPartial(finalized)
        finally:
            self._retire(prepared)


AllGatherQ32DeviceTensorAdapter.prepare.round16_api_identity = (
    SOURCE_BOUND_MOE_COMM_PREPARE_API
)
AllGatherQ32DeviceTensorAdapter.execute.round16_api_identity = (
    ALLGATHER_DEVICE_WEIGHTED_PARTIAL_API
)
AllGatherQ32DeviceTensorAdapter.cleanup.round16_api_identity = (
    SOURCE_BOUND_MOE_COMM_CLEANUP_API
)


__all__ = (
    "ALLGATHER_BUILD_FUSED_INPUT_API",
    "ALLGATHER_DEVICE_WEIGHTED_PARTIAL_API",
    "ALLGATHER_PHYSICAL_TENSOR_PAYLOAD_API",
    "ALLGATHER_Q32_DEVICE_ADAPTER_API",
    "ALLGATHER_Q32_OFFLINE_FAKE_API",
    "ALLGATHER_STREAM_IDENTITY_API",
    "AllGatherQ32DeviceTensorAdapter",
    "AllGatherQ32LayerBinding",
    "AllGatherQ32OfflineFakeAdapter",
    "AllGatherQ32TensorBinding",
    "AllGatherRuntimeContract",
    "DevicePhysicalTopKPayload",
    "DeviceSelectedTopK",
    "DeviceWeightedPartial",
)
