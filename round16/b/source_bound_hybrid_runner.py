"""CPU-only contract for a source-bound Q32/KT Hybrid MoE runner.

This module models the reviewed runner seam without importing torch, vLLM, or
vLLM-Ascend.  The binding remains live-ineligible until the donor image
provides a verified native operation that consumes already-selected Top-K
routes and the KT device callback/ABI is frozen.

The offline ``run_decode`` path owns one route/split/merge transaction.  It is
used to prove identity, cancellation, and arithmetic ownership only; it is not
presented as the donor ``MoERunner.forward`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from threading import Lock
from typing import Any, Callable, Mapping, Sequence

from round16.common.contracts import RouteBatch

from .kt_expert_backend import (
    KTPendingWeightedPartial,
    KTExpertBackend,
    KTWeightedPartial,
    KTWeightedPartialBackend,
)
from .lifecycle_bridge import RequestLease, RequestLeaseManager, TaskIdentity
from .routing_bridge import HybridRoutePlan, merge_route_outputs, plan_routes
from .selective_loader import SelectiveLoadPlan


SOURCE_BOUND_HYBRID_RUNNER_API = (
    "round16-b:source-bound-hybrid-runner:offline-contract-v1"
)
SOURCE_BOUND_ROUTE_ONCE_API = "round16-b:route-once:preselected-topk-v1"
SOURCE_BOUND_NATIVE_PRESELECTED_API = (
    "round16-b:native-preselected-topk-execute:device-abi-v1"
)
SOURCE_BOUND_NATIVE_WEIGHTED_PARTIAL_API = (
    "round16-b:moe-comm-preselected-weighted-partial:offline-contract-v1"
)
SOURCE_BOUND_SHARED_EXECUTE_API = "round16-b:shared-expert-execute:v1"
SOURCE_BOUND_FORWARD_CONTEXT_API = "round16-b:forward-context-provider:v1"
SOURCE_BOUND_MOE_COMM_PREPARE_API = (
    "round16-b:moe-comm-prepare-before-selector:source-contract-v1"
)
SOURCE_BOUND_MOE_COMM_CONTEXT_API = (
    "round16-b:current-moe-comm-object-provider:source-contract-v1"
)
SOURCE_BOUND_MOE_COMM_CLEANUP_API = (
    "round16-b:moe-comm-one-shot-cleanup:source-contract-v1"
)
UPSTREAM_MOE_RUNNER_SOURCE_API = (
    "vllm@568afb3a13806beb53bb2e6bd518269357b237c0:MoERunner:source-v1"
)
SOURCE_BOUND_NATIVE_RUNNER_FACTORY_API = (
    "round16-b:qwen-self-experts-native-runner-factory:offline-v1"
)


def _freeze_ids(rows: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    try:
        frozen = tuple(tuple(row) for row in rows)
    except TypeError as error:
        raise ValueError("Top-K ids must be nested sequences") from error
    if any(type(value) is not int for row in frozen for value in row):
        raise ValueError("Top-K expert ids must be exact integers")
    return frozen


def _freeze_weights(
    rows: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    try:
        frozen = tuple(tuple(row) for row in rows)
    except TypeError as error:
        raise ValueError("Top-K weights must be nested sequences") from error
    converted: list[tuple[float, ...]] = []
    for row in frozen:
        values: list[float] = []
        for value in row:
            if type(value) not in (int, float):
                raise ValueError("Top-K weights must be finite numbers")
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError("Top-K weights must be finite numbers")
            values.append(numeric)
        converted.append(tuple(values))
    return tuple(converted)


@dataclass(frozen=True)
class FrozenTopK:
    """A router result frozen before either execution branch can consume it."""

    topk_ids: tuple[tuple[int, ...], ...]
    topk_weights: tuple[tuple[float, ...], ...]
    valid_rows: tuple[bool, ...]
    routed_scaling_applied: bool = False

    def __post_init__(self) -> None:
        ids = _freeze_ids(self.topk_ids)
        weights = _freeze_weights(self.topk_weights)
        try:
            valid = tuple(self.valid_rows)
        except TypeError as error:
            raise ValueError("valid_rows must be a sequence") from error
        if any(type(value) is not bool for value in valid):
            raise ValueError("valid_rows entries must be exact booleans")
        if type(self.routed_scaling_applied) is not bool:
            raise ValueError("routed scaling marker must be a boolean")
        if self.routed_scaling_applied:
            raise ValueError(
                "routed scaling must remain unapplied until the single merge owner"
            )
        # RouteBatch performs row-count, width, padding, and finite checks.
        routes = RouteBatch(ids, weights, valid)
        if routes.token_count and routes.top_k != 10:
            raise ValueError("source-bound Hybrid routing requires frozen Top-10")
        object.__setattr__(self, "topk_ids", ids)
        object.__setattr__(self, "topk_weights", weights)
        object.__setattr__(self, "valid_rows", valid)

    def as_route_batch(self) -> RouteBatch:
        return RouteBatch(self.topk_ids, self.topk_weights, self.valid_rows)


def _require_identity(callback: Callable[..., Any], expected: str, name: str) -> None:
    if getattr(callback, "round16_api_identity", None) != expected:
        raise ValueError(f"{name} API identity is not verified")


@dataclass(frozen=True)
class SourceBoundRunnerDependencies:
    lifecycle: RequestLeaseManager
    route_once: Callable[..., FrozenTopK]
    cpu_backend: KTExpertBackend | KTWeightedPartialBackend
    native_preselected_execute: (
        Callable[..., Mapping[Any, Sequence[float]]] | None
    ) = None
    shared_execute: Callable[..., Sequence[Sequence[float]]] | None = None
    forward_context_provider: Callable[[str], "SourceBoundForwardContext"] | None = None
    native_moe_comm_prepare: Callable[..., "NativeMoeCommPrepared"] | None = None
    moe_comm_method_provider: Callable[[], Any] | None = None
    native_weighted_partial_execute: Callable[..., "NativeWeightedPartial"] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lifecycle, RequestLeaseManager):
            raise TypeError("lifecycle must be a RequestLeaseManager")
        if not callable(self.route_once):
            raise TypeError("route_once must be callable")
        _require_identity(
            self.route_once, SOURCE_BOUND_ROUTE_ONCE_API, "route-once"
        )
        if self.native_preselected_execute is not None:
            if not callable(self.native_preselected_execute):
                raise TypeError(
                    "native_preselected_execute must be callable or None"
                )
            _require_identity(
                self.native_preselected_execute,
                SOURCE_BOUND_NATIVE_PRESELECTED_API,
                "native preselected execute",
            )
        if not isinstance(
            self.cpu_backend, (KTExpertBackend, KTWeightedPartialBackend)
        ):
            raise TypeError("cpu_backend must be a reviewed KT backend")
        if self.shared_execute is not None:
            if not callable(self.shared_execute):
                raise TypeError("shared_execute must be callable or None")
            _require_identity(
                self.shared_execute,
                SOURCE_BOUND_SHARED_EXECUTE_API,
                "shared execute",
            )
        if self.forward_context_provider is not None:
            if not callable(self.forward_context_provider):
                raise TypeError("forward_context_provider must be callable or None")
            _require_identity(
                self.forward_context_provider,
                SOURCE_BOUND_FORWARD_CONTEXT_API,
                "forward context provider",
            )
        if self.native_moe_comm_prepare is not None:
            if not callable(self.native_moe_comm_prepare):
                raise TypeError("native_moe_comm_prepare must be callable or None")
            _require_identity(
                self.native_moe_comm_prepare,
                SOURCE_BOUND_MOE_COMM_PREPARE_API,
                "native MoEComm prepare",
            )
        if self.moe_comm_method_provider is not None:
            if not callable(self.moe_comm_method_provider):
                raise TypeError("moe_comm_method_provider must be callable or None")
            _require_identity(
                self.moe_comm_method_provider,
                SOURCE_BOUND_MOE_COMM_CONTEXT_API,
                "current MoEComm object provider",
            )
        if self.native_weighted_partial_execute is not None:
            if not callable(self.native_weighted_partial_execute):
                raise TypeError(
                    "native_weighted_partial_execute must be callable or None"
                )
            _require_identity(
                self.native_weighted_partial_execute,
                SOURCE_BOUND_NATIVE_WEIGHTED_PARTIAL_API,
                "native weighted partial execute",
            )


@dataclass(frozen=True)
class SourceBoundForwardContext:
    """Request identity supplied when Qwen calls ``self.experts.forward``."""

    lease: RequestLease
    input_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.lease, RequestLease):
            raise TypeError("forward context lease must be a RequestLease")
        if type(self.input_fingerprint) is not str or not self.input_fingerprint:
            raise ValueError("forward context fingerprint must be a non-empty string")


@dataclass(frozen=True)
class NativeMoeCommPrepared:
    """Opaque result of the exact donor ``MoECommMethod.prepare`` envelope.

    The first live candidate is intentionally bounded to TP1/EP1 ALLGATHER.
    ALLGATHER does not remap expert IDs, so a 512-to-32 expert map must not be
    supplied to it.  The native adapter remains responsible for masking CPU
    occurrences with zero weight and a safe resident placeholder before its
    builder consumes the frozen Top-K arrays.
    """

    comm_method: Any
    selector_hidden_states: Any
    selector_router_logits: Any
    valid_rows: tuple[bool, ...]
    opaque_state: Any
    comm_type: str
    ep_size: int
    tp_size: int
    quant_type: str
    expert_map: Any
    log2phy: Any
    physical_expert_count: int
    cleanup: Callable[["NativeMoeCommPrepared"], None] | None = None

    def __post_init__(self) -> None:
        if self.comm_method is None:
            raise ValueError("prepared MoEComm object identity is required")
        try:
            valid_rows = tuple(self.valid_rows)
        except TypeError as error:
            raise ValueError("prepared valid_rows must be a sequence") from error
        if any(type(value) is not bool for value in valid_rows):
            raise ValueError("prepared valid_rows entries must be exact booleans")
        object.__setattr__(self, "valid_rows", valid_rows)
        if self.comm_type != "ALLGATHER":
            raise ValueError("first source-bound path requires ALLGATHER")
        if type(self.ep_size) is not int or self.ep_size != 1:
            raise ValueError("first source-bound path requires EP1")
        if type(self.tp_size) is not int or self.tp_size != 1:
            raise ValueError("first source-bound path requires TP1")
        if self.quant_type != "NONE":
            raise ValueError("first source-bound path requires QuantType.NONE")
        if self.expert_map is not None or self.log2phy is not None:
            raise ValueError(
                "ALLGATHER Q32 must not receive expert_map or log2phy remapping"
            )
        if (
            type(self.physical_expert_count) is not int
            or self.physical_expert_count != 32
        ):
            raise ValueError("prepared MoEComm path requires physical Q32")
        if self.cleanup is not None:
            if not callable(self.cleanup):
                raise TypeError("prepared cleanup must be callable or None")
            _require_identity(
                self.cleanup,
                SOURCE_BOUND_MOE_COMM_CLEANUP_API,
                "prepared MoEComm cleanup",
            )


@dataclass(frozen=True)
class NativeWeightedPartial:
    """Token-level native routed aggregate after Top-K weights, before scale."""

    output: tuple[tuple[float, ...], ...]
    route_weights_applied: bool = True
    routed_scaling_applied: bool = False
    prepare_applied: bool = True
    finalize_applied: bool = True

    def __post_init__(self) -> None:
        try:
            rows = tuple(tuple(float(value) for value in row) for row in self.output)
        except (TypeError, ValueError) as error:
            raise ValueError("native weighted partial must contain numeric rows") from error
        if len({len(row) for row in rows}) > 1:
            raise ValueError("native weighted partial rows must have equal width")
        if any(not isfinite(value) for row in rows for value in row):
            raise ValueError("native weighted partial must be finite")
        markers = (
            self.route_weights_applied,
            self.routed_scaling_applied,
            self.prepare_applied,
            self.finalize_applied,
        )
        if any(type(marker) is not bool for marker in markers):
            raise ValueError("native weighted partial markers must be booleans")
        if self.route_weights_applied is not True:
            raise ValueError("native aggregate must include original Top-K weights")
        if self.routed_scaling_applied is not False:
            raise ValueError("native aggregate must exclude routed scaling")
        if self.prepare_applied is not True or self.finalize_applied is not True:
            raise ValueError(
                "native aggregate must include MoEComm prepare/finalize envelope"
            )
        object.__setattr__(self, "output", rows)


@dataclass(frozen=True)
class SourceBoundDecodeResult:
    output: tuple[tuple[float, ...], ...]
    route_plan: HybridRoutePlan
    cpu_task: TaskIdentity | None
    work_id: tuple[Any, ...] | None


@dataclass(frozen=True)
class SourceBoundRunnerBinding:
    status: str
    runner_cls: type | None
    live_eligible: bool
    api_identity: str = SOURCE_BOUND_HYBRID_RUNNER_API
    unresolved_requirements: tuple[str, ...] = (
        "LIVE_QWEN_SELF_EXPERTS_IMAGE_PROOF",
        "EP1_ALLGATHER_Q32_PRESELECTED_DEVICE_ADAPTER",
        "Q32_NATIVE_OWNER_AND_RUNNER_DEVICE_PAIR",
        "S15_WEIGHTED_AGGREGATE_DEVICE_PARITY",
        "EXACT_SOURCE_TOKEN_ATOMIC_LIVE_REGISTRATION",
        "KT_NATIVE_STREAM_SINGLE_OUTSTANDING_DEVICE_DRAIN",
    )


class SourceBoundPreselectedExecutor:
    """Execute one already-selected routed branch without applying postprocessing.

    The exact upstream ``MoERunner._apply_quant_method`` owns selection and
    calls the routed owner's ``forward_modular``.  This executor begins at that
    preselected boundary and returns only the unscaled routed partial.  The
    inherited upstream ``MoERunner.forward`` remains the owner of routed scale,
    shared addition/gating, output transform, reduction, and zero-expert output.
    """

    def __init__(
        self,
        *,
        layer_name: str,
        plan: SelectiveLoadPlan,
        dependencies: SourceBoundRunnerDependencies,
    ) -> None:
        if type(layer_name) is not str or not layer_name:
            raise ValueError("layer_name must be a non-empty string")
        if not isinstance(plan, SelectiveLoadPlan):
            raise TypeError("preselected executor plan must be SelectiveLoadPlan")
        if not isinstance(dependencies, SourceBoundRunnerDependencies):
            raise TypeError("preselected executor dependencies are invalid")
        if dependencies.forward_context_provider is None:
            raise ValueError(
                "Qwen self.experts wiring requires a verified forward context provider"
            )
        if dependencies.native_weighted_partial_execute is None:
            raise ValueError(
                "Qwen self.experts wiring requires a native weighted aggregate seam"
            )
        if dependencies.native_moe_comm_prepare is None:
            raise ValueError(
                "Qwen self.experts wiring requires MoEComm prepare before selection"
            )
        if dependencies.moe_comm_method_provider is None:
            raise ValueError(
                "Qwen self.experts wiring requires current MoEComm identity provider"
            )
        if not isinstance(dependencies.cpu_backend, KTWeightedPartialBackend):
            raise TypeError(
                "Qwen self.experts wiring requires KT weighted aggregate backend"
            )
        self.layer_name = layer_name
        self.plan = plan
        self.dependencies = dependencies
        self._epoch_lock = Lock()
        self._decode_epoch = 0
        self._prepared_lock = Lock()
        self._active_prepared: NativeMoeCommPrepared | None = None

    def begin_prepared(self, prepared: NativeMoeCommPrepared) -> None:
        if not isinstance(prepared, NativeMoeCommPrepared):
            raise TypeError("native prepare must return NativeMoeCommPrepared")
        if prepared.physical_expert_count != self.plan.physical_expert_count:
            raise ValueError("prepared native physical expert count differs from plan")
        provider = self.dependencies.moe_comm_method_provider
        if provider is None or provider() is not prepared.comm_method:
            raise RuntimeError(
                "current MoEComm object changed after source-bound prepare"
            )
        with self._prepared_lock:
            if self._active_prepared is not None:
                raise RuntimeError("native MoEComm prepare envelope is already active")
            self._active_prepared = prepared

    def end_prepared(self, prepared: NativeMoeCommPrepared) -> None:
        with self._prepared_lock:
            if self._active_prepared is not prepared:
                raise RuntimeError("native MoEComm prepare envelope identity changed")
            self._active_prepared = None
        if prepared.cleanup is not None:
            prepared.cleanup(prepared)

    def _require_prepared(self) -> NativeMoeCommPrepared:
        with self._prepared_lock:
            prepared = self._active_prepared
        if prepared is None:
            raise RuntimeError(
                "preselected execution requires prepare before the selector"
            )
        provider = self.dependencies.moe_comm_method_provider
        if provider is None or provider() is not prepared.comm_method:
            raise RuntimeError(
                "current MoEComm object changed before fused_experts"
            )
        return prepared

    def _next_decode_epoch(self) -> int:
        with self._epoch_lock:
            self._decode_epoch += 1
            return self._decode_epoch

    def forward_modular(
        self,
        *,
        hidden_states: Sequence[Any],
        topk_ids: Sequence[Sequence[int]],
        topk_weights: Sequence[Sequence[float]],
        input_ids: Any = None,
    ) -> tuple[tuple[float, ...], ...]:
        del input_ids
        provider = self.dependencies.forward_context_provider
        if provider is None:
            raise RuntimeError("forward context provider disappeared")
        context = provider(self.layer_name)
        if not isinstance(context, SourceBoundForwardContext):
            raise TypeError(
                "forward context provider must return SourceBoundForwardContext"
            )
        prepared = self._require_prepared()
        if hidden_states is not prepared.selector_hidden_states:
            raise RuntimeError(
                "routed owner did not consume the prepared hidden-state object"
            )
        frozen = FrozenTopK(
            topk_ids,
            topk_weights,
            valid_rows=prepared.valid_rows,
            routed_scaling_applied=False,
        )
        return self.execute(
            context.lease,
            hidden_states,
            frozen=frozen,
            input_fingerprint=context.input_fingerprint,
            prepared=prepared,
        ).output

    def execute(
        self,
        lease: RequestLease,
        hidden_states: Sequence[Any],
        *,
        frozen: FrozenTopK,
        input_fingerprint: str,
        prepared: NativeMoeCommPrepared,
    ) -> SourceBoundDecodeResult:
        if not isinstance(lease, RequestLease):
            raise TypeError("preselected execution requires a RequestLease")
        if type(input_fingerprint) is not str or not input_fingerprint:
            raise ValueError("input_fingerprint must be a non-empty string")
        if not isinstance(prepared, NativeMoeCommPrepared):
            raise TypeError("native prepared state is required")
        dependencies = self.dependencies
        dependencies.lifecycle.assert_active(lease)
        # Preserve token rows for [T,H] aggregates while making every padded
        # row harmless before either native or KT can observe the payload.
        # ``valid_rows`` remains false, and zero-weight expert 0 is only a safe
        # placeholder for APIs that still require a dense Top-K tensor.
        safe_ids = tuple(
            ids if valid else tuple(0 for _value in ids)
            for ids, valid in zip(frozen.topk_ids, frozen.valid_rows)
        )
        safe_weights = tuple(
            weights if valid else tuple(0.0 for _value in weights)
            for weights, valid in zip(frozen.topk_weights, frozen.valid_rows)
        )
        branch_frozen = FrozenTopK(
            safe_ids,
            safe_weights,
            frozen.valid_rows,
            routed_scaling_applied=False,
        )
        route_plan = plan_routes(
            branch_frozen.as_route_batch(),
            resident_logical_ids=self.plan.resident_logical_ids,
        )
        decode_epoch = self._next_decode_epoch()
        base_identity = (
            *lease.identity,
            self.layer_name,
            "decode_epoch",
            decode_epoch,
            "input_fingerprint",
            input_fingerprint,
        )
        task: TaskIdentity | None = None
        work_id: tuple[Any, ...] | None = None
        if route_plan.cpu_routes:
            task = dependencies.lifecycle.begin_task(lease, "cpu-moe-decode")
            work_id = (*task.identity, *base_identity)
        identity_key = work_id if work_id is not None else base_identity
        kt_pending: KTPendingWeightedPartial | None = None
        try:
            dependencies.lifecycle.assert_active(lease)
            if task is not None and isinstance(
                dependencies.cpu_backend, KTWeightedPartialBackend
            ):
                kt_pending = dependencies.cpu_backend.submit(
                    route_plan,
                    hidden_states,
                    identity_key=identity_key,
                )
                dependencies.lifecycle.assert_active(lease)
            native_partial = dependencies.native_weighted_partial_execute(
                prepared, route_plan, hidden_states, identity_key
            )
            if native_partial is not None and not isinstance(
                native_partial, NativeWeightedPartial
            ):
                raise TypeError(
                    "native preselected seam must return NativeWeightedPartial"
                )
            dependencies.lifecycle.assert_active(lease)
            if task is not None:
                if isinstance(dependencies.cpu_backend, KTWeightedPartialBackend):
                    if kt_pending is None:
                        raise RuntimeError("KT CPU branch was not submitted")
                    cpu_result = dependencies.cpu_backend.sync(kt_pending)
                    kt_pending = None
                else:
                    cpu_result = dependencies.cpu_backend.execute(
                        route_plan,
                        hidden_states,
                        identity_key=identity_key,
                    )
                if not dependencies.lifecycle.complete(task, work_id):
                    raise RuntimeError("late or cancelled CPU completion was rejected")
            else:
                cpu_result = {}
            dependencies.lifecycle.assert_active(lease)
            if not isinstance(cpu_result, (KTWeightedPartial, dict)):
                raise TypeError("CPU branch must return a reviewed weighted partial")
            cpu_partial = cpu_result if isinstance(cpu_result, KTWeightedPartial) else None
            output = self._merge_weighted_aggregates(
                native_partial,
                cpu_partial,
                token_count=route_plan.routes.token_count,
            )
            return SourceBoundDecodeResult(output, route_plan, task, work_id)
        except BaseException as error:
            if kt_pending is not None and isinstance(
                dependencies.cpu_backend, KTWeightedPartialBackend
            ):
                try:
                    dependencies.cpu_backend.drain_discard(kt_pending)
                except BaseException as cleanup_error:
                    error.add_note(
                        "KT pending drain also failed: " + str(cleanup_error)
                    )
            if task is not None:
                dependencies.lifecycle.abort_task(task)
            raise

    @staticmethod
    def _merge_weighted_aggregates(
        native: NativeWeightedPartial | None,
        cpu: KTWeightedPartial | None,
        *,
        token_count: int,
    ) -> tuple[tuple[float, ...], ...]:
        if type(token_count) is not int or token_count < 0:
            raise ValueError("token_count must be a non-negative integer")
        partials = tuple(item for item in (native, cpu) if item is not None)
        if not partials:
            return tuple(() for _ in range(token_count))
        for partial in partials:
            if partial.route_weights_applied is not True:
                raise ValueError("routed aggregate is missing Top-K weights")
            if partial.routed_scaling_applied is not False:
                raise ValueError("routed aggregate already contains routed scaling")
            if len(partial.output) != token_count:
                raise ValueError("routed aggregate row count differs from tokens")
        widths = {
            len(row)
            for partial in partials
            for row in partial.output
        }
        if len(widths) > 1:
            raise ValueError("native and CPU aggregate widths differ")
        width = next(iter(widths), 0)
        merged = [[0.0] * width for _ in range(token_count)]
        for partial in partials:
            for row_index, row in enumerate(partial.output):
                for column, value in enumerate(row):
                    merged[row_index][column] += value
        return tuple(tuple(row) for row in merged)


def build_source_bound_native_runner_class(
    upstream_moe_runner_cls: type,
    *,
    upstream_api_identity: str,
) -> type:
    """Bind the bridge below exact upstream routing and above postprocessing."""

    if not isinstance(upstream_moe_runner_cls, type):
        raise TypeError("upstream MoERunner must be a class")
    if upstream_api_identity != UPSTREAM_MOE_RUNNER_SOURCE_API:
        raise ValueError("upstream MoERunner exact-source API identity mismatch")
    for method_name in ("forward", "_apply_quant_method"):
        if not callable(getattr(upstream_moe_runner_cls, method_name, None)):
            raise TypeError(f"upstream MoERunner lacks {method_name}")

    class SourceBoundNativeMoERunner(upstream_moe_runner_cls):
        round16_source_bound_runner_api_identity = SOURCE_BOUND_HYBRID_RUNNER_API
        round16_native_runner_factory_identity = (
            SOURCE_BOUND_NATIVE_RUNNER_FACTORY_API
        )
        runtime_runner_binding_verified = False
        live_eligible = False
        router_owner = "UPSTREAM_MOERUNNER_APPLY_QUANT_METHOD"
        routed_scaling_owner = "UPSTREAM_MOERUNNER_FORWARD"
        shared_owner = "UPSTREAM_MOERUNNER_FORWARD_AND_SHARED_HELPERS"
        output_transform_owner = "UPSTREAM_MOERUNNER_FORWARD"
        reduction_owner = "UPSTREAM_MOERUNNER_FORWARD"
        residual_owner = "CALLING_QWEN_LAYER"

        def __init__(
            self,
            layer_name: str,
            moe_config: Any,
            router: Any,
            routed_experts: Any,
            enable_dbo: bool = False,
            gate: Any = None,
            shared_experts: Any = None,
            shared_expert_gate: Any = None,
            routed_input_transform: Any = None,
            routed_output_transform: Any = None,
            routed_scaling_factor: float = 1.0,
            *,
            round16_selective_plan: SelectiveLoadPlan,
            round16_runner_dependencies: SourceBoundRunnerDependencies,
            **runner_kwargs: Any,
        ) -> None:
            if not isinstance(round16_selective_plan, SelectiveLoadPlan):
                raise TypeError("round16_selective_plan must be a SelectiveLoadPlan")
            if (
                getattr(routed_experts, "round16_selective_plan", None)
                is not round16_selective_plan
            ):
                raise ValueError(
                    "runner and routed owner must consume the identical plan"
                )
            if not isinstance(
                round16_runner_dependencies, SourceBoundRunnerDependencies
            ):
                raise TypeError("round16 runner dependencies are invalid")
            binder = getattr(
                routed_experts, "round16_bind_preselected_executor", None
            )
            if not callable(binder):
                raise TypeError(
                    "Q32 routed owner lacks preselected forward_modular binder"
                )
            super().__init__(
                layer_name=layer_name,
                moe_config=moe_config,
                router=router,
                routed_experts=routed_experts,
                enable_dbo=enable_dbo,
                gate=gate,
                shared_experts=shared_experts,
                shared_expert_gate=shared_expert_gate,
                routed_input_transform=routed_input_transform,
                routed_output_transform=routed_output_transform,
                routed_scaling_factor=routed_scaling_factor,
                **runner_kwargs,
            )
            if getattr(self, "routed_experts", None) is not routed_experts:
                raise RuntimeError("upstream MoERunner replaced the Q32 owner")
            executor = SourceBoundPreselectedExecutor(
                layer_name=layer_name,
                plan=round16_selective_plan,
                dependencies=round16_runner_dependencies,
            )
            binder(executor.forward_modular, round16_selective_plan)
            self.round16_selective_plan = round16_selective_plan
            self.round16_routed_owner = routed_experts
            self.round16_preselected_executor = executor
            self.round16_runner_owner_binding_verified = True

        def _apply_quant_method(
            self,
            hidden_states: Any,
            router_logits: Any,
            shared_experts_input: Any,
            input_ids: Any = None,
        ) -> Any:
            """Wrap exact upstream selection in the donor prepare envelope.

            Selection, shared-expert helpers, and the routed-owner call remain
            inside the reviewed upstream implementation.  The only added
            behavior is binding a one-shot prepared MoEComm object before that
            selector runs so ``forward_modular`` can call the preselected
            builder/fused/finalize seam without invoking Ascend quant apply.
            """

            prepare = self.round16_preselected_executor.dependencies.native_moe_comm_prepare
            if prepare is None:
                raise RuntimeError("native MoEComm prepare dependency disappeared")
            prepared = prepare(hidden_states, router_logits, self.layer_name)
            began = False
            primary_error: BaseException | None = None
            try:
                self.round16_preselected_executor.begin_prepared(prepared)
                began = True
                return super()._apply_quant_method(
                    prepared.selector_hidden_states,
                    prepared.selector_router_logits,
                    shared_experts_input,
                    input_ids,
                )
            except BaseException as error:
                primary_error = error
                raise
            finally:
                try:
                    if began:
                        self.round16_preselected_executor.end_prepared(prepared)
                    elif prepared.cleanup is not None:
                        prepared.cleanup(prepared)
                except BaseException as cleanup_error:
                    if primary_error is None:
                        raise
                    primary_error.add_note(
                        "native MoEComm cleanup also failed: " + str(cleanup_error)
                    )

    SourceBoundNativeMoERunner.__name__ = "SourceBoundNativeMoERunner"
    SourceBoundNativeMoERunner.__qualname__ = "SourceBoundNativeMoERunner"
    SourceBoundNativeMoERunner.__module__ = __name__
    return SourceBoundNativeMoERunner


class SourceBoundHybridRunner:
    """Dependency-injected arithmetic/lifecycle contract for the runner seam."""

    round16_source_bound_runner_api_identity = SOURCE_BOUND_HYBRID_RUNNER_API
    # Source review proves where this class must bind, but the native
    # preselected-Top-K kernel and KT device callback are not yet available.
    runtime_runner_binding_verified = False
    live_eligible = False
    residual_owner = "CALLING_MODEL_LAYER"
    router_owner = "SOURCE_BOUND_HYBRID_RUNNER"
    routed_weight_owner = "SOURCE_BOUND_HYBRID_RUNNER_MERGE"
    routed_scaling_owner = "UPSTREAM_MOERUNNER_FORWARD_EQUIVALENT"
    shared_owner = "UPSTREAM_MOERUNNER_FORWARD_EQUIVALENT"

    def __init__(
        self,
        layer_name: str,
        moe_config: Any,
        router: Any,
        routed_experts: Any,
        enable_dbo: bool = False,
        gate: Any = None,
        shared_experts: Any = None,
        shared_expert_gate: Any = None,
        routed_input_transform: Any = None,
        routed_output_transform: Any = None,
        routed_scaling_factor: float = 1.0,
        *,
        round16_selective_plan: SelectiveLoadPlan,
        round16_runner_dependencies: SourceBoundRunnerDependencies,
        **runner_kwargs: Any,
    ) -> None:
        del (
            moe_config,
            router,
            enable_dbo,
            gate,
            shared_experts,
            shared_expert_gate,
            routed_input_transform,
            routed_output_transform,
            runner_kwargs,
        )
        if type(layer_name) is not str or not layer_name:
            raise ValueError("layer_name must be a non-empty string")
        if not isinstance(round16_selective_plan, SelectiveLoadPlan):
            raise TypeError("round16_selective_plan must be a SelectiveLoadPlan")
        if (
            getattr(routed_experts, "round16_selective_plan", None)
            is not round16_selective_plan
        ):
            raise ValueError("runner and routed owner must consume the identical plan")
        if not isinstance(
            round16_runner_dependencies, SourceBoundRunnerDependencies
        ):
            raise TypeError(
                "round16_runner_dependencies must be SourceBoundRunnerDependencies"
            )
        if type(routed_scaling_factor) not in (int, float) or not isfinite(
            float(routed_scaling_factor)
        ):
            raise ValueError("routed scaling factor must be finite")
        self.layer_name = layer_name
        self.routed_experts = routed_experts
        self._plan = round16_selective_plan
        self.round16_selective_plan = round16_selective_plan
        self.round16_routed_owner = routed_experts
        self.round16_runner_owner_binding_verified = True
        self._dependencies = round16_runner_dependencies
        self.routed_scaling_factor = float(routed_scaling_factor)
        self._epoch_lock = Lock()
        self._decode_epoch = 0

    def _next_decode_epoch(self) -> int:
        with self._epoch_lock:
            self._decode_epoch += 1
            return self._decode_epoch

    @staticmethod
    def _merge_kt_weighted_partial(
        route_plan: HybridRoutePlan,
        npu_outputs: Mapping[tuple[int, int, int], Sequence[float]],
        cpu_partial: KTWeightedPartial,
        shared_output: Sequence[Sequence[float]] | None,
        scaling: float,
    ) -> tuple[tuple[float, ...], ...]:
        if cpu_partial.route_weights_applied is not True:
            raise ValueError("KT CPU partial is missing Top-K route weights")
        if cpu_partial.routed_scaling_applied is not False:
            raise ValueError("KT CPU partial already contains routed scaling")
        width = len(cpu_partial.output[0]) if cpu_partial.output else 0
        zero_cpu = {
            route.key: tuple(0.0 for _ in range(width))
            for route in route_plan.cpu_routes
        }
        native_partial = merge_route_outputs(
            route_plan,
            npu_outputs,
            zero_cpu,
            scaling=1.0,
        )
        if len(native_partial) != len(cpu_partial.output):
            raise ValueError("native and KT routed partial row counts differ")
        merged: list[tuple[float, ...]] = []
        for row_index, (native, cpu) in enumerate(
            zip(native_partial, cpu_partial.output)
        ):
            if len(native) != len(cpu):
                raise ValueError("native and KT routed partial widths differ")
            routed = tuple((left + right) * scaling for left, right in zip(native, cpu))
            if shared_output is not None:
                if len(shared_output) != len(cpu_partial.output):
                    raise ValueError("shared output row count must match route tokens")
                shared = tuple(shared_output[row_index])
                if len(shared) != len(routed):
                    raise ValueError("shared and routed output widths differ")
                routed = tuple(
                    value + float(shared_value)
                    for value, shared_value in zip(routed, shared)
                )
            merged.append(routed)
        return tuple(merged)

    def run_decode(
        self,
        lease: RequestLease,
        hidden_states: Sequence[Any],
        *,
        router_logits: Any,
        input_fingerprint: str,
    ) -> SourceBoundDecodeResult:
        if not isinstance(lease, RequestLease):
            raise TypeError("run_decode requires a RequestLease")
        if type(input_fingerprint) is not str or not input_fingerprint:
            raise ValueError("input_fingerprint must be a non-empty string")
        dependencies = self._dependencies
        if dependencies.native_preselected_execute is None:
            raise RuntimeError(
                "legacy occurrence-mapping decode requires its mock-only "
                "native_preselected_execute callback"
            )
        dependencies.lifecycle.assert_active(lease)
        frozen = dependencies.route_once(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=10,
            routed_scaling_factor=1.0,
        )
        if not isinstance(frozen, FrozenTopK):
            raise TypeError("route_once must return FrozenTopK")
        route_plan = plan_routes(
            frozen.as_route_batch(),
            resident_logical_ids=self._plan.resident_logical_ids,
        )
        decode_epoch = self._next_decode_epoch()
        base_identity = (
            *lease.identity,
            self.layer_name,
            "decode_epoch",
            decode_epoch,
            "input_fingerprint",
            input_fingerprint,
        )
        task: TaskIdentity | None = None
        work_id: tuple[Any, ...] | None = None
        if route_plan.cpu_routes:
            task = dependencies.lifecycle.begin_task(lease, "cpu-moe-decode")
            work_id = (*task.identity, *base_identity)
        identity_key = work_id if work_id is not None else base_identity
        kt_pending: KTPendingWeightedPartial | None = None
        try:
            dependencies.lifecycle.assert_active(lease)
            if task is not None and isinstance(
                dependencies.cpu_backend, KTWeightedPartialBackend
            ):
                kt_pending = dependencies.cpu_backend.submit(
                    route_plan,
                    hidden_states,
                    identity_key=identity_key,
                )
                dependencies.lifecycle.assert_active(lease)
            if route_plan.npu_routes:
                npu_outputs = dependencies.native_preselected_execute(
                    route_plan, hidden_states, identity_key
                )
            else:
                npu_outputs = {}
            dependencies.lifecycle.assert_active(lease)
            if task is not None:
                if isinstance(
                    dependencies.cpu_backend, KTWeightedPartialBackend
                ):
                    if kt_pending is None:
                        raise RuntimeError("KT CPU branch was not submitted")
                    cpu_result = dependencies.cpu_backend.sync(kt_pending)
                    kt_pending = None
                else:
                    cpu_result = dependencies.cpu_backend.execute(
                        route_plan,
                        hidden_states,
                        identity_key=identity_key,
                    )
                if not dependencies.lifecycle.complete(task, work_id):
                    raise RuntimeError(
                        "late or cancelled CPU completion was rejected"
                    )
            else:
                cpu_result = {}
            dependencies.lifecycle.assert_active(lease)
            shared_output = (
                dependencies.shared_execute(hidden_states, identity_key)
                if dependencies.shared_execute is not None
                else None
            )
            if isinstance(cpu_result, KTWeightedPartial):
                output = self._merge_kt_weighted_partial(
                    route_plan,
                    npu_outputs,
                    cpu_result,
                    shared_output,
                    self.routed_scaling_factor,
                )
            else:
                output = merge_route_outputs(
                    route_plan,
                    npu_outputs,
                    cpu_result,
                    shared_output=shared_output,
                    scaling=self.routed_scaling_factor,
                )
            return SourceBoundDecodeResult(output, route_plan, task, work_id)
        except BaseException as error:
            if kt_pending is not None and isinstance(
                dependencies.cpu_backend, KTWeightedPartialBackend
            ):
                try:
                    dependencies.cpu_backend.drain_discard(kt_pending)
                except BaseException as cleanup_error:
                    error.add_note(
                        "KT pending drain also failed: " + str(cleanup_error)
                    )
            if task is not None:
                dependencies.lifecycle.abort_task(task)
            raise


def build_source_bound_hybrid_runner_binding(
    *,
    enabled: bool = False,
    dependencies: SourceBoundRunnerDependencies | None = None,
) -> SourceBoundRunnerBinding:
    """Build the offline runner seam; never claim a device-live binding."""

    if type(enabled) is not bool:
        return SourceBoundRunnerBinding("INVALID_ENABLE_FLAG", None, False)
    if not enabled:
        return SourceBoundRunnerBinding("DISABLED_DEFAULT_OFF", None, False)
    if not isinstance(dependencies, SourceBoundRunnerDependencies):
        raise ValueError("verified source-bound runner dependencies are required")
    return SourceBoundRunnerBinding(
        "DEVICE_NATIVE_ABI_UNRESOLVED",
        SourceBoundHybridRunner,
        False,
    )


__all__ = (
    "FrozenTopK",
    "SOURCE_BOUND_HYBRID_RUNNER_API",
    "SOURCE_BOUND_FORWARD_CONTEXT_API",
    "SOURCE_BOUND_MOE_COMM_CONTEXT_API",
    "SOURCE_BOUND_MOE_COMM_CLEANUP_API",
    "SOURCE_BOUND_MOE_COMM_PREPARE_API",
    "SOURCE_BOUND_NATIVE_RUNNER_FACTORY_API",
    "SOURCE_BOUND_NATIVE_PRESELECTED_API",
    "SOURCE_BOUND_NATIVE_WEIGHTED_PARTIAL_API",
    "SOURCE_BOUND_ROUTE_ONCE_API",
    "SOURCE_BOUND_SHARED_EXECUTE_API",
    "SourceBoundDecodeResult",
    "SourceBoundForwardContext",
    "NativeMoeCommPrepared",
    "NativeWeightedPartial",
    "SourceBoundHybridRunner",
    "SourceBoundPreselectedExecutor",
    "SourceBoundRunnerBinding",
    "SourceBoundRunnerDependencies",
    "build_source_bound_hybrid_runner_binding",
    "build_source_bound_native_runner_class",
    "UPSTREAM_MOE_RUNNER_SOURCE_API",
)
