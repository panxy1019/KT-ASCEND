"""Exact-version, default-OFF worker hook for the frozen donor image.

This module imports no torch, vLLM, or vLLM-Ascend package.  The donor overlay
injects those objects only after hashing the exact sources listed below.  The
hook surrounds the one ``NPUModelRunner.load_model`` call that constructs the
target Qwen model and loads its checkpoint, so physical-Q32 ownership is
chosen before ``RoutedExperts.create_weights``.  Post-load registration is
committed only after all target layers retain that same owner/runner plan.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from threading import Lock, RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .allgather_q32_device_adapter import (
    AllGatherQ32DeviceTensorAdapter,
    AllGatherQ32TensorBinding,
    DeviceSelectedTopK,
)
from .model_plugin import TARGET_ARCHITECTURE
from .native_source_preflight import (
    SourcePreflightToken,
    validate_exact_source_preflight_token,
)
from .selective_loader import SelectiveLoadPlan
from .selective_preallocation_installer import (
    ModelScopedSelectivePreallocationInstaller,
    SOURCE_BOUND_HYBRID_RUNNER_API,
)
from .source_bound_hybrid_runner import NativeMoeCommPrepared


EXACT_DONOR_WORKER_HOOK_API = "round16-b:exact-donor-worker-hook:v1"
EXACT_WORKER_SOURCE_TOKEN_API = "round16-b:exact-worker-source-token:v1"
EXACT_LAYER_RUNTIME_API = "round16-b:exact-worker-layer-runtime:v1"
EXACT_LAYER_RUNTIME_FACTORY_API = (
    "round16-b:exact-worker-layer-runtime-factory:v1"
)
EXACT_UPSTREAM_MOE_RUNNER_API = (
    "vllm@568afb3a13806beb53bb2e6bd518269357b237c0:MoERunner:source-v1"
)
EXACT_WORKER_TENSOR_BINDING_FACTORY_API = (
    "round16-b:exact-worker-current-allgather-binding:v1"
)
EXACT_WORKER_FORWARD_IDENTITY_API = (
    "round16-b:exact-worker-forward-identity:v1"
)
EXACT_WORKER_EXPERT_SELECTOR_API = (
    "round16-b:exact-worker-expert-selector:v1"
)
EXACT_WORKER_KT_SUBMIT_API = "round16-integration:p1-kt-tensor-submit:v1"
EXACT_WORKER_KT_SYNC_API = "round16-integration:p1-kt-tensor-sync:v1"
EXACT_WORKER_KT_ABORT_API = "round16-integration:p1-kt-tensor-abort-drain:v1"
EXACT_WORKER_TENSOR_MERGE_API = (
    "round16-integration:p1-tensor-weighted-merge-callback:v1"
)
EXACT_MODEL_RUNNER_MODULE = "vllm_ascend.worker.model_runner_v1"
EXACT_MODEL_RUNNER_CLASS = "NPUModelRunner"

# These are the bounded files that establish the worker/model construction,
# target-Qwen FusedMoE binding, and current forward-context/comm seam.  The
# broader 17-source+config token remains mandatory on every live layer binding.
EXACT_WORKER_SOURCE_SHA256 = MappingProxyType(
    {
        "worker.py": "eb0bb11de6dffecbed135979655cf032cccef3f3ea4fe3b93254933d47e0f8c0",
        "model_runner_v1.py": "de9f108bb1365c7eb5967c0cd098a005510fe627f7a27e0732d120a0f33e2608",
        "qwen4_exp/model.py": "b06a3123073234efb39958c502b53bee23384d1a7eb61f40d12b3d8b55481133",
        "qwen4_exp/nvidia/model.py": "a38101a1c4a3cb078b0169d3c4b7b6c4b39ba41db8283a29cf8fcc5eb7ac9689",
        "fused_moe.py": "2d03cf6963f4e7edd529751383c779c407560011760c86478cf0598661d47f7c",
        "ascend_forward_context.py": "af4a6e79ffcf6803c2a2a41dd598618398f090ea3a97253642c340ea26295c9f",
    }
)


def _canonical_digest(value: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ExactWorkerSourceToken:
    api_identity: str
    observed_sha256: tuple[tuple[str, str], ...]
    binding_sha256: str


def validate_exact_worker_source_identity(
    observed_sha256: Mapping[str, str],
) -> ExactWorkerSourceToken:
    """Return an unforgeable-by-accident token for the exact bounded sources."""

    if not isinstance(observed_sha256, Mapping):
        raise TypeError("worker source identity must be a mapping")
    observed = dict(observed_sha256)
    expected = dict(EXACT_WORKER_SOURCE_SHA256)
    if observed != expected:
        raise ValueError("exact donor worker source identity mismatch")
    frozen = tuple((name, observed[name]) for name in expected)
    return ExactWorkerSourceToken(
        EXACT_WORKER_SOURCE_TOKEN_API,
        frozen,
        _canonical_digest(observed),
    )


def validate_exact_worker_source_token(token: Any) -> bool:
    if not isinstance(token, ExactWorkerSourceToken):
        return False
    expected = tuple(EXACT_WORKER_SOURCE_SHA256.items())
    return (
        token.api_identity == EXACT_WORKER_SOURCE_TOKEN_API
        and token.observed_sha256 == expected
        and token.binding_sha256
        == _canonical_digest(dict(EXACT_WORKER_SOURCE_SHA256))
    )


def load_exact_worker_source_receipt(
    receipt: str | bytes | Mapping[str, Any],
) -> ExactWorkerSourceToken:
    """Load the bounded preflight receipt without trusting caller constants."""

    if isinstance(receipt, bytes):
        payload = json.loads(receipt)
    elif isinstance(receipt, str):
        payload = json.loads(receipt)
    elif isinstance(receipt, Mapping):
        payload = dict(receipt)
    else:
        raise TypeError("exact worker receipt must be JSON or a mapping")
    if (
        payload.get("status") != "PASS_B2_EXACT_WORKER_SOURCE"
        or payload.get("passed") is not True
    ):
        raise ValueError("exact worker preflight receipt is not passing")
    observed = payload.get("observed_sha256")
    if not isinstance(observed, Mapping):
        raise ValueError("exact worker preflight observed digests are missing")
    token = validate_exact_worker_source_identity(observed)
    if payload.get("worker_source_binding_sha256") != token.binding_sha256:
        raise ValueError("exact worker preflight receipt binding differs")
    base_binding = payload.get("base_source_binding_sha256")
    if type(base_binding) is not str or len(base_binding) != 64:
        raise ValueError("broader exact-source binding is missing")
    return token


def _require_callback(callback: Any, identity: str, name: str) -> None:
    if not callable(callback):
        raise TypeError(f"{name} must be callable")
    if getattr(callback, "round16_api_identity", None) != identity:
        raise ValueError(f"{name} API identity is not verified")


class ExactWorkerRuntimeRegistry:
    """Process-local post-load registry bound to one plan and source token."""

    api_identity = EXACT_DONOR_WORKER_HOOK_API

    def __init__(
        self,
        *,
        source_token: ExactWorkerSourceToken,
        plan: SelectiveLoadPlan,
        expected_layer_count: int,
    ) -> None:
        if not validate_exact_worker_source_token(source_token):
            raise ValueError("exact worker source token is required")
        if not isinstance(plan, SelectiveLoadPlan):
            raise TypeError("runtime registry plan must be SelectiveLoadPlan")
        if plan.logical_expert_count != 512 or plan.physical_expert_count != 32:
            raise ValueError("runtime registry requires logical512/physical32")
        if type(expected_layer_count) is not int or expected_layer_count <= 0:
            raise ValueError("expected layer count must be a positive integer")
        self.source_token = source_token
        self.plan = plan
        self.expected_layer_count = expected_layer_count
        self._lock = RLock()
        self._layers: dict[str, tuple[Any, Any]] = {}

    @property
    def layer_count(self) -> int:
        with self._lock:
            return len(self._layers)

    def register_prepared(
        self,
        prepared: Sequence[tuple[str, Any, Any]],
    ) -> None:
        """Validate all layers before making any one of them visible."""

        try:
            frozen = tuple(prepared)
        except TypeError as error:
            raise TypeError("prepared layer registrations must be a sequence") from error
        if len(frozen) != self.expected_layer_count:
            raise RuntimeError(
                "exact worker registration expected "
                f"{self.expected_layer_count} layers but observed {len(frozen)}"
            )
        candidate: dict[str, tuple[Any, Any]] = {}
        for layer_name, runner, runtime in frozen:
            if type(layer_name) is not str or not layer_name:
                raise ValueError("registered layer name must be non-empty")
            if layer_name in candidate:
                raise RuntimeError("duplicate exact worker layer registration")
            if (
                getattr(runner, "round16_source_bound_runner_api_identity", None)
                != SOURCE_BOUND_HYBRID_RUNNER_API
            ):
                raise ValueError("registered runner identity is not source-bound")
            if getattr(runner, "round16_selective_plan", None) is not self.plan:
                raise ValueError("registered runner does not retain the Q32 plan")
            if getattr(runner, "round16_runtime_registry", None) is not self:
                raise ValueError("registered runner uses a different worker registry")
            owner = getattr(runner, "routed_experts", None)
            if (
                owner is None
                or getattr(runner, "round16_routed_owner", owner) is not owner
                or getattr(owner, "round16_selective_plan", None) is not self.plan
            ):
                raise ValueError("registered runner/Q32 owner identity differs")
            if getattr(runtime, "round16_api_identity", None) != EXACT_LAYER_RUNTIME_API:
                raise ValueError("registered exact layer runtime identity mismatch")
            if getattr(runtime, "layer_name", None) != layer_name:
                raise ValueError("registered runtime layer name mismatch")
            if getattr(runtime, "runner", None) is not runner:
                raise ValueError("registered runtime runner identity mismatch")
            if getattr(runtime, "routed_owner", None) is not owner:
                raise ValueError("registered runtime owner identity mismatch")
            for method_name in ("prepare", "execute_selected", "abort_prepared"):
                if not callable(getattr(runtime, method_name, None)):
                    raise TypeError(
                        f"registered exact layer runtime lacks {method_name}"
                    )
            candidate[layer_name] = (runner, runtime)
        with self._lock:
            if self._layers:
                if all(
                    name in self._layers
                    and self._layers[name][0] is pair[0]
                    and self._layers[name][1] is pair[1]
                    for name, pair in candidate.items()
                ) and len(candidate) == len(self._layers):
                    return
                raise RuntimeError("exact worker runtime registry is already bound")
            self._layers = candidate

    def require(self, layer_name: str, runner: Any) -> Any:
        with self._lock:
            pair = self._layers.get(layer_name)
        if pair is None:
            raise RuntimeError(
                "exact worker layer runtime is not registered after model load"
            )
        registered_runner, runtime = pair
        if registered_runner is not runner:
            raise RuntimeError("exact worker runner instance identity changed")
        return runtime

    def clear(self) -> None:
        with self._lock:
            self._layers = {}


def build_exact_worker_runner_class(
    upstream_moe_runner_cls: type,
    *,
    upstream_api_identity: str,
    shared_order_no_overlap: Any,
    shared_order_overlap: Any,
    expert_selector: Callable[..., tuple[Any, Any]] | None = None,
) -> type:
    """Build the Qwen ``self.experts`` runner at the exact modular seam.

    The inherited upstream ``forward`` remains the sole owner of routed scale,
    shared addition/gating, output transform, reduction, and zero-expert
    output.  This override replaces only ``_apply_quant_method``: donor
    ``prepare`` runs first, the upstream router selects exactly once from the
    prepared tensors, and the registered runtime executes KT/native weighted
    aggregate branches.  Ascend quant ``apply`` is never called.
    """

    if not isinstance(upstream_moe_runner_cls, type):
        raise TypeError("upstream MoERunner must be a class")
    if upstream_api_identity != EXACT_UPSTREAM_MOE_RUNNER_API:
        raise ValueError("upstream MoERunner exact-source identity mismatch")
    for method_name in ("forward", "_maybe_apply_shared_experts"):
        if not callable(getattr(upstream_moe_runner_cls, method_name, None)):
            raise TypeError(f"upstream MoERunner lacks {method_name}")
    if expert_selector is not None:
        _require_callback(
            expert_selector,
            EXACT_WORKER_EXPERT_SELECTOR_API,
            "exact worker expert selector",
        )

    class ExactWorkerHybridMoERunner(upstream_moe_runner_cls):
        round16_source_bound_runner_api_identity = SOURCE_BOUND_HYBRID_RUNNER_API
        round16_exact_worker_runner_api_identity = EXACT_DONOR_WORKER_HOOK_API
        live_eligible = False
        router_owner = (
            "ASCEND_EXPERT_SELECTOR_ON_PREPARED_INPUTS"
            if expert_selector is not None
            else "UPSTREAM_ROUTER_SELECT_EXPERTS_ON_PREPARED_INPUTS"
        )
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
            round16_runtime_registry: ExactWorkerRuntimeRegistry,
            **runner_kwargs: Any,
        ) -> None:
            if not isinstance(round16_selective_plan, SelectiveLoadPlan):
                raise TypeError("exact worker runner plan must be SelectiveLoadPlan")
            if not isinstance(round16_runtime_registry, ExactWorkerRuntimeRegistry):
                raise TypeError("exact worker runner registry is invalid")
            if round16_runtime_registry.plan is not round16_selective_plan:
                raise ValueError("runner and worker registry plan identity differs")
            if (
                getattr(routed_experts, "round16_selective_plan", None)
                is not round16_selective_plan
            ):
                raise ValueError("runner and routed owner plan identity differs")
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
            self.round16_selective_plan = round16_selective_plan
            self.round16_routed_owner = routed_experts
            self.round16_runtime_registry = round16_runtime_registry
            self.round16_expert_selector = expert_selector
            self.round16_runner_owner_binding_verified = True

        def _apply_quant_method(
            self,
            hidden_states: Any,
            router_logits: Any,
            shared_experts_input: Any,
            input_ids: Any = None,
        ) -> tuple[Any, Any]:
            runtime = self.round16_runtime_registry.require(
                self.layer_name, self
            )
            prepared = runtime.prepare(
                hidden_states, router_logits, self.layer_name
            )
            completed = False
            primary: BaseException | None = None
            try:
                self._maybe_apply_shared_experts(
                    shared_experts_input, shared_order_no_overlap
                )
                if self.round16_expert_selector is None:
                    topk_weights, topk_ids = self.router.select_experts(
                        hidden_states=prepared.selector_hidden_states,
                        router_logits=prepared.selector_router_logits,
                        topk_indices_dtype=self._quant_method.topk_indices_dtype,
                        input_ids=input_ids,
                    )
                else:
                    topk_weights, topk_ids = self.round16_expert_selector(
                        runner=self,
                        hidden_states=prepared.selector_hidden_states,
                        router_logits=prepared.selector_router_logits,
                        input_ids=input_ids,
                    )
                selected = DeviceSelectedTopK(
                    topk_ids, topk_weights, prepared.valid_rows
                )
                fused_out = runtime.execute_selected(
                    prepared,
                    selected,
                    prepared.selector_hidden_states,
                    self.layer_name,
                )
                completed = True
                self._maybe_apply_shared_experts(
                    shared_experts_input, shared_order_overlap
                )
                return (
                    self._shared_experts.output
                    if self._shared_experts is not None
                    else None,
                    fused_out,
                )
            except BaseException as error:
                primary = error
                raise
            finally:
                if not completed:
                    try:
                        runtime.abort_prepared(prepared)
                    except BaseException as cleanup_error:
                        if primary is None:
                            raise
                        primary.add_note(
                            "exact worker prepared cleanup also failed: "
                            + repr(cleanup_error)
                        )

    ExactWorkerHybridMoERunner.__name__ = "ExactWorkerHybridMoERunner"
    ExactWorkerHybridMoERunner.__qualname__ = "ExactWorkerHybridMoERunner"
    ExactWorkerHybridMoERunner.__module__ = __name__
    return ExactWorkerHybridMoERunner


class ExactDeviceLayerRuntime:
    """Tensor-preserving forward runtime built lazily from current context.

    The current ALLGATHER comm object and current stream are resolved by the
    injected binding factory inside ``prepare``.  Consequently registration
    after model load never captures a stale or absent forward context.
    """

    round16_api_identity = EXACT_LAYER_RUNTIME_API
    live_eligible = False

    def __init__(
        self,
        *,
        layer_name: str,
        runner: Any,
        plan: SelectiveLoadPlan,
        source_token: ExactWorkerSourceToken,
        source_preflight_token: SourcePreflightToken,
        tensor_binding_factory: Callable[..., AllGatherQ32TensorBinding],
        forward_identity_provider: Callable[[str], tuple[Any, ...]],
        kt_submit: Callable[..., Any],
        kt_sync: Callable[..., Any],
        kt_abort_drain: Callable[..., None],
        merge_weighted_partials: Callable[..., Any],
    ) -> None:
        if type(layer_name) is not str or not layer_name:
            raise ValueError("exact device runtime layer name must be non-empty")
        if not validate_exact_worker_source_token(source_token):
            raise ValueError("exact worker source token is required")
        if not validate_exact_source_preflight_token(source_preflight_token):
            raise ValueError("broader exact source preflight token is required")
        if not isinstance(plan, SelectiveLoadPlan):
            raise TypeError("exact device runtime plan is invalid")
        owner = getattr(runner, "routed_experts", None)
        if owner is None or getattr(owner, "round16_selective_plan", None) is not plan:
            raise ValueError("exact device runtime runner/Q32 owner differs")
        _require_callback(
            tensor_binding_factory,
            EXACT_WORKER_TENSOR_BINDING_FACTORY_API,
            "current tensor binding factory",
        )
        _require_callback(
            forward_identity_provider,
            EXACT_WORKER_FORWARD_IDENTITY_API,
            "forward identity provider",
        )
        _require_callback(kt_submit, EXACT_WORKER_KT_SUBMIT_API, "KT submit")
        _require_callback(kt_sync, EXACT_WORKER_KT_SYNC_API, "KT sync")
        _require_callback(
            kt_abort_drain, EXACT_WORKER_KT_ABORT_API, "KT abort/drain"
        )
        _require_callback(
            merge_weighted_partials,
            EXACT_WORKER_TENSOR_MERGE_API,
            "tensor weighted merge",
        )
        self.layer_name = layer_name
        self.runner = runner
        self.routed_owner = owner
        self.plan = plan
        self.source_token = source_token
        self.source_preflight_token = source_preflight_token
        self._binding_factory = tensor_binding_factory
        self._identity_provider = forward_identity_provider
        self._kt_submit = kt_submit
        self._kt_sync = kt_sync
        self._kt_abort_drain = kt_abort_drain
        self._merge = merge_weighted_partials
        self._lock = Lock()
        self._active: tuple[AllGatherQ32DeviceTensorAdapter, NativeMoeCommPrepared] | None = None
        self._epoch = 0

    def prepare(
        self, hidden_states: Any, router_logits: Any, layer_name: str
    ) -> NativeMoeCommPrepared:
        if layer_name != self.layer_name:
            raise ValueError("exact runtime prepared layer identity differs")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("exact layer runtime already has an active forward")
        binding = self._binding_factory(
            layer_name=self.layer_name,
            runner=self.runner,
            routed_owner=self.routed_owner,
            plan=self.plan,
            source_token=self.source_token,
            source_preflight_token=self.source_preflight_token,
        )
        if not isinstance(binding, AllGatherQ32TensorBinding):
            raise TypeError("current tensor binding factory returned wrong type")
        if binding.layer.layer_name != self.layer_name:
            raise ValueError("current tensor binding layer identity differs")
        if binding.layer.routed_owner is not self.routed_owner:
            raise ValueError("current tensor binding owner identity differs")
        if binding.layer.plan is not self.plan:
            raise ValueError("current tensor binding plan identity differs")
        if binding.layer.source_preflight_token != self.source_preflight_token:
            raise ValueError("current tensor binding source token differs")
        adapter = AllGatherQ32DeviceTensorAdapter(binding)
        prepared = adapter.prepare(hidden_states, router_logits, layer_name)
        with self._lock:
            if self._active is not None:
                adapter.cleanup(prepared)
                raise RuntimeError("exact layer runtime prepare raced")
            self._active = (adapter, prepared)
        return prepared

    def _take_active(
        self, prepared: NativeMoeCommPrepared
    ) -> AllGatherQ32DeviceTensorAdapter:
        with self._lock:
            active = self._active
        if active is None or active[1] is not prepared:
            raise RuntimeError("exact layer prepared state is stale")
        return active[0]

    def _retire(self, prepared: NativeMoeCommPrepared) -> None:
        with self._lock:
            if self._active is not None and self._active[1] is prepared:
                self._active = None

    def abort_prepared(self, prepared: NativeMoeCommPrepared) -> None:
        adapter = self._take_active(prepared)
        try:
            adapter.cleanup(prepared)
        finally:
            self._retire(prepared)

    def execute_selected(
        self,
        prepared: NativeMoeCommPrepared,
        selected: DeviceSelectedTopK,
        hidden_states: Any,
        layer_name: str,
    ) -> Any:
        if layer_name != self.layer_name:
            raise ValueError("exact execution layer identity differs")
        adapter = self._take_active(prepared)
        identity = self._identity_provider(self.layer_name)
        if type(identity) is not tuple or not identity:
            raise ValueError("forward identity provider must return a non-empty tuple")
        with self._lock:
            self._epoch += 1
            epoch = self._epoch
        identity_key = (*identity, self.layer_name, "forward_epoch", epoch)
        stream = adapter.layer.stream_identity_provider()
        pending: Any = None
        native: Any = None
        try:
            pending = self._kt_submit(
                hidden_states=hidden_states,
                topk_ids=selected.topk_ids,
                topk_weights=selected.topk_weights,
                stream_identity=stream,
                identity_key=identity_key,
            )
            native = adapter.execute(
                prepared, selected, hidden_states, identity_key
            )
            kt = self._kt_sync(
                pending=pending,
                reference_tensor=native.output,
                stream_identity=stream,
                identity_key=identity_key,
            )
            pending = None
            combined = self._merge(native=native, kt=kt)
            if getattr(combined, "route_weights_applied", None) is not True:
                raise ValueError("combined routed partial lacks Top-K weights")
            if getattr(combined, "routed_scaling_applied", None) is not False:
                raise ValueError("combined routed partial already contains scaling")
            if not hasattr(combined, "output"):
                raise TypeError("combined routed partial lacks tensor output")
            return combined.output
        except BaseException as primary:
            if pending is not None:
                try:
                    self._kt_abort_drain(
                        pending=pending,
                        reference_tensor=(
                            None if native is None else native.output
                        ),
                        stream_identity=stream,
                        identity_key=identity_key,
                    )
                except BaseException as cleanup_error:
                    primary.add_note(
                        "KT abort/drain also failed: " + repr(cleanup_error)
                    )
            # adapter.execute retires its own prepared state even on failure.
            # If execution never reached it, clean the active state here.
            try:
                adapter.cleanup(prepared)
            except RuntimeError:
                pass
            raise
        finally:
            self._retire(prepared)


def resolve_exact_qwen_conditional_experts(
    loaded_model: Any,
    *,
    expected_layer_count: int,
) -> tuple[tuple[str, Any], ...]:
    """Resolve the exact ConditionalGeneration language-model layer path."""

    try:
        layers = loaded_model.language_model.model.layers
    except AttributeError as error:
        raise RuntimeError(
            "target ConditionalGeneration model lacks language_model.model.layers"
        ) from error
    try:
        count = len(layers)
    except TypeError as error:
        raise RuntimeError("target Qwen layer container has no bounded length") from error
    if count != expected_layer_count:
        raise RuntimeError(
            f"target Qwen expected {expected_layer_count} layers but observed {count}"
        )
    resolved = []
    for index in range(count):
        try:
            runner = layers[index].mlp.experts
        except (AttributeError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"target Qwen layer {index} lacks mlp.experts runner"
            ) from error
        layer_name = getattr(runner, "layer_name", None)
        expected_suffix = f"language_model.model.layers.{index}.mlp.experts"
        if type(layer_name) is not str or not layer_name.endswith(expected_suffix):
            raise RuntimeError(
                f"target Qwen layer {index} runner path identity differs"
            )
        resolved.append((layer_name, runner))
    return tuple(resolved)


def _target_architecture(model_runner: Any) -> str | None:
    config = getattr(getattr(model_runner, "model_config", None), "hf_config", None)
    architectures = getattr(config, "architectures", None)
    if not isinstance(architectures, (list, tuple)):
        return None
    frozen = tuple(architectures)
    if any(type(item) is not str for item in frozen):
        return None
    return TARGET_ARCHITECTURE if frozen == (TARGET_ARCHITECTURE,) else None


_PATCH_LOCK = RLock()


@dataclass
class ExactWorkerHookHandle:
    installed: bool
    model_runner_cls: type
    original_load_model: Callable[..., Any]
    wrapper_load_model: Callable[..., Any] | None = None

    def uninstall(self) -> None:
        if not self.installed:
            return
        with _PATCH_LOCK:
            if self.wrapper_load_model is None or (
                self.model_runner_cls.load_model is not self.wrapper_load_model
            ):
                raise RuntimeError("exact worker load hook identity changed")
            self.model_runner_cls.load_model = self.original_load_model
            self.installed = False


def install_exact_worker_load_hook(
    *,
    model_runner_cls: type,
    source_token: ExactWorkerSourceToken,
    installer: ModelScopedSelectivePreallocationInstaller,
    plan: SelectiveLoadPlan,
    runtime_registry: ExactWorkerRuntimeRegistry,
    runtime_factory: Callable[..., Any],
    enabled: bool = False,
) -> ExactWorkerHookHandle:
    """Patch only the frozen ``NPUModelRunner.load_model`` revision.

    ``enabled`` defaults false.  Non-target architectures call the untouched
    original path.  For the target, the Q32 construction scope begins before
    ``get_model`` and post-load runtime registration commits only after all
    48 owner/runner pairs and runtime objects validate.
    """

    if not isinstance(model_runner_cls, type):
        raise TypeError("model_runner_cls must be a class")
    original = getattr(model_runner_cls, "load_model", None)
    if not callable(original):
        raise TypeError("NPUModelRunner.load_model is unavailable")
    handle = ExactWorkerHookHandle(False, model_runner_cls, original)
    if enabled is not True:
        if enabled is not False:
            raise ValueError("exact worker hook enabled flag must be boolean")
        return handle
    if not validate_exact_worker_source_token(source_token):
        raise ValueError("exact worker source token is required before patching")
    if (
        model_runner_cls.__module__ != EXACT_MODEL_RUNNER_MODULE
        or model_runner_cls.__name__ != EXACT_MODEL_RUNNER_CLASS
    ):
        raise ValueError("model runner class identity differs from frozen donor")
    signature = inspect.signature(original)
    if tuple(signature.parameters) != ("self",):
        raise ValueError("frozen NPUModelRunner.load_model signature changed")
    if not isinstance(installer, ModelScopedSelectivePreallocationInstaller):
        raise TypeError("exact worker hook requires the scoped installer")
    if installer.source_bound_runner_pairing_verified is not True:
        raise ValueError("Q32 owner and Hybrid runner must be paired")
    if not isinstance(plan, SelectiveLoadPlan):
        raise TypeError("exact worker hook plan is invalid")
    if not isinstance(runtime_registry, ExactWorkerRuntimeRegistry):
        raise TypeError("exact worker runtime registry is invalid")
    if runtime_registry.plan is not plan:
        raise ValueError("installer plan and runtime registry plan differ")
    if runtime_registry.source_token != source_token:
        raise ValueError("runtime registry source token differs")
    _require_callback(
        runtime_factory,
        EXACT_LAYER_RUNTIME_FACTORY_API,
        "exact layer runtime factory",
    )

    def wrapped(self: Any) -> Any:
        architecture = _target_architecture(self)
        if architecture is None:
            return original(self)
        if runtime_registry.layer_count:
            raise RuntimeError("exact worker hook cannot load a second target model")
        try:
            with installer.activate(
                plan=plan,
                architecture=architecture,
                model_class=object,
            ) as acknowledge:
                result = original(self)
                loaded_model = self.get_model()
                acknowledge(loaded_model)
            resolved = resolve_exact_qwen_conditional_experts(
                loaded_model,
                expected_layer_count=runtime_registry.expected_layer_count,
            )
            registrations = []
            for layer_name, runner in resolved:
                runtime = runtime_factory(
                    layer_name=layer_name,
                    runner=runner,
                    source_token=source_token,
                    plan=plan,
                )
                registrations.append((layer_name, runner, runtime))
            runtime_registry.register_prepared(tuple(registrations))
            self.round16_b_runtime_registry = runtime_registry
            return result
        except BaseException:
            runtime_registry.clear()
            if getattr(self, "round16_b_runtime_registry", None) is runtime_registry:
                delattr(self, "round16_b_runtime_registry")
            raise

    wrapped.__name__ = original.__name__
    wrapped.__qualname__ = original.__qualname__
    wrapped.__module__ = original.__module__
    wrapped.round16_api_identity = EXACT_DONOR_WORKER_HOOK_API

    with _PATCH_LOCK:
        if model_runner_cls.load_model is not original:
            raise RuntimeError("NPUModelRunner.load_model changed before patch")
        model_runner_cls.load_model = wrapped
    handle.installed = True
    handle.wrapper_load_model = wrapped
    return handle


__all__ = (
    "EXACT_DONOR_WORKER_HOOK_API",
    "EXACT_LAYER_RUNTIME_API",
    "EXACT_LAYER_RUNTIME_FACTORY_API",
    "EXACT_UPSTREAM_MOE_RUNNER_API",
    "EXACT_WORKER_FORWARD_IDENTITY_API",
    "EXACT_WORKER_EXPERT_SELECTOR_API",
    "EXACT_WORKER_KT_ABORT_API",
    "EXACT_WORKER_KT_SUBMIT_API",
    "EXACT_WORKER_KT_SYNC_API",
    "EXACT_WORKER_SOURCE_SHA256",
    "EXACT_WORKER_TENSOR_BINDING_FACTORY_API",
    "EXACT_WORKER_TENSOR_MERGE_API",
    "ExactDeviceLayerRuntime",
    "ExactWorkerHookHandle",
    "ExactWorkerRuntimeRegistry",
    "ExactWorkerSourceToken",
    "build_exact_worker_runner_class",
    "install_exact_worker_load_hook",
    "load_exact_worker_source_receipt",
    "resolve_exact_qwen_conditional_experts",
    "validate_exact_worker_source_identity",
    "validate_exact_worker_source_token",
)
