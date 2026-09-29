"""Integration-owned constructor/worker plugin for the exact donor revision.

Importing this module has no side effect and imports no device runtime.  The
host image supplies already-imported exact classes and callbacks after the B2
source preflight.  ``enabled`` defaults false.  A device Q32 RoutedExperts
class must carry the reviewed constructor identity; without it installation
stops before ``NPUModelRunner.load_model`` is patched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from round16.b.allgather_q32_device_adapter import (
    ALLGATHER_BUILD_FUSED_INPUT_API,
    ALLGATHER_STREAM_IDENTITY_API,
    AllGatherQ32LayerBinding,
    AllGatherQ32TensorBinding,
)
from round16.b.exact_donor_worker_hook import (
    EXACT_LAYER_RUNTIME_FACTORY_API,
    EXACT_UPSTREAM_MOE_RUNNER_API,
    EXACT_WORKER_FORWARD_IDENTITY_API,
    EXACT_WORKER_TENSOR_BINDING_FACTORY_API,
    EXACT_WORKER_TENSOR_MERGE_API,
    ExactDeviceLayerRuntime,
    ExactWorkerHookHandle,
    ExactWorkerRuntimeRegistry,
    ExactWorkerSourceToken,
    build_exact_worker_runner_class,
    install_exact_worker_load_hook,
    validate_exact_worker_source_token,
)
from round16.b.native_source_preflight import SourcePreflightToken
from round16.b.selective_loader import SelectiveLoadPlan
from round16.b.selective_preallocation_installer import (
    ASCEND_FUSED_MOE_FORWARDING_API,
    ModelScopedSelectivePreallocationInstaller,
    ROUTED_EXPERTS_Q32_API,
    SOURCE_BOUND_HYBRID_RUNNER_API,
)
from round16_integration.p1_allgather_q32_probe import (
    DevicePhysicalTopKPayloadBuilder,
    TorchNpuTensorOps,
)


B2_WORKER_PLUGIN_API = "round16-integration:b2-exact-worker-plugin:v1"


@dataclass(frozen=True)
class B2ExactWorkerPluginBindings:
    model_runner_cls: type
    qwen_module: Any
    ascend_fused_moe: Callable[..., Any]
    routed_experts_cls: type
    upstream_moe_runner_cls: type
    shared_order_no_overlap: Any
    shared_order_overlap: Any
    plan: SelectiveLoadPlan
    worker_source_token: ExactWorkerSourceToken
    runtime_factory: Callable[..., ExactDeviceLayerRuntime]
    expert_selector: Callable[..., tuple[Any, Any]] | None = None
    expected_layer_count: int = 48


def make_exact_device_runtime_factory(
    *,
    source_preflight_token: SourcePreflightToken,
    tensor_binding_factory: Callable[..., AllGatherQ32TensorBinding],
    forward_identity_provider: Callable[[str], tuple[Any, ...]],
    kt_submit: Callable[..., Any],
    kt_sync: Callable[..., Any],
    kt_abort_drain: Callable[..., None],
    merge_weighted_partials: Callable[..., Any],
) -> Callable[..., ExactDeviceLayerRuntime]:
    """Create the post-load factory without capturing a forward context."""

    def factory(
        *,
        layer_name: str,
        runner: Any,
        source_token: ExactWorkerSourceToken,
        plan: SelectiveLoadPlan,
    ) -> ExactDeviceLayerRuntime:
        return ExactDeviceLayerRuntime(
            layer_name=layer_name,
            runner=runner,
            plan=plan,
            source_token=source_token,
            source_preflight_token=source_preflight_token,
            tensor_binding_factory=tensor_binding_factory,
            forward_identity_provider=forward_identity_provider,
            kt_submit=kt_submit,
            kt_sync=kt_sync,
            kt_abort_drain=kt_abort_drain,
            merge_weighted_partials=merge_weighted_partials,
        )

    factory.round16_api_identity = EXACT_LAYER_RUNTIME_FACTORY_API
    return factory


def make_current_allgather_tensor_binding_factory(
    *,
    get_forward_context: Callable[[], Any],
    current_stream: Callable[[], Any],
    build_fused_experts_input: Callable[..., Any],
    quant_none: Any,
    tensor_ops: TorchNpuTensorOps,
) -> Callable[..., AllGatherQ32TensorBinding]:
    """Bind current comm/stream lazily inside an active donor forward.

    ``get_forward_context`` is the exact vLLM function.  The Ascend context
    manager has already assigned ``moe_comm_method`` before Qwen calls
    ``self.experts``.  No comm or stream object is retained across forwards.
    """

    if not callable(get_forward_context) or not callable(current_stream):
        raise TypeError("current forward context and stream providers are required")
    if not callable(build_fused_experts_input):
        raise TypeError("exact fused input builder is required")
    def fused_input_builder(**kwargs: Any) -> Any:
        return build_fused_experts_input(**kwargs)

    fused_input_builder.round16_api_identity = ALLGATHER_BUILD_FUSED_INPUT_API

    def factory(
        *,
        layer_name: str,
        runner: Any,
        routed_owner: Any,
        plan: SelectiveLoadPlan,
        source_token: ExactWorkerSourceToken,
        source_preflight_token: SourcePreflightToken,
    ) -> AllGatherQ32TensorBinding:
        del runner
        if not validate_exact_worker_source_token(source_token):
            raise ValueError("exact worker source token changed at forward")
        context = get_forward_context()
        comm_method = getattr(context, "moe_comm_method", None)
        if comm_method is None:
            raise RuntimeError("current Ascend forward context lacks moe_comm_method")

        def comm_provider() -> Any:
            return getattr(get_forward_context(), "moe_comm_method", None)

        comm_provider.round16_api_identity = (
            "round16-b:current-moe-comm-object-provider:source-contract-v1"
        )

        def stream_provider() -> Any:
            return current_stream()

        stream_provider.round16_api_identity = ALLGATHER_STREAM_IDENTITY_API
        layer_binding = AllGatherQ32LayerBinding(
            layer_name=layer_name,
            plan=plan,
            routed_owner=routed_owner,
            comm_method=comm_method,
            comm_method_provider=comm_provider,
            stream_identity_provider=stream_provider,
            build_fused_experts_input=fused_input_builder,
            source_preflight_token=source_preflight_token,
            quant_none=quant_none,
        )
        return AllGatherQ32TensorBinding(
            layer_binding,
            DevicePhysicalTopKPayloadBuilder(plan, tensor_ops),
        )

    factory.round16_api_identity = EXACT_WORKER_TENSOR_BINDING_FACTORY_API
    return factory


def install_b2_exact_worker_plugin(
    bindings: B2ExactWorkerPluginBindings,
    *,
    enabled: bool = False,
) -> tuple[ExactWorkerHookHandle, ExactWorkerRuntimeRegistry | None]:
    """Install the exact hook or return a no-op handle when default OFF."""

    if not isinstance(bindings, B2ExactWorkerPluginBindings):
        raise TypeError("B2 exact worker plugin bindings are invalid")
    original = getattr(bindings.model_runner_cls, "load_model", None)
    if not callable(original):
        raise TypeError("exact model runner load_model is unavailable")
    if enabled is not True:
        if enabled is not False:
            raise ValueError("B2 worker plugin enabled flag must be boolean")
        return (
            ExactWorkerHookHandle(False, bindings.model_runner_cls, original),
            None,
        )
    if not validate_exact_worker_source_token(bindings.worker_source_token):
        raise ValueError("exact worker source preflight is required")
    if (
        not isinstance(bindings.routed_experts_cls, type)
        or getattr(
            bindings.routed_experts_cls, "round16_q32_api_identity", None
        )
        != ROUTED_EXPERTS_Q32_API
    ):
        raise RuntimeError(
            "SELECTIVE_Q32_PREALLOCATION_REQUIRED: verified physical-Q32 "
            "RoutedExperts class is unavailable"
        )
    if (
        getattr(bindings.runtime_factory, "round16_api_identity", None)
        != EXACT_LAYER_RUNTIME_FACTORY_API
    ):
        raise ValueError("exact layer runtime factory identity mismatch")
    runner_cls = build_exact_worker_runner_class(
        bindings.upstream_moe_runner_cls,
        upstream_api_identity=EXACT_UPSTREAM_MOE_RUNNER_API,
        shared_order_no_overlap=bindings.shared_order_no_overlap,
        shared_order_overlap=bindings.shared_order_overlap,
        expert_selector=bindings.expert_selector,
    )
    registry = ExactWorkerRuntimeRegistry(
        source_token=bindings.worker_source_token,
        plan=bindings.plan,
        expected_layer_count=bindings.expected_layer_count,
    )
    installer = ModelScopedSelectivePreallocationInstaller(
        qwen_module=bindings.qwen_module,
        ascend_fused_moe=bindings.ascend_fused_moe,
        ascend_api_identity=ASCEND_FUSED_MOE_FORWARDING_API,
        routed_experts_cls=bindings.routed_experts_cls,
        routed_experts_api_identity=ROUTED_EXPERTS_Q32_API,
        hybrid_runner_cls=runner_cls,
        hybrid_runner_api_identity=SOURCE_BOUND_HYBRID_RUNNER_API,
        hybrid_runner_args={"round16_runtime_registry": registry},
        expected_layer_count=bindings.expected_layer_count,
    )
    handle = install_exact_worker_load_hook(
        model_runner_cls=bindings.model_runner_cls,
        source_token=bindings.worker_source_token,
        installer=installer,
        plan=bindings.plan,
        runtime_registry=registry,
        runtime_factory=bindings.runtime_factory,
        enabled=True,
    )
    return handle, registry


__all__ = (
    "B2ExactWorkerPluginBindings",
    "B2_WORKER_PLUGIN_API",
    "install_b2_exact_worker_plugin",
    "make_current_allgather_tensor_binding_factory",
    "make_exact_device_runtime_factory",
)
