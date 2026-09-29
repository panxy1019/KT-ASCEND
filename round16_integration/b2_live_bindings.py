"""Exact-image Round16-B bindings for the first full TP1 eager model.

The module is import-safe on the jumper: device/framework imports occur only
inside :func:`build_live_bindings`, which is invoked by the donor worker after
its device and distributed state are initialized.  The binding preserves the
logical 512-expert router while the device owner allocates physical Q32.
"""

from __future__ import annotations

from copy import copy
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
from threading import Lock
from typing import Any, Callable

from round16.b.exact_donor_worker_hook import (
    EXACT_WORKER_EXPERT_SELECTOR_API,
    EXACT_WORKER_FORWARD_IDENTITY_API,
    EXACT_WORKER_KT_ABORT_API,
    EXACT_WORKER_KT_SUBMIT_API,
    EXACT_WORKER_KT_SYNC_API,
    EXACT_WORKER_TENSOR_MERGE_API,
)
from round16.b.selective_loader import SelectiveLoadPlan, plan_q32
from round16.b.selective_preallocation_installer import ROUTED_EXPERTS_Q32_API
from round16.b.source_bound_q32_adapter import (
    PreparedQ32State,
    Q32_ALLOCATION_ROLLBACK_API,
    Q32_PHYSICAL_CONFIG_PREPARER_API,
    build_source_bound_q32_routed_experts_class,
)
from round16_integration.b2_exact_worker_plugin import (
    B2ExactWorkerPluginBindings,
    make_current_allgather_tensor_binding_factory,
    make_exact_device_runtime_factory,
)
from round16_integration.b2_worker_overlay_entry import B2_BINDINGS_FACTORY_API
from round16_integration.p1_allgather_q32_probe import (
    CombinedDeviceWeightedPartial,
    KTDeviceWeightedPartial,
    TensorWeightedAggregateMerger,
    TorchNpuTensorOps,
    load_snapshot_manifest,
    load_source_preflight_token,
)


B2_LIVE_BINDINGS_API = "round16-integration:b2-live-bindings:v1"
HOST_PLE_BRIDGE_API = "round16-integration:host-ple-bridge:v1"
HOST_PLE_POSTLOAD_API = "round16-integration:host-ple-postload:v1"
ASCEND_QSA_CACHE_SHAPE_API = "round16-integration:ascend-qsa-cache-shape:v1"
ASCEND_QSA_CACHE_RESHAPE_API = "round16-integration:ascend-qsa-cache-reshape:v1"
ASCEND_QSA_CACHE_SPLIT_API = "round16-integration:ascend-qsa-cache-split:v1"
ASCEND_QSA_MULTIHEAD_STORE_API = (
    "round16-integration:ascend-qsa-multihead-store:v1"
)
_HOST_PLE_MARKER = "round16_host_ple_subtree"
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")


def install_ascend_qsa_cache_shape(*, qsa_backend_cls: type) -> None:
    """Bind QSA cache allocation to the Ascend runner's split K/V contract.

    The frozen QSA backend inherits the CUDA packed shape
    ``[blocks, heads, block, 2 * width]``.  The Ascend model runner expects a
    leading K/V axis, drops that axis when it materializes each cache, and
    therefore gave QSA two three-dimensional tensors.  QSA's store primitive
    consumes separate ``[blocks, block, heads, width]`` tensors.  Returning
    the native Ascend five-dimensional allocation shape preserves the same
    payload while making that split explicit.
    """

    installed = getattr(qsa_backend_cls, "round16_cache_shape_api", None)
    if installed not in (None, ASCEND_QSA_CACHE_SHAPE_API):
        raise RuntimeError("another Round16 QSA cache-shape binding is installed")
    if installed == ASCEND_QSA_CACHE_SHAPE_API:
        return

    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, int, int, int, int]:
        del cache_dtype_str
        dimensions = (num_blocks, block_size, num_kv_heads, head_size)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("QSA cache dimensions must be positive integers")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    get_kv_cache_shape.round16_api_identity = ASCEND_QSA_CACHE_SHAPE_API
    qsa_backend_cls.get_kv_cache_shape = staticmethod(get_kv_cache_shape)
    qsa_backend_cls.round16_cache_shape_api = ASCEND_QSA_CACHE_SHAPE_API


def install_ascend_qsa_cache_reshape(
    *, model_runner_cls: type, qsa_impl_cls: type
) -> None:
    """Restore the omitted singleton block axis for QSA cache views.

    The frozen Ascend allocator unconditionally removes the first dimension
    returned by a backend for ordinary ``AttentionSpec`` caches.  When the
    cache group retained the upstream QSA backend, its four-dimensional shape
    had no leading K/V axis; with the qualified context range the allocation
    has one block, so the resulting K and V views became three-dimensional.
    Correct those views before vLLM binds them to the attention layer.  The
    operation is a view-only ``unsqueeze`` and is restricted to the concrete
    Ascend QSA implementation.
    """

    installed = getattr(model_runner_cls, "round16_qsa_cache_reshape_api", None)
    if installed not in (None, ASCEND_QSA_CACHE_RESHAPE_API):
        raise RuntimeError("another Round16 QSA cache reshape binding is installed")
    if installed == ASCEND_QSA_CACHE_RESHAPE_API:
        return
    original = getattr(model_runner_cls, "_reshape_kv_cache_tensors", None)
    if not callable(original):
        raise TypeError("Ascend model runner lacks _reshape_kv_cache_tensors")

    def reshape_kv_cache_tensors(self: Any, *args: Any, **kwargs: Any) -> Any:
        caches = original(self, *args, **kwargs)
        context = self.compilation_config.static_forward_context
        for layer_name, layer in context.items():
            if not isinstance(getattr(layer, "impl", None), qsa_impl_cls):
                continue
            cache = caches.get(layer_name)
            if not isinstance(cache, tuple) or len(cache) != 2:
                raise RuntimeError(f"{layer_name} QSA cache must contain K and V")
            ranks = tuple(int(value.ndim) for value in cache)
            if ranks == (3, 3):
                num_heads = int(layer.num_kv_heads)
                head_dim = int(layer.head_dim)
                for value in cache:
                    if tuple(value.shape[-2:]) != (num_heads, head_dim):
                        raise RuntimeError(
                            f"{layer_name} QSA cache has an unexpected physical shape"
                        )
                cache = tuple(value.unsqueeze(0) for value in cache)
                caches[layer_name] = cache
                ranks = (4, 4)
            if ranks != (4, 4):
                raise RuntimeError(f"{layer_name} QSA cache rank is unsupported: {ranks}")
            for value in cache:
                if int(value.shape[2]) != int(layer.num_kv_heads):
                    raise RuntimeError(f"{layer_name} QSA cache head count changed")
                if int(value.shape[3]) != int(layer.head_dim):
                    raise RuntimeError(f"{layer_name} QSA cache width changed")
        return caches

    reshape_kv_cache_tensors.round16_api_identity = ASCEND_QSA_CACHE_RESHAPE_API
    model_runner_cls._reshape_kv_cache_tensors = reshape_kv_cache_tensors
    model_runner_cls.round16_qsa_cache_reshape_api = ASCEND_QSA_CACHE_RESHAPE_API


def install_ascend_qsa_cache_split(*, qsa_module: Any) -> None:
    """Canonicalize QSA K/V cache views at their shared consumer boundary.

    The frozen allocator can bind a single page group as
    ``[page_size, 1, width]``.  Both cache updates and sparse attention pass
    through ``_split_qsa_kv_cache``; restore only that omitted singleton page
    axis and require every returned K/V view to match the QSA
    ``[pages, page_size, heads, width]`` contract.  TP1 retains the model's
    two KV heads; only the cache-store primitive is singleton-head and is
    adapted separately.
    """

    installed = getattr(qsa_module, "round16_cache_split_api", None)
    if installed not in (None, ASCEND_QSA_CACHE_SPLIT_API):
        raise RuntimeError("another Round16 QSA cache split binding is installed")
    if installed == ASCEND_QSA_CACHE_SPLIT_API:
        return
    original = getattr(qsa_module, "_split_qsa_kv_cache", None)
    if not callable(original):
        raise TypeError("Ascend QSA module lacks _split_qsa_kv_cache")

    def split_qsa_kv_cache(kv_cache: Any, head_size: int) -> tuple[Any, Any]:
        views = original(kv_cache, head_size)
        if not isinstance(views, tuple) or len(views) != 2:
            raise RuntimeError("QSA cache split must return K and V views")
        normalized = []
        for value in views:
            ndim = int(value.ndim)
            if ndim == 3:
                shape = tuple(int(size) for size in value.shape)
                if shape[-1] != head_size:
                    raise RuntimeError(
                        f"QSA cache omitted-axis width is invalid: {shape}"
                    )
                if shape[1] == 1:
                    # NHD without the singleton page axis.
                    value = value.unsqueeze(0)
                elif shape[0] == 1:
                    # HND without the singleton page axis.
                    value = value.unsqueeze(0).permute(0, 2, 1, 3)
                else:
                    raise RuntimeError(
                        f"QSA cache omitted-axis view has invalid shape: {shape}"
                    )
                ndim = 4
            if ndim != 4:
                raise RuntimeError(f"QSA cache view rank is unsupported: {ndim}")
            shape = tuple(int(size) for size in value.shape)
            if shape[2] != 1 and shape[1] == 1 and shape[3] == head_size:
                # FlashAttention HND view -> QSA NHD view; this is a stride-only
                # permutation and retains the same cache storage.
                value = value.permute(0, 2, 1, 3)
                shape = tuple(int(size) for size in value.shape)
            if shape[2] <= 0 or shape[3] != head_size:
                raise RuntimeError(
                    f"QSA cache view violates the qualified layout: {shape}"
                )
            normalized.append(value)
        return normalized[0], normalized[1]

    split_qsa_kv_cache.round16_api_identity = ASCEND_QSA_CACHE_SPLIT_API
    qsa_module._split_qsa_kv_cache = split_qsa_kv_cache
    qsa_module.round16_cache_split_api = ASCEND_QSA_CACHE_SPLIT_API


def install_ascend_qsa_multihead_store(
    *, qsa_impl_cls: type, qsa_module: Any
) -> None:
    """Store every TP1 QSA KV head through the qualified singleton kernel.

    The frozen Ascend sparse-attention kernel accepts grouped-query caches
    with multiple KV heads, but its cache-store wrapper requires one head per
    invocation.  Preserve all model heads by presenting a view of one head at
    a time.  No cache or row data is copied and the original non-tuple cache
    path remains untouched.
    """

    installed = getattr(qsa_impl_cls, "round16_multihead_store_api", None)
    if installed not in (None, ASCEND_QSA_MULTIHEAD_STORE_API):
        raise RuntimeError("another Round16 QSA multi-head store is installed")
    if installed == ASCEND_QSA_MULTIHEAD_STORE_API:
        return
    original = getattr(qsa_impl_cls, "do_kv_cache_update", None)
    split_cache = getattr(qsa_module, "_split_qsa_kv_cache", None)
    store_rows = getattr(qsa_module, "qsa_store_cache_rows", None)
    if not callable(original):
        raise TypeError("Ascend QSA implementation lacks do_kv_cache_update")
    if not callable(split_cache) or not callable(store_rows):
        raise TypeError("Ascend QSA module lacks cache split/store primitives")

    def do_kv_cache_update(
        self: Any,
        layer: Any,
        key: Any,
        value: Any,
        kv_cache: Any,
        slot_mapping: Any,
    ) -> None:
        if not isinstance(kv_cache, tuple):
            return original(self, layer, key, value, kv_cache, slot_mapping)

        key_cache, value_cache = qsa_module._split_qsa_kv_cache(
            kv_cache, int(self.head_size)
        )
        cache_shape = tuple(int(size) for size in key_cache.shape)
        value_cache_shape = tuple(int(size) for size in value_cache.shape)
        key_shape = tuple(int(size) for size in key.shape)
        value_shape = tuple(int(size) for size in value.shape)
        expected_heads = int(self.num_kv_heads)
        head_size = int(self.head_size)

        if (
            len(cache_shape) != 4
            or value_cache_shape != cache_shape
            or cache_shape[2] != expected_heads
            or cache_shape[3] != head_size
        ):
            raise RuntimeError(
                "QSA K/V cache does not match the model KV-head contract: "
                f"K={cache_shape} V={value_cache_shape} "
                f"expected_heads={expected_heads} head_size={head_size}"
            )
        expected_row_tail = (expected_heads, head_size)
        if (
            len(key_shape) != 3
            or len(value_shape) != 3
            or key_shape[1:] != expected_row_tail
            or value_shape[1:] != expected_row_tail
            or key_shape[0] != value_shape[0]
        ):
            raise RuntimeError(
                "QSA K/V rows do not match the model KV-head contract: "
                f"K={key_shape} V={value_shape} expected_tail={expected_row_tail}"
            )

        for head in range(expected_heads):
            cache_slice = (slice(None), slice(None), slice(head, head + 1), slice(None))
            row_slice = (slice(None), slice(head, head + 1), slice(None))
            qsa_module.qsa_store_cache_rows(
                key_cache[cache_slice], slot_mapping, key[row_slice]
            )
            qsa_module.qsa_store_cache_rows(
                value_cache[cache_slice], slot_mapping, value[row_slice]
            )

    do_kv_cache_update.round16_api_identity = ASCEND_QSA_MULTIHEAD_STORE_API
    qsa_impl_cls.do_kv_cache_update = do_kv_cache_update
    qsa_impl_cls.round16_multihead_store_api = ASCEND_QSA_MULTIHEAD_STORE_API


def install_host_ple_bridge(
    *,
    torch_module: Any,
    ple_offload_cls: type,
    ple_embedding_cls: type | None = None,
    postload_patch_module: Any | None = None,
) -> None:
    """Keep the single 95 GiB PLE embedding on Host for the B worker.

    The frozen Ascend image deliberately disables its CUDA-IPC PLE offloader.
    A TP1 device-resident construction therefore attempts one 95.37 GiB NPU
    allocation before any routed-expert layer is built.  This model-scoped
    bridge retains vLLM ownership of PLE history/state, creates only the
    embedding subtree on CPU, and copies the small lookup result back to the
    caller device before the existing projection/conv path runs.
    """

    installed = getattr(ple_offload_cls, "round16_host_ple_api", None)
    if installed not in (None, HOST_PLE_BRIDGE_API):
        raise RuntimeError("another Round16 Host PLE bridge is already installed")
    if installed is None:
        if not callable(getattr(ple_offload_cls, "forward", None)):
            raise TypeError("PLE offload class lacks a forward method")

        @classmethod
        def get_target_device(cls: type) -> Any:
            del cls
            return torch_module.device("cpu")

        def to_host(value: Any) -> Any:
            if not callable(getattr(value, "to", None)):
                return value
            detach = getattr(value, "detach", None)
            if callable(detach):
                value = detach()
            return value.to(device="cpu", non_blocking=False)

        def forward(
            self: Any,
            hidden_states: Any,
            input_ids: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            output_buffer = kwargs.get("output_buffer")
            if output_buffer is not None:
                raise RuntimeError(
                    "Round16 Host PLE does not accept an external device output buffer"
                )
            target_device = getattr(hidden_states, "device", None)
            if target_device is None:
                raise TypeError("PLE hidden states must expose a device")
            host_hidden = to_host(hidden_states)
            host_input_ids = to_host(input_ids)
            host_args = tuple(to_host(value) for value in args)
            host_kwargs = {
                name: to_host(value) for name, value in kwargs.items()
            }
            output = self.forward_impl(
                host_hidden,
                host_input_ids,
                *host_args,
                **host_kwargs,
            )
            if not callable(getattr(output, "to", None)):
                raise TypeError("PLE forward_impl must return a tensor-like result")
            return output.to(device=target_device, non_blocking=False)

        get_target_device.round16_api_identity = HOST_PLE_BRIDGE_API
        forward.round16_api_identity = HOST_PLE_BRIDGE_API
        ple_offload_cls.get_target_device = get_target_device
        ple_offload_cls.forward = forward
        ple_offload_cls.round16_host_ple_api = HOST_PLE_BRIDGE_API

    optional = (ple_embedding_cls, postload_patch_module)
    if any(value is not None for value in optional) and not all(
        value is not None for value in optional
    ):
        raise ValueError(
            "Host PLE post-load binding requires both embedding class and patch module"
        )
    if ple_embedding_cls is None:
        return

    original_init = ple_embedding_cls.__init__
    if getattr(original_init, "round16_api_identity", None) != HOST_PLE_POSTLOAD_API:
        def tagged_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            modules = getattr(self, "modules", None)
            if not callable(modules):
                raise TypeError("PLE embedding must expose its module subtree")
            for module in modules():
                setattr(module, _HOST_PLE_MARKER, True)

        tagged_init.round16_api_identity = HOST_PLE_POSTLOAD_API
        ple_embedding_cls.__init__ = tagged_init

    original_context = getattr(postload_patch_module, "device_loading_context", None)
    if not callable(original_context):
        raise TypeError("Ascend post-load patch lacks device_loading_context")
    if (
        getattr(original_context, "round16_api_identity", None)
        != HOST_PLE_POSTLOAD_API
    ):
        @contextmanager
        def host_aware_context(module: Any, target_device: Any) -> Any:
            selected_device = (
                torch_module.device("cpu")
                if getattr(module, _HOST_PLE_MARKER, False)
                else target_device
            )
            with original_context(module, selected_device) as loaded:
                yield loaded

        host_aware_context.round16_api_identity = HOST_PLE_POSTLOAD_API
        postload_patch_module.device_loading_context = host_aware_context


def make_ascend_expert_selector(
    ascend_select_experts: Callable[..., tuple[Any, Any]],
    *,
    enable_custom_op: Callable[[], bool] | None = None,
) -> Callable[..., tuple[Any, Any]]:
    """Bind the frozen Ascend Top-K implementation to the Hybrid runner.

    The generic vLLM router defaults to the CUDA ``_moe_C.topk_softmax`` op.
    The target Ascend image already supplies an equivalent device selector;
    this adapter forwards the original 512-expert routing contract to it and
    leaves physical Q32 remapping to the later Hybrid split.
    """

    if not callable(ascend_select_experts):
        raise TypeError("Ascend expert selector must be callable")
    if enable_custom_op is not None and not callable(enable_custom_op):
        raise TypeError("Ascend custom-op initializer must be callable")
    registration_lock = Lock()
    registration_complete = False

    def ensure_registered() -> None:
        nonlocal registration_complete
        if registration_complete or enable_custom_op is None:
            return
        with registration_lock:
            if registration_complete:
                return
            if enable_custom_op() is not True:
                raise RuntimeError("Ascend custom-op registration was unavailable")
            registration_complete = True

    def expert_selector(
        *,
        runner: Any,
        hidden_states: Any,
        router_logits: Any,
        input_ids: Any,
    ) -> tuple[Any, Any]:
        ensure_registered()
        owner = runner.routed_experts
        return ascend_select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=runner.moe_config.experts_per_token,
            use_grouped_topk=owner.use_grouped_topk,
            renormalize=owner.renormalize,
            topk_group=owner.topk_group,
            num_expert_group=owner.num_expert_group,
            custom_routing_function=owner.custom_routing_function,
            scoring_func=owner.scoring_func,
            routed_scaling_factor=owner.routed_scaling_factor,
            e_score_correction_bias=owner.e_score_correction_bias,
            indices_type=runner._quant_method.topk_indices_dtype,
            mix_placement=False,
            num_logical_experts=512,
            num_shared_experts=0,
            num_experts=512,
            input_ids=input_ids,
            tid2eid=None,
        )

    expert_selector.round16_api_identity = EXACT_WORKER_EXPERT_SELECTOR_API
    return expert_selector


class PhysicalQ32ExpertMapManager:
    """Physical owner map: logical 512 remains visible, only Q32 is local."""

    def __init__(
        self, torch_module: Any, logical_to_physical: tuple[int, ...]
    ) -> None:
        if len(logical_to_physical) != 512:
            raise ValueError("physical Q32 manager requires a 512-entry map")
        if sum(value >= 0 for value in logical_to_physical) != 32:
            raise ValueError("physical Q32 manager requires exactly 32 slots")
        self.global_num_experts = 512
        self.local_num_experts = 32
        self.num_fused_shared_experts = 0
        self._mapping = logical_to_physical
        self._placement_strategy = "linear"
        self._expert_map = torch_module.tensor(
            logical_to_physical, dtype=torch_module.int32
        )
        self._expert_mask = None
        self._routing_tables = None

    @property
    def placement_strategy(self) -> str:
        return self._placement_strategy

    @property
    def expert_map(self) -> Any:
        return self._expert_map

    @property
    def expert_mask(self) -> None:
        return self._expert_mask

    @property
    def routing_tables(self) -> None:
        return self._routing_tables

    def map_global_to_local(self, expert_id: int) -> int:
        if type(expert_id) is not int or not (0 <= expert_id < 512):
            raise ValueError("logical expert id must be in [0,512)")
        return self._mapping[expert_id]

    def is_local_expert(self, expert_id: int) -> bool:
        return self.map_global_to_local(expert_id) >= 0

    def get_local_expert_ids(self) -> list[int]:
        return [
            expert_id
            for expert_id, physical_id in enumerate(self._mapping)
            if physical_id >= 0
        ]

    def update(self, _moe_parallel_config: Any, global_num_experts: int) -> None:
        # Dynamic EPLB is outside the frozen TP1/Q32 contract.  Retaining an
        # update method prevents an accidental AttributeError from obscuring
        # the actual contract violation while still refusing a remap.
        if global_num_experts != 512:
            raise RuntimeError("physical Q32 owner cannot change logical capacity")
        raise RuntimeError("physical Q32 owner does not support dynamic EPLB")


def build_live_q32_owner_class(
    *,
    native_routed_experts_cls: type,
    torch_module: Any,
    ascend_method_cls: type,
    setup_moe_comm_method: Callable[[Any], Any],
    plan: SelectiveLoadPlan,
) -> type:
    """Build the V1-V4-qualified owner with the Ascend load/runtime lifecycle."""

    if not isinstance(plan, SelectiveLoadPlan):
        raise TypeError("live Q32 owner requires SelectiveLoadPlan")
    if plan.logical_expert_count != 512 or plan.physical_expert_count != 32:
        raise ValueError("live Q32 owner requires logical512/physical32")
    if not isinstance(ascend_method_cls, type):
        raise TypeError("Ascend BF16 method must be a class")
    if not callable(setup_moe_comm_method):
        raise TypeError("MoE communication setup must be callable")

    def prepare(
        upstream_config: Any,
        _upstream_manager: Any,
        selected_plan: SelectiveLoadPlan,
    ) -> PreparedQ32State:
        if selected_plan is not plan:
            raise ValueError("worker and owner must share one immutable Q32 plan")
        physical_config = copy(upstream_config)
        physical_config.num_experts = 512
        physical_config.num_logical_experts = 512
        physical_config.num_local_experts = 32
        manager = PhysicalQ32ExpertMapManager(
            torch_module, selected_plan.logical_to_physical
        )
        return PreparedQ32State(
            physical_config,
            manager,
            selected_plan.logical_to_physical,
        )

    prepare.round16_api_identity = Q32_PHYSICAL_CONFIG_PREPARER_API

    def rollback(owner: Any, _state: PreparedQ32State) -> None:
        for name in ("_parameters", "_buffers"):
            values = getattr(owner, name, None)
            if isinstance(values, dict):
                values.clear()

    rollback.round16_api_identity = Q32_ALLOCATION_ROLLBACK_API
    offline_cls = build_source_bound_q32_routed_experts_class(
        native_routed_experts_cls,
        physical_config_preparer=prepare,
        allocation_rollback=rollback,
    )

    class LivePhysicalQ32RoutedExperts(offline_cls):
        round16_q32_api_identity = ROUTED_EXPERTS_Q32_API
        runtime_runner_binding_verified = True
        live_eligible = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # Match the exact Ascend BF16 lifecycle before AutoWeightsLoader
            # materializes checkpoint rows and invokes post-load conversion.
            self.quant_method = ascend_method_cls(self.moe_config)
            setup_moe_comm_method(self.moe_config)
            self.round16_ascend_execution_method = type(
                self.quant_method
            ).__name__
            self.round16_moe_comm_setup_verified = True
            self.round16_loaded_projection_bindings: set[
                tuple[int, str]
            ] = set()
            self.round16_full_fused_source_tensors = 0

        def weight_loader(
            self,
            param: Any,
            loaded_weight: Any,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: bool = False,
        ) -> Any:
            result = super().weight_loader(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=return_success,
            )
            if return_success and result is True:
                self.round16_loaded_projection_bindings.add(
                    (int(expert_id), str(shard_id))
                )
            return result

        def load_weights(self, weights: Any) -> Any:
            def guarded_weights() -> Any:
                for name, loaded_weight in weights:
                    shape = tuple(getattr(loaded_weight, "shape", ()))
                    device = getattr(loaded_weight, "device", None)
                    device_type = str(getattr(device, "type", device)).lower()
                    if len(shape) == 3 and shape[0] == 512:
                        self.round16_full_fused_source_tensors += 1
                        if "npu" in device_type:
                            raise RuntimeError(
                                "full 512-expert checkpoint tensor reached NPU "
                                "before resident row selection"
                            )
                    yield name, loaded_weight

            yield from super().load_weights(guarded_weights())

        def round16_validate_checkpoint_coverage(self) -> None:
            expected = {
                (logical_id, shard_id)
                for logical_id in self.round16_resident_logical_ids
                for shard_id in ("w1", "w2", "w3")
            }
            observed = set(self.round16_loaded_projection_bindings)
            if observed != expected:
                missing = sorted(expected - observed)
                unexpected = sorted(observed - expected)
                raise RuntimeError(
                    "Q32 checkpoint coverage differs: "
                    f"missing={missing[:8]} unexpected={unexpected[:8]}"
                )

    LivePhysicalQ32RoutedExperts.__name__ = "LivePhysicalQ32RoutedExperts"
    LivePhysicalQ32RoutedExperts.__qualname__ = (
        "LivePhysicalQ32RoutedExperts"
    )
    return LivePhysicalQ32RoutedExperts


@dataclass(frozen=True)
class _PendingKT:
    wrapper: Any
    hidden_states: Any
    stream_handle: int
    identity_key: tuple[Any, ...]


class FullModelKTLayerFactory:
    """Build one layer wrapper while sharing KT's process-global CPUInfer."""

    def __init__(
        self,
        *,
        torch_module: Any,
        wrapper_cls: type,
        stream_handle_provider: Callable[[str], int],
        bank_root: Path,
        plan: SelectiveLoadPlan,
    ) -> None:
        self.torch = torch_module
        self.wrapper_cls = wrapper_cls
        self.stream_handle_provider = stream_handle_provider
        self.bank_root = bank_root
        self.plan = plan
        self.tensor_ops = TorchNpuTensorOps(torch_module)
        self.merger = TensorWeightedAggregateMerger(self.tensor_ops)
        self._lock = Lock()
        self._wrappers: dict[int, Any] = {}

    @staticmethod
    def layer_index(layer_name: str) -> int:
        match = _LAYER_PATTERN.search(layer_name)
        if match is None:
            raise ValueError("cannot resolve KT bank layer from owner path")
        value = int(match.group(1))
        if not (0 <= value < 48):
            raise ValueError("KT bank layer must be in [0,48)")
        return value

    def _create_wrapper(self, layer: int) -> Any:
        bank = self.bank_root / f"layer-{layer:02d}-mxfp4.gguf"
        if not bank.is_file():
            raise FileNotFoundError(f"Round16-B MXFP4 bank is missing: {bank}")
        mask = self.torch.zeros(512, dtype=self.torch.bool, device="cpu")
        for logical_id in self.plan.resident_logical_ids:
            mask[logical_id] = True
        with self.torch.device("cpu"):
            wrapper = self.wrapper_cls(
                layer_idx=layer,
                num_experts=512,
                num_experts_per_tok=10,
                hidden_size=2560,
                moe_intermediate_size=640,
                gpu_experts_mask=mask,
                cpuinfer_threads=128,
                threadpool_count=4,
                weight_path=str(bank),
                chunked_prefill_size=32,
                cpu_save=False,
                max_deferred_experts_per_token=0,
                method="LLAMAFILE",
                numa_nodes=[2, 3, 4, 7],
            )
        wrapper.load_weights(
            self.torch.arange(512, dtype=self.torch.int32, device="cpu")
        )
        return wrapper

    def wrapper_for(self, layer_name: str) -> Any:
        layer = self.layer_index(layer_name)
        with self._lock:
            wrapper = self._wrappers.get(layer)
            if wrapper is None:
                wrapper = self._create_wrapper(layer)
                self._wrappers[layer] = wrapper
            return wrapper

    def callbacks(self, layer_name: str) -> tuple[Callable[..., Any], ...]:
        wrapper = self.wrapper_for(layer_name)

        def submit(**kwargs: Any) -> _PendingKT:
            stream_handle = int(self.stream_handle_provider("npu"))
            wrapper.submit_forward(
                kwargs["hidden_states"],
                kwargs["topk_ids"],
                kwargs["topk_weights"],
                stream_handle,
            )
            return _PendingKT(
                wrapper,
                kwargs["hidden_states"],
                stream_handle,
                tuple(kwargs["identity_key"]),
            )

        submit.round16_api_identity = EXACT_WORKER_KT_SUBMIT_API

        def sync(**kwargs: Any) -> KTDeviceWeightedPartial:
            pending = kwargs["pending"]
            if not isinstance(pending, _PendingKT) or pending.wrapper is not wrapper:
                raise RuntimeError("KT completion belongs to another layer")
            if tuple(kwargs["identity_key"]) != pending.identity_key:
                raise RuntimeError("KT completion identity changed")
            output = wrapper.sync_forward(
                kwargs["reference_tensor"], pending.stream_handle
            )
            output = output.to(dtype=kwargs["reference_tensor"].dtype)
            return KTDeviceWeightedPartial(output)

        sync.round16_api_identity = EXACT_WORKER_KT_SYNC_API

        def abort(**kwargs: Any) -> None:
            pending = kwargs["pending"]
            if not isinstance(pending, _PendingKT) or pending.wrapper is not wrapper:
                raise RuntimeError("KT abort belongs to another layer")
            reference = kwargs.get("reference_tensor")
            if reference is None:
                reference = pending.hidden_states
            wrapper.sync_forward(reference, pending.stream_handle)

        abort.round16_api_identity = EXACT_WORKER_KT_ABORT_API
        return submit, sync, abort

    def merge(self, **kwargs: Any) -> CombinedDeviceWeightedPartial:
        return self.merger.merge(kwargs["native"], kwargs["kt"])


FullModelKTLayerFactory.merge.round16_api_identity = (
    EXACT_WORKER_TENSOR_MERGE_API
)


def _require_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    path = Path(value).resolve()
    if not path.exists():
        raise ValueError(f"{name} path is unavailable: {path}")
    return path


def build_live_bindings(
    *, model_runner_cls: type, worker_source_token: Any
) -> B2ExactWorkerPluginBindings:
    """Resolve all exact donor objects inside the real TP1 worker process."""

    import torch
    import torch_npu  # noqa: F401
    import vllm.model_executor.models.qwen3_next as qwen_module
    from kt_kernel import get_current_device_stream_handle
    from kt_kernel.experts_base import BaseMoEWrapper
    from kt_kernel.utils.llamafile import LlamafileMoEWrapper
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExpertsOrder,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm_ascend.models.qwen4_exp.nvidia import model as _qwen_model  # noqa: F401
    from vllm_ascend.models.qwen4_exp.nvidia.ple_layer import (
        Qwen4ExpNGramEmbedding,
    )
    from vllm_ascend.models.qwen4_exp.ple_offload_layer import PleOffloadLayer
    from vllm_ascend.models.qwen4_exp import qsa as ascend_qsa_module
    from vllm_ascend.ops.fused_moe.fused_moe import (
        AscendUnquantizedFusedMoEMethod,
    )
    from vllm_ascend.ops.fused_moe.experts_selector import (
        select_experts as ascend_select_experts,
    )
    from vllm_ascend.ops.fused_moe.moe_comm_method import setup_moe_comm_method
    from vllm_ascend.ops.fused_moe.moe_runtime_args import (
        build_fused_experts_input,
    )
    from vllm_ascend.patch.platform.patch_fused_moe import _ascend_FusedMoE
    from vllm_ascend.patch.worker import patch_process_weights_after_loading
    from vllm_ascend.quantization.quant_type import QuantType
    from vllm_ascend.utils import enable_custom_op

    install_host_ple_bridge(
        torch_module=torch,
        ple_offload_cls=PleOffloadLayer,
        ple_embedding_cls=Qwen4ExpNGramEmbedding,
        postload_patch_module=patch_process_weights_after_loading,
    )
    install_ascend_qsa_cache_split(qsa_module=ascend_qsa_module)
    install_ascend_qsa_multihead_store(
        qsa_impl_cls=ascend_qsa_module.AscendQSAImpl,
        qsa_module=ascend_qsa_module,
    )

    snapshot = load_snapshot_manifest(
        _require_env_path("ROUND16_B_SNAPSHOT_MANIFEST")
    )
    source_token = load_source_preflight_token(
        _require_env_path("ROUND16_B_SOURCE_TOKEN")
    )
    plan = plan_q32(snapshot.resident_logical_ids)
    owner_cls = build_live_q32_owner_class(
        native_routed_experts_cls=RoutedExperts,
        torch_module=torch,
        ascend_method_cls=AscendUnquantizedFusedMoEMethod,
        setup_moe_comm_method=setup_moe_comm_method,
        plan=plan,
    )
    if qwen_module.FusedMoE is not _ascend_FusedMoE:
        raise RuntimeError("target Qwen module lacks the exact Ascend FusedMoE binding")

    BaseMoEWrapper.set_capture_batch_sizes([1, 32])
    kt_layers = FullModelKTLayerFactory(
        torch_module=torch,
        wrapper_cls=LlamafileMoEWrapper,
        stream_handle_provider=get_current_device_stream_handle,
        bank_root=_require_env_path("ROUND16_B_BANK_ROOT"),
        plan=plan,
    )

    stream_cache: dict[int, Any] = {}

    def current_stream() -> Any:
        stream = torch.npu.current_stream()
        key = int(getattr(stream, "npu_stream", id(stream)))
        cached = stream_cache.get(key)
        if cached is None:
            stream_cache[key] = stream
            cached = stream
        return cached

    tensor_binding_factory = make_current_allgather_tensor_binding_factory(
        get_forward_context=get_forward_context,
        current_stream=current_stream,
        build_fused_experts_input=build_fused_experts_input,
        quant_none=QuantType.NONE,
        tensor_ops=TorchNpuTensorOps(torch),
    )

    def forward_identity(layer_name: str) -> tuple[Any, ...]:
        context = get_forward_context()
        input_ids = getattr(context, "input_ids", None)
        pointer = None
        if input_ids is not None and callable(getattr(input_ids, "data_ptr", None)):
            pointer = int(input_ids.data_ptr())
        return ("vllm", id(context), pointer, layer_name)

    forward_identity.round16_api_identity = EXACT_WORKER_FORWARD_IDENTITY_API

    def runtime_factory(**kwargs: Any) -> Any:
        submit, sync, abort = kt_layers.callbacks(kwargs["layer_name"])
        factory = make_exact_device_runtime_factory(
            source_preflight_token=source_token,
            tensor_binding_factory=tensor_binding_factory,
            forward_identity_provider=forward_identity,
            kt_submit=submit,
            kt_sync=sync,
            kt_abort_drain=abort,
            merge_weighted_partials=kt_layers.merge,
        )
        return factory(**kwargs)

    expert_selector = make_ascend_expert_selector(
        ascend_select_experts,
        enable_custom_op=enable_custom_op,
    )

    from round16.b.exact_donor_worker_hook import EXACT_LAYER_RUNTIME_FACTORY_API

    runtime_factory.round16_api_identity = EXACT_LAYER_RUNTIME_FACTORY_API
    return B2ExactWorkerPluginBindings(
        model_runner_cls=model_runner_cls,
        qwen_module=qwen_module,
        ascend_fused_moe=_ascend_FusedMoE,
        routed_experts_cls=owner_cls,
        upstream_moe_runner_cls=MoERunner,
        shared_order_no_overlap=SharedExpertsOrder.NO_OVERLAP,
        shared_order_overlap=SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        plan=plan,
        worker_source_token=worker_source_token,
        runtime_factory=runtime_factory,
        expert_selector=expert_selector,
        expected_layer_count=48,
    )


build_live_bindings.round16_api_identity = B2_BINDINGS_FACTORY_API


__all__ = (
    "ASCEND_QSA_CACHE_RESHAPE_API",
    "ASCEND_QSA_CACHE_SPLIT_API",
    "ASCEND_QSA_CACHE_SHAPE_API",
    "ASCEND_QSA_MULTIHEAD_STORE_API",
    "B2_LIVE_BINDINGS_API",
    "HOST_PLE_BRIDGE_API",
    "FullModelKTLayerFactory",
    "PhysicalQ32ExpertMapManager",
    "build_live_bindings",
    "build_live_q32_owner_class",
    "install_ascend_qsa_cache_reshape",
    "install_ascend_qsa_cache_split",
    "install_ascend_qsa_cache_shape",
    "install_ascend_qsa_multihead_store",
    "install_host_ple_bridge",
    "make_ascend_expert_selector",
)
