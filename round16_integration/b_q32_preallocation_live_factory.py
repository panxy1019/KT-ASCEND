"""Exact-image factory for the bounded Round16-B V1 Q32 constructor probe.

This module is imported only after the outer probe validates frozen source and
snapshot identities and explicit device enablement.  It constructs no model,
reads no checkpoint or MXFP4 bank, and initializes no CPUInfer workers.
"""

from __future__ import annotations

from copy import copy
import gc
from typing import Any

from round16.b.selective_loader import plan_q32
from round16.b.source_bound_q32_adapter import (
    PreparedQ32State,
    Q32_ALLOCATION_ROLLBACK_API,
    Q32_PHYSICAL_CONFIG_PREPARER_API,
    build_source_bound_q32_routed_experts_class,
)
from round16_integration.b_q32_preallocation_probe import (
    Q32PreallocationBinding,
    Q32_V1_CONTEXT_API,
    Q32_V1_FACTORY_API,
)


class _LogicalManager:
    global_num_experts = 512
    local_num_experts = 512

    @staticmethod
    def map_global_to_local(expert_id: int) -> int:
        return expert_id


class _PhysicalQ32Manager:
    def __init__(self, torch_module: Any, mapping: tuple[int, ...]) -> None:
        self.global_num_experts = 512
        self.local_num_experts = 32
        self._mapping = mapping
        self.placement_strategy = "linear"
        self.expert_map = torch_module.tensor(mapping, dtype=torch_module.int32)
        self.expert_mask = None
        self.routing_tables = None

    def map_global_to_local(self, expert_id: int) -> int:
        return self._mapping[expert_id]


def _build_moe_config(torch_module: Any) -> Any:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather",
        enable_eplb=False,
    )
    return FusedMoEConfig(
        num_experts=512,
        experts_per_token=10,
        hidden_dim=2560,
        intermediate_size=640,
        num_local_experts=512,
        num_logical_experts=512,
        activation=MoEActivation.SILU,
        device="npu",
        routing_method=RoutingMethodType.Sigmoid,
        moe_parallel_config=parallel,
        in_dtype=torch_module.bfloat16,
        router_logits_dtype=torch_module.float32,
        max_num_tokens=32,
        has_bias=False,
        moe_backend="auto",
    )


def _create_raw_q32_binding(context: Any) -> Q32PreallocationBinding:
    """Construct the reviewed raw Q32 owner after an outer stage gate.

    V1 and later bounded probes share this constructor so they cannot drift in
    the allocation order that V1 qualified.  Each public probe validates its
    own immutable context identity before calling this helper.
    """

    if not all(hasattr(context, name) for name in ("source_token", "snapshot", "layer_id")):
        raise TypeError("Q32 live factory context fields are incomplete")
    import torch
    import torch_npu  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    plan = plan_q32(context.snapshot.resident_logical_ids)

    def prepare(moe_config: Any, manager: Any, selected_plan: Any) -> PreparedQ32State:
        if selected_plan is not plan:
            raise ValueError("V1 factory did not receive the identical resident plan")
        physical_config = copy(moe_config)
        physical_config.num_experts = 512
        physical_config.num_logical_experts = 512
        physical_config.num_local_experts = 32
        physical_manager = _PhysicalQ32Manager(
            torch, selected_plan.logical_to_physical
        )
        return PreparedQ32State(
            physical_config,
            physical_manager,
            selected_plan.logical_to_physical,
        )

    prepare.round16_api_identity = Q32_PHYSICAL_CONFIG_PREPARER_API

    def rollback(owner: Any, _state: PreparedQ32State) -> None:
        parameters = getattr(owner, "_parameters", None)
        if isinstance(parameters, dict):
            parameters.clear()

    rollback.round16_api_identity = Q32_ALLOCATION_ROLLBACK_API

    routed_cls = build_source_bound_q32_routed_experts_class(
        RoutedExperts,
        physical_config_preparer=prepare,
        allocation_rollback=rollback,
    )
    logical_config = _build_moe_config(torch)
    logical_manager = _LogicalManager()
    torch.npu.synchronize()
    before = int(torch.npu.memory_allocated())
    torch.npu.reset_peak_memory_stats()
    # A real worker establishes this context before model construction.
    # The bounded component probe must reproduce that lifecycle rather than
    # instantiate CustomOp outside its supported configuration boundary.
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config), torch.device("npu"):
        owner = routed_cls(
            f"model.layers.{context.layer_id}.mlp.experts",
            torch.bfloat16,
            logical_config,
            None,
            expert_map_manager=logical_manager,
            round16_selective_plan=plan,
            round16_resident_logical_ids=plan.resident_logical_ids,
            round16_logical_to_physical=plan.logical_to_physical,
            round16_logical_expert_count=512,
            round16_physical_expert_count=32,
        )
    torch.npu.synchronize()
    after = int(torch.npu.memory_allocated())
    peak = int(torch.npu.max_memory_allocated())
    native_method = getattr(owner.quant_method, "_native", None)
    construction_method = type(native_method).__name__

    def cleanup() -> None:
        parameters = getattr(owner, "_parameters", None)
        if isinstance(parameters, dict):
            parameters.clear()
        buffers = getattr(owner, "_buffers", None)
        if isinstance(buffers, dict):
            buffers.clear()
        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()

    return Q32PreallocationBinding(
        owner=owner,
        torch_module=torch,
        memory_before_bytes=before,
        memory_after_bytes=after,
        peak_memory_bytes=peak,
        construction_method=construction_method,
        cleanup=cleanup,
    )


def create_q32_preallocation_binding(context: Any) -> Q32PreallocationBinding:
    # ``python -m`` executes the outer probe as ``__main__``.  The factory is
    # imported by canonical package name, so Python class identity is not a
    # stable cross-module protocol boundary.  Require the immutable API token
    # and exact fields instead of accepting an arbitrary duck type.
    if getattr(context, "round16_api_identity", None) != Q32_V1_CONTEXT_API:
        raise TypeError("V1 live factory received an unverified context contract")
    return _create_raw_q32_binding(context)


create_q32_preallocation_binding.round16_api_identity = Q32_V1_FACTORY_API


__all__ = ("create_q32_preallocation_binding",)
