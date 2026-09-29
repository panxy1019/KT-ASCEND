"""Source-bound Q32 RoutedExperts and checkpoint allocation contract.

The adapter in this module is deliberately *not* a live Hybrid runner binding.
The frozen FusedMoE factory creates the upstream ExpertMapManager and
``moe_config`` before it constructs RoutedExperts, while the default
AscendMoERunner consumes those original objects afterwards.  A Q32
RoutedExperts owner therefore has to be paired with a verified custom Hybrid
runner.  Until that runner seam exists, classes built here expose only the
source-bound construction/loader contract and are rejected by the live
installer.

There are no torch, vLLM, vLLM-Ascend, or device imports in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .selective_loader import (
    LOGICAL_EXPERT_COUNT,
    RESIDENT_EXPERT_COUNT,
    SelectiveLoadPlan,
    plan_q32,
)
from .selective_preallocation_installer import ROUTED_EXPERTS_Q32_API


SOURCE_BOUND_Q32_ROUTED_EXPERTS_API = (
    "round16-b:source-bound-q32-routed-experts:offline-v1"
)
Q32_PHYSICAL_CONFIG_PREPARER_API = (
    "round16-b:q32-physical-config-preparer:clone-v1"
)
Q32_ALLOCATION_ROLLBACK_API = "round16-b:q32-allocation-rollback:v1"


@dataclass(frozen=True)
class PreparedQ32State:
    """Cloned physical state prepared before native parameter allocation."""

    moe_config: Any
    expert_map_manager: Any
    logical_to_physical: tuple[int, ...]
    rollback_token: Any = None

    def __post_init__(self) -> None:
        try:
            mapping = tuple(self.logical_to_physical)
        except TypeError as error:
            raise ValueError("prepared logical-to-physical mapping must be iterable") from error
        if len(mapping) != LOGICAL_EXPERT_COUNT or any(
            type(value) is not int for value in mapping
        ):
            raise ValueError("prepared mapping must contain 512 exact integer entries")
        object.__setattr__(self, "logical_to_physical", mapping)


@dataclass(frozen=True)
class Q32CheckpointBinding:
    """One reviewed logical checkpoint shard to physical Q32 slot binding."""

    parameter_name: str
    checkpoint_name: str
    shard_id: str
    logical_expert_id: int
    physical_expert_id: int

    def __post_init__(self) -> None:
        if self.parameter_name not in ("experts.w13_weight", "experts.w2_weight"):
            raise ValueError("checkpoint target must be w13_weight or w2_weight")
        if self.shard_id not in ("w1", "w2", "w3"):
            raise ValueError("checkpoint shard id must be w1, w2, or w3")
        if not self.checkpoint_name:
            raise ValueError("checkpoint name must be non-empty")
        if type(self.logical_expert_id) is not int or not (
            0 <= self.logical_expert_id < LOGICAL_EXPERT_COUNT
        ):
            raise ValueError("logical checkpoint expert id is invalid")
        if type(self.physical_expert_id) is not int or not (
            0 <= self.physical_expert_id < RESIDENT_EXPERT_COUNT
        ):
            raise ValueError("physical checkpoint expert id is invalid")


class Q32RollbackError(Exception):
    """Rollback failure chained behind the original constructor failure."""


def _strict_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_plan(plan: Any) -> SelectiveLoadPlan:
    if not isinstance(plan, SelectiveLoadPlan):
        raise ValueError("round16 selective plan must be a SelectiveLoadPlan")
    expected = plan_q32(plan.resident_logical_ids)
    if plan != expected:
        raise ValueError("round16 selective plan is not an exact Q32 plan")
    return plan


def _validate_installer_args(
    plan: Any,
    resident_logical_ids: Any,
    logical_to_physical: Any,
    logical_expert_count: Any,
    physical_expert_count: Any,
) -> SelectiveLoadPlan:
    plan = _validated_plan(plan)
    try:
        resident = tuple(resident_logical_ids)
        mapping = tuple(logical_to_physical)
    except TypeError as error:
        raise ValueError("immutable Q32 installer arguments are malformed") from error
    if (
        resident != plan.resident_logical_ids
        or mapping != plan.logical_to_physical
        or type(logical_expert_count) is not int
        or logical_expert_count != LOGICAL_EXPERT_COUNT
        or type(physical_expert_count) is not int
        or physical_expert_count != RESIDENT_EXPERT_COUNT
    ):
        raise ValueError("immutable Q32 installer arguments do not match the plan")
    return plan


def _field_exact_int(owner: Any, name: str, expected: int) -> None:
    value = getattr(owner, name, None)
    if type(value) is not int or value != expected:
        raise ValueError(f"prepared {name} must be exactly {expected}")


def _validate_prepared_state(
    state: Any,
    *,
    upstream_moe_config: Any,
    upstream_expert_map_manager: Any,
    plan: SelectiveLoadPlan,
) -> PreparedQ32State:
    if not isinstance(state, PreparedQ32State):
        raise TypeError("physical config preparer must return PreparedQ32State")
    if state.moe_config is upstream_moe_config:
        raise ValueError(
            "source-bound Q32 preparation must not mutate shared upstream moe_config"
        )
    if state.expert_map_manager is upstream_expert_map_manager:
        raise ValueError(
            "source-bound Q32 preparation must not mutate shared ExpertMapManager"
        )
    if state.logical_to_physical != plan.logical_to_physical:
        raise ValueError("prepared Q32 mapping differs from the frozen plan")
    _field_exact_int(state.moe_config, "num_experts", LOGICAL_EXPERT_COUNT)
    _field_exact_int(state.moe_config, "num_logical_experts", LOGICAL_EXPERT_COUNT)
    _field_exact_int(state.moe_config, "num_local_experts", RESIDENT_EXPERT_COUNT)
    _field_exact_int(
        state.expert_map_manager, "global_num_experts", LOGICAL_EXPERT_COUNT
    )
    _field_exact_int(
        state.expert_map_manager, "local_num_experts", RESIDENT_EXPERT_COUNT
    )
    mapper = getattr(state.expert_map_manager, "map_global_to_local", None)
    if not callable(mapper):
        raise ValueError("prepared ExpertMapManager has no global-to-local mapper")
    for logical_id, expected_physical_id in enumerate(plan.logical_to_physical):
        observed_physical_id = mapper(logical_id)
        if (
            type(observed_physical_id) is not int
            or observed_physical_id != expected_physical_id
        ):
            raise ValueError(
                "prepared ExpertMapManager mapper does not match the frozen "
                f"Q32 plan at logical expert {logical_id}"
            )
    return state


def _shape(value: Any, name: str) -> tuple[int, ...]:
    try:
        shape = tuple(getattr(value, "shape"))
    except (AttributeError, TypeError) as error:
        raise RuntimeError(f"native allocation did not expose {name}.shape") from error
    if not shape or any(type(dimension) is not int for dimension in shape):
        raise RuntimeError(f"native allocation exposed an invalid {name} shape")
    if shape[0] != RESIDENT_EXPERT_COUNT:
        raise RuntimeError(
            f"{name} physical expert dimension must be exactly {RESIDENT_EXPERT_COUNT}"
        )
    return shape


class SourceBoundQ32QuantMethodAdapter:
    """Gate native ``create_weights`` at the reviewed Q32 source contract."""

    def __init__(self, native_quant_method: Any) -> None:
        create_weights = getattr(native_quant_method, "create_weights", None)
        if not callable(create_weights):
            raise TypeError("native quant method must expose create_weights")
        self._native = native_quant_method

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)

    def create_weights(
        self,
        layer: Any,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: Any,
        **extra_weight_attrs: Any,
    ) -> None:
        if type(num_experts) is not int or num_experts != RESIDENT_EXPERT_COUNT:
            raise ValueError("native Q32 create_weights must receive physical num_experts=32")
        global_count = extra_weight_attrs.get("global_num_experts")
        if type(global_count) is not int or global_count != LOGICAL_EXPERT_COUNT:
            raise ValueError("native Q32 create_weights must preserve global_num_experts=512")
        if not callable(extra_weight_attrs.get("weight_loader")):
            raise ValueError("native Q32 create_weights must retain a callable weight_loader")
        _strict_positive_int(hidden_size, "hidden_size")
        _strict_positive_int(
            intermediate_size_per_partition,
            "intermediate_size_per_partition",
        )
        self._native.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )
        w13 = getattr(layer, "w13_weight", None)
        w2 = getattr(layer, "w2_weight", None)
        if w13 is w2:
            raise RuntimeError("w13_weight and w2_weight cannot share one allocation")
        _shape(w13, "w13_weight")
        _shape(w2, "w2_weight")


def build_q32_checkpoint_bindings(
    plan: SelectiveLoadPlan,
    *,
    gate_proj_name: str = "gate_proj",
    down_proj_name: str = "down_proj",
    up_proj_name: str = "up_proj",
) -> tuple[Q32CheckpointBinding, ...]:
    """Build the reviewed w13/w2 source contract without reading a checkpoint."""

    plan = _validated_plan(plan)
    projections = (gate_proj_name, down_proj_name, up_proj_name)
    if any(type(name) is not str or not name for name in projections):
        raise ValueError("checkpoint projection names must be non-empty strings")
    if len(set(projections)) != 3:
        raise ValueError("checkpoint projection names must be distinct")
    shard_specs = (
        ("w1", gate_proj_name, "experts.w13_weight"),
        ("w2", down_proj_name, "experts.w2_weight"),
        ("w3", up_proj_name, "experts.w13_weight"),
    )
    bindings = []
    for physical_id, logical_id in enumerate(plan.resident_logical_ids):
        for shard_id, projection, parameter_name in shard_specs:
            bindings.append(
                Q32CheckpointBinding(
                    parameter_name=parameter_name,
                    checkpoint_name=f"experts.{logical_id}.{projection}.weight",
                    shard_id=shard_id,
                    logical_expert_id=logical_id,
                    physical_expert_id=physical_id,
                )
            )
    return tuple(bindings)


def _rollback_preserving_primary(
    primary: BaseException,
    rollback: Callable[[Any, PreparedQ32State], Any],
    owner: Any,
    state: PreparedQ32State,
) -> None:
    try:
        rollback(owner, state)
    except BaseException as cleanup_error:
        aggregate = Q32RollbackError(f"Q32 allocation rollback failed: {cleanup_error}")
        if hasattr(primary, "add_note"):
            primary.add_note(str(aggregate))
        raise primary from aggregate
    raise primary


def build_source_bound_q32_routed_experts_class(
    native_routed_experts_cls: type,
    *,
    physical_config_preparer: Callable[[Any, Any, SelectiveLoadPlan], PreparedQ32State],
    allocation_rollback: Callable[[Any, PreparedQ32State], Any],
) -> type:
    """Create an offline source-bound native subclass.

    The returned class intentionally lacks ``round16_q32_api_identity`` and
    sets ``runtime_runner_binding_verified`` false.  The live scoped installer
    must reject it until a matching custom Hybrid runner is source- and
    device-verified.
    """

    if not isinstance(native_routed_experts_cls, type):
        raise TypeError("native RoutedExperts must be a class")
    if (
        getattr(physical_config_preparer, "round16_api_identity", None)
        != Q32_PHYSICAL_CONFIG_PREPARER_API
    ):
        raise ValueError("physical config preparer API identity mismatch")
    if (
        getattr(allocation_rollback, "round16_api_identity", None)
        != Q32_ALLOCATION_ROLLBACK_API
    ):
        raise ValueError("allocation rollback API identity mismatch")
    if not callable(physical_config_preparer) or not callable(allocation_rollback):
        raise TypeError("Q32 preparation and rollback hooks must be callable")

    class SourceBoundQ32RoutedExperts(native_routed_experts_cls):
        round16_source_bound_api_identity = SOURCE_BOUND_Q32_ROUTED_EXPERTS_API
        runtime_runner_binding_verified = False
        _round16_preselected_executor: Callable[..., Any] | None = None

        def __init__(
            self,
            layer_name: str,
            params_dtype: Any,
            moe_config: Any,
            quant_config: Any,
            expert_map_manager: Any,
            ckpt_gate_proj_name: str = "gate_proj",
            ckpt_down_proj_name: str = "down_proj",
            ckpt_up_proj_name: str = "up_proj",
            *,
            round16_selective_plan: SelectiveLoadPlan,
            round16_resident_logical_ids: Sequence[int],
            round16_logical_to_physical: Sequence[int],
            round16_logical_expert_count: int,
            round16_physical_expert_count: int,
            **kwargs: Any,
        ) -> None:
            plan = _validate_installer_args(
                round16_selective_plan,
                round16_resident_logical_ids,
                round16_logical_to_physical,
                round16_logical_expert_count,
                round16_physical_expert_count,
            )
            upstream_config = moe_config
            upstream_manager = expert_map_manager
            state = physical_config_preparer(upstream_config, upstream_manager, plan)
            state = _validate_prepared_state(
                state,
                upstream_moe_config=upstream_config,
                upstream_expert_map_manager=upstream_manager,
                plan=plan,
            )
            self.round16_selective_plan = plan
            self.round16_resident_logical_ids = plan.resident_logical_ids
            self.round16_logical_to_physical = plan.logical_to_physical
            self.round16_logical_expert_count = LOGICAL_EXPERT_COUNT
            self.round16_physical_expert_count = RESIDENT_EXPERT_COUNT
            self.round16_upstream_moe_config = upstream_config
            self.round16_upstream_expert_map_manager = upstream_manager
            self._round16_prepared_state = state
            try:
                super().__init__(
                    layer_name,
                    params_dtype,
                    state.moe_config,
                    quant_config,
                    expert_map_manager=state.expert_map_manager,
                    ckpt_gate_proj_name=ckpt_gate_proj_name,
                    ckpt_down_proj_name=ckpt_down_proj_name,
                    ckpt_up_proj_name=ckpt_up_proj_name,
                    **kwargs,
                )
            except BaseException as primary_error:
                _rollback_preserving_primary(
                    primary_error,
                    allocation_rollback,
                    self,
                    state,
                )
            if self.local_num_experts != RESIDENT_EXPERT_COUNT:
                primary_error = RuntimeError(
                    "native RoutedExperts did not retain physical expert count 32"
                )
                _rollback_preserving_primary(
                    primary_error,
                    allocation_rollback,
                    self,
                    state,
                )
            if self.global_num_experts != LOGICAL_EXPERT_COUNT:
                primary_error = RuntimeError(
                    "native RoutedExperts did not retain logical expert count 512"
                )
                _rollback_preserving_primary(
                    primary_error,
                    allocation_rollback,
                    self,
                    state,
                )

        def _get_quant_method(
            self,
            prefix: str,
            quant_config: Any,
            moe_config: Any,
        ) -> SourceBoundQ32QuantMethodAdapter:
            native = super()._get_quant_method(prefix, quant_config, moe_config)
            return SourceBoundQ32QuantMethodAdapter(native)

        def get_round16_checkpoint_bindings(
            self,
        ) -> tuple[Q32CheckpointBinding, ...]:
            return build_q32_checkpoint_bindings(
                self.round16_selective_plan,
                gate_proj_name=self.ckpt_gate_proj_name,
                down_proj_name=self.ckpt_down_proj_name,
                up_proj_name=self.ckpt_up_proj_name,
            )

        def round16_bind_preselected_executor(
            self,
            executor: Callable[..., Any],
            plan: SelectiveLoadPlan,
        ) -> None:
            if not callable(executor):
                raise TypeError("preselected routed executor must be callable")
            if plan is not self.round16_selective_plan:
                raise ValueError("runner and Q32 owner must share the identical plan")
            if self._round16_preselected_executor is not None:
                raise RuntimeError("preselected routed executor is already bound")
            self._round16_preselected_executor = executor

        def forward_modular(
            self,
            *,
            x: Any,
            topk_weights: Any,
            topk_ids: Any,
            shared_experts: Any = None,
            shared_experts_input: Any = None,
        ) -> Any:
            del shared_experts, shared_experts_input
            executor = self._round16_preselected_executor
            if executor is None:
                raise RuntimeError(
                    "source-bound Q32 owner has no preselected runner executor"
                )
            return executor(
                hidden_states=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                input_ids=None,
            )

    SourceBoundQ32RoutedExperts.__name__ = "SourceBoundQ32RoutedExperts"
    SourceBoundQ32RoutedExperts.__qualname__ = "SourceBoundQ32RoutedExperts"
    SourceBoundQ32RoutedExperts.__module__ = __name__
    return SourceBoundQ32RoutedExperts


__all__ = (
    "PreparedQ32State",
    "Q32_ALLOCATION_ROLLBACK_API",
    "Q32_PHYSICAL_CONFIG_PREPARER_API",
    "Q32CheckpointBinding",
    "Q32RollbackError",
    "ROUTED_EXPERTS_Q32_API",
    "SOURCE_BOUND_Q32_ROUTED_EXPERTS_API",
    "SourceBoundQ32QuantMethodAdapter",
    "build_q32_checkpoint_bindings",
    "build_source_bound_q32_routed_experts_class",
)
