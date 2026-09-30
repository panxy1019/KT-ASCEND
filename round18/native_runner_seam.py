"""Q32 routed-expert seam that preserves the native Ascend MoE runner.

Round16 proved the physical-Q32 owner and selective loader with a custom
Hybrid runner.  Round18 deliberately keeps the frozen ``AscendMoERunner``.
The owner is still constructed with a cloned physical view, while this module
binds the same Q32 global-to-physical map to the already-created native runner.

This module uses no torch or vLLM imports so its ownership and failure rules
can be tested before loading a model or initializing an NPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock, get_ident
from types import MappingProxyType
from typing import Any, Iterator, Sequence

from round16.b.selective_loader import SelectiveLoadPlan, plan_q32


NATIVE_PARTIAL_SEAM_API = "round18:native-ascend-runner-q32-partial:v1"
_CONSTRUCTION_LOCK = RLock()


def _tuple_values(value: Any, name: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not an integer sequence") from error
    return result


def _shape(value: Any, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} has no valid shape") from error
    if not result or result[0] != 32:
        raise ValueError(f"{name} physical expert dimension must be exactly 32")
    return result


def _validate_plan(plan: Any) -> SelectiveLoadPlan:
    if not isinstance(plan, SelectiveLoadPlan):
        raise TypeError("native partial seam requires a SelectiveLoadPlan")
    if plan != plan_q32(plan.resident_logical_ids):
        raise ValueError("native partial seam requires an exact Q32 plan")
    return plan


@dataclass(frozen=True)
class NativeRunnerBindingReceipt:
    api: str
    runner_class: str
    logical_experts: int
    runner_local_experts: int
    physical_slots: int
    dynamic_eplb: bool
    runner_class_preserved: bool


@dataclass(frozen=True)
class NativePhysicalRuntimeReceipt:
    runner_class: str
    quant_method_class: str
    logical_experts: int
    physical_slots: int
    runner_class_preserved: bool


def rebind_native_physical_quant_method(
    runner: Any,
    *,
    ascend_method_cls: type,
    setup_moe_comm_method: Any,
) -> NativePhysicalRuntimeReceipt:
    """Restore the native Ascend method with the owner's physical-Q32 view.

    The frozen Ascend runner replaces ``RoutedExperts.quant_method`` during
    construction and calls ``setup_moe_comm_method`` with its EP1 logical
    config.  That is correct for an ordinary 512-row owner, but it produces a
    512-entry group list for the Round18 32-row physical owner.  Recreate the
    *same native method* and communication implementation from the owner's
    physical config before checkpoint loading.  The runner, router and
    finalize implementations remain untouched.
    """

    runner_class = type(runner)
    if "AscendMoERunner" not in runner_class.__name__:
        raise TypeError("Round18 requires the frozen native AscendMoERunner")
    if not isinstance(ascend_method_cls, type):
        raise TypeError("native Ascend quant method must be a class")
    if not callable(setup_moe_comm_method):
        raise TypeError("native MoE communication setup must be callable")
    owner = getattr(runner, "routed_experts", None)
    if owner is None:
        raise ValueError("native runner does not expose RoutedExperts")
    physical_config = getattr(owner, "moe_config", None)
    if physical_config is None:
        raise ValueError("Q32 owner has no physical MoE config")
    identity = (
        getattr(physical_config, "num_logical_experts", None),
        getattr(physical_config, "num_experts", None),
        getattr(physical_config, "num_local_experts", None),
        getattr(physical_config, "ep_size", 1),
    )
    if identity != (512, 512, 32, 1):
        raise ValueError(
            "native physical runtime requires logical512/global512/local32/EP1"
        )

    method = ascend_method_cls(
        physical_config,
        tid2eid=getattr(runner, "tid2eid", None),
    )
    owner.quant_method = method
    setup_moe_comm_method(physical_config)
    owner.round18_physical_quant_method = method
    owner.round18_physical_moe_config = physical_config
    runner.round18_physical_quant_method = method
    if getattr(runner, "_quant_method", None) is not method:
        raise RuntimeError("native runner did not resolve the physical Q32 method")
    if type(runner) is not runner_class:
        raise RuntimeError("native runner class changed during physical binding")
    return NativePhysicalRuntimeReceipt(
        runner_class=runner_class.__name__,
        quant_method_class=type(method).__name__,
        logical_experts=512,
        physical_slots=32,
        runner_class_preserved=True,
    )


def execution_map_from_owner(
    owner: Any, plan: SelectiveLoadPlan
) -> Any:
    """Return the concrete frozen Q32 map from either manager API spelling."""

    plan = _validate_plan(plan)
    manager = getattr(owner, "expert_map_manager", None)
    if manager is None:
        raise ValueError("Q32 owner has no expert-map manager")
    execution_map = getattr(manager, "_expert_map", None)
    if execution_map is None:
        execution_map = getattr(manager, "expert_map", None)
    if execution_map is None:
        raise ValueError("Q32 owner has no physical execution map")
    if _tuple_values(execution_map, "physical owner expert map") != (
        plan.logical_to_physical
    ):
        raise ValueError("physical owner expert map differs from the Q32 plan")
    return execution_map


def bind_q32_map_to_native_runner(
    runner: Any,
    plan: SelectiveLoadPlan,
) -> NativeRunnerBindingReceipt:
    """Bind Q32 execution metadata without replacing the native runner.

    The runner keeps its original EP1 logical configuration (512 local model
    experts).  Only its execution map points at the physical-Q32 owner.  This
    distinction is required because the original router and checkpoint remain
    in the 512-expert domain.
    """

    plan = _validate_plan(plan)
    runner_class = type(runner)
    if "AscendMoERunner" not in runner_class.__name__:
        raise TypeError("Round18 requires the frozen native AscendMoERunner")
    owner = getattr(runner, "routed_experts", None)
    if owner is None:
        raise ValueError("native runner does not expose RoutedExperts")
    if getattr(owner, "round16_selective_plan", None) is not plan:
        raise ValueError("native runner and Q32 owner do not share one plan")
    if tuple(getattr(owner, "round16_logical_to_physical", ())) != (
        plan.logical_to_physical
    ):
        raise ValueError("Q32 owner mapping differs from the selected plan")
    _shape(getattr(owner, "w13_weight", None), "w13_weight")
    _shape(getattr(owner, "w2_weight", None), "w2_weight")

    config = getattr(runner, "moe_config", None)
    if config is None:
        raise ValueError("native runner has no MoE config")
    logical = getattr(config, "num_logical_experts", None)
    global_count = getattr(config, "num_experts", None)
    local_count = getattr(config, "num_local_experts", None)
    if (logical, global_count, local_count) != (512, 512, 512):
        raise ValueError(
            "native runner must preserve the original logical512/EP1 capacity"
        )
    if getattr(runner, "dynamic_eplb", False):
        raise ValueError("dynamic EPLB is outside the Round18 Q32 contract")
    if getattr(config, "ep_size", 1) != 1:
        raise ValueError("Round18 native partial seam is TP1/EP1 only")

    manager = owner.expert_map_manager
    execution_map = execution_map_from_owner(owner, plan)

    # Keep the concrete tensor supplied by the owner.  Its device and dtype
    # were established inside the frozen model-construction context.
    runner._expert_map = execution_map
    owner._expert_map = execution_map
    manager._expert_map = execution_map
    runner.round18_native_partial_api = NATIVE_PARTIAL_SEAM_API
    runner.round18_selective_plan = plan
    runner.round18_routed_owner = owner
    runner.round18_native_runner_class = runner_class
    owner.round18_native_partial_api = NATIVE_PARTIAL_SEAM_API

    if type(runner) is not runner_class:
        raise RuntimeError("native runner class changed during Q32 binding")
    return NativeRunnerBindingReceipt(
        api=NATIVE_PARTIAL_SEAM_API,
        runner_class=runner_class.__name__,
        logical_experts=512,
        runner_local_experts=512,
        physical_slots=32,
        dynamic_eplb=False,
        runner_class_preserved=True,
    )


@dataclass(frozen=True)
class SplitTopK:
    resident_logical_ids: tuple[tuple[int, ...], ...]
    resident_physical_ids: tuple[tuple[int, ...], ...]
    resident_weights: tuple[tuple[float, ...], ...]
    cpu_logical_ids: tuple[tuple[int, ...], ...]
    cpu_weights: tuple[tuple[float, ...], ...]


def split_topk_occurrences(
    topk_ids: Sequence[Sequence[int]],
    topk_weights: Sequence[Sequence[float]],
    plan: SelectiveLoadPlan,
) -> SplitTopK:
    """Reference split in occurrence space without Top-K recomputation."""

    plan = _validate_plan(plan)
    if len(topk_ids) != len(topk_weights):
        raise ValueError("Top-K IDs and weights have different token counts")
    resident_ids = []
    physical_ids = []
    resident_weights = []
    cpu_ids = []
    cpu_weights = []
    for row_ids, row_weights in zip(topk_ids, topk_weights, strict=True):
        ids = tuple(row_ids)
        weights = tuple(float(value) for value in row_weights)
        if len(ids) != len(weights) or not ids:
            raise ValueError("Top-K row shape is invalid")
        if len(set(ids)) != len(ids):
            raise ValueError("Top-K contains duplicate logical experts")
        resident_id_row = []
        physical_id_row = []
        resident_weight_row = []
        cpu_id_row = []
        cpu_weight_row = []
        for expert_id, weight in zip(ids, weights, strict=True):
            if type(expert_id) is not int or not (0 <= expert_id < 512):
                raise ValueError("Top-K logical expert ID is invalid")
            slot = plan.logical_to_physical[expert_id]
            resident = slot >= 0
            resident_id_row.append(expert_id if resident else -1)
            physical_id_row.append(slot if resident else -1)
            resident_weight_row.append(weight if resident else 0.0)
            cpu_id_row.append(-1 if resident else expert_id)
            cpu_weight_row.append(0.0 if resident else weight)
        resident_ids.append(tuple(resident_id_row))
        physical_ids.append(tuple(physical_id_row))
        resident_weights.append(tuple(resident_weight_row))
        cpu_ids.append(tuple(cpu_id_row))
        cpu_weights.append(tuple(cpu_weight_row))
    return SplitTopK(
        tuple(resident_ids),
        tuple(physical_ids),
        tuple(resident_weights),
        tuple(cpu_ids),
        tuple(cpu_weights),
    )


class ModelScopedNativePartialInstaller:
    """Inject only the Q32 owner and post-bind the untouched native runner."""

    def __init__(
        self,
        *,
        qwen_module: Any,
        ascend_fused_moe: Any,
        routed_experts_cls: type,
        physical_runtime_binding: Any,
        expected_layer_count: int,
    ) -> None:
        if not callable(ascend_fused_moe):
            raise TypeError("Ascend FusedMoE factory must be callable")
        if not isinstance(routed_experts_cls, type):
            raise TypeError("Q32 RoutedExperts owner must be a class")
        if not callable(physical_runtime_binding):
            raise TypeError("physical Q32 runtime binding must be callable")
        if type(expected_layer_count) is not int or expected_layer_count <= 0:
            raise ValueError("expected layer count must be positive")
        self._qwen_module = qwen_module
        self._factory = ascend_fused_moe
        self._owner_cls = routed_experts_cls
        self._physical_runtime_binding = physical_runtime_binding
        self._expected_layer_count = expected_layer_count

    @staticmethod
    def _owner_args(plan: SelectiveLoadPlan) -> MappingProxyType:
        return MappingProxyType(
            {
                # Reuse the qualified Round16 physical-Q32 owner verbatim.  Do
                # not leak a Round18-only keyword into its native base class:
                # unknown constructor keywords are forwarded by that owner.
                "round16_selective_plan": plan,
                "round16_resident_logical_ids": plan.resident_logical_ids,
                "round16_logical_to_physical": plan.logical_to_physical,
                "round16_logical_expert_count": plan.logical_expert_count,
                "round16_physical_expert_count": plan.physical_expert_count,
            }
        )

    @contextmanager
    def activate(self, *, plan: SelectiveLoadPlan) -> Iterator[Any]:
        plan = _validate_plan(plan)
        owner_thread = get_ident()
        observed: list[
            tuple[
                str,
                Any,
                NativePhysicalRuntimeReceipt,
                NativeRunnerBindingReceipt,
            ]
        ] = []
        frozen_args = self._owner_args(plan)
        with _CONSTRUCTION_LOCK:
            original = getattr(self._qwen_module, "FusedMoE", None)
            if original is not self._factory:
                raise RuntimeError("Qwen FusedMoE binding changed before construction")

            def scoped_fused_moe(*args: Any, **kwargs: Any) -> Any:
                if get_ident() != owner_thread:
                    raise RuntimeError("Qwen construction escaped the owner thread")
                if kwargs.get("runner_cls") is not None or kwargs.get(
                    "runner_args"
                ) is not None:
                    raise RuntimeError("Round18 refuses a custom MoE runner")
                if kwargs.get("routed_experts_cls") is not None or kwargs.get(
                    "routed_experts_args"
                ) is not None:
                    raise RuntimeError("Qwen call already overrides RoutedExperts")
                forwarded = dict(kwargs)
                forwarded["routed_experts_cls"] = self._owner_cls
                forwarded["routed_experts_args"] = frozen_args
                runner = self._factory(*args, **forwarded)
                physical_receipt = self._physical_runtime_binding(runner)
                receipt = bind_q32_map_to_native_runner(runner, plan)
                observed.append(
                    (
                        str(forwarded.get("prefix", "")),
                        runner,
                        physical_receipt,
                        receipt,
                    )
                )
                return runner

            def acknowledge(native_model: Any) -> bool:
                if native_model is None:
                    raise RuntimeError("native model construction did not complete")
                if len(observed) != self._expected_layer_count:
                    raise RuntimeError(
                        f"expected {self._expected_layer_count} native MoE layers, "
                        f"observed {len(observed)}"
                    )
                prefixes = tuple(
                    prefix
                    for prefix, _runner, _physical_receipt, _receipt in observed
                )
                if len(prefixes) != len(set(prefixes)):
                    raise RuntimeError("duplicate native MoE layer prefixes")
                for _prefix, runner, physical_receipt, receipt in observed:
                    if type(runner) is not runner.round18_native_runner_class:
                        raise RuntimeError("native MoE runner class was replaced")
                    if not physical_receipt.runner_class_preserved:
                        raise RuntimeError("physical Q32 binding replaced the runner")
                    if physical_receipt.physical_slots != 32:
                        raise RuntimeError("physical Q32 runtime capacity changed")
                    if receipt.api != NATIVE_PARTIAL_SEAM_API:
                        raise RuntimeError("native partial binding receipt changed")
                return True

            self._qwen_module.FusedMoE = scoped_fused_moe
            try:
                yield acknowledge
            finally:
                self._qwen_module.FusedMoE = original


__all__ = (
    "ModelScopedNativePartialInstaller",
    "NATIVE_PARTIAL_SEAM_API",
    "NativeRunnerBindingReceipt",
    "NativePhysicalRuntimeReceipt",
    "SplitTopK",
    "bind_q32_map_to_native_runner",
    "rebind_native_physical_quant_method",
    "split_topk_occurrences",
)
