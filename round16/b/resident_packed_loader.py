"""Resident-aware loader contract for packed Qwen routed-expert tensors.

The checkpoint keeps all 512 logical experts in two packed tensors.  This
module deliberately accepts only already-selected Q32 CPU tensors, then calls
the real parameter ``weight_loader`` once per logical expert and projection.
It therefore preserves the upstream TP/load semantics without letting the
packed row number masquerade as a global expert ID for a non-contiguous map.

No torch, safetensors, vLLM, device runtime, or checkpoint is imported here.
The live integration probe owns those dependencies after its source and V1
receipts have been validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .selective_loader import SelectiveLoadPlan


PACKED_GATE_UP_SHAPE = (512, 1280, 2560)
PACKED_DOWN_SHAPE = (512, 2560, 640)
SELECTED_GATE_UP_SHAPE = (32, 1280, 2560)
SELECTED_DOWN_SHAPE = (32, 2560, 640)
INTERMEDIATE_SIZE = 640
PROJECTIONS = ("gate", "up", "down")
RESIDENT_PACKED_LOADER_API = "round16-b:resident-packed-loader:v1"


@dataclass(frozen=True)
class ProjectionLoad:
    logical_expert_id: int
    physical_slot: int
    projection: str

    def __post_init__(self) -> None:
        if type(self.logical_expert_id) is not int or not (
            0 <= self.logical_expert_id < 512
        ):
            raise ValueError("projection load logical expert is outside [0,512)")
        if type(self.physical_slot) is not int or not (0 <= self.physical_slot < 32):
            raise ValueError("projection load physical slot is outside [0,32)")
        if self.projection not in PROJECTIONS:
            raise ValueError("projection load kind is invalid")


def _shape(value: Any, name: str) -> tuple[int, ...]:
    try:
        result = tuple(value.shape)
    except (AttributeError, TypeError) as error:
        raise TypeError(f"{name} does not expose a tensor shape") from error
    if any(type(dimension) is not int or dimension <= 0 for dimension in result):
        raise ValueError(f"{name} has an invalid shape")
    return result


def planned_row_reads(plan: SelectiveLoadPlan) -> tuple[tuple[int, int], ...]:
    """Return ``(physical_slot, logical_id)`` reads in physical storage order."""

    if not isinstance(plan, SelectiveLoadPlan):
        raise TypeError("packed loader requires a SelectiveLoadPlan")
    return tuple(enumerate(plan.resident_logical_ids))


def validate_packed_source_metadata(
    gate_up_shape: Sequence[int], down_shape: Sequence[int]
) -> None:
    if tuple(gate_up_shape) != PACKED_GATE_UP_SHAPE:
        raise ValueError("packed gate/up checkpoint shape is not logical512")
    if tuple(down_shape) != PACKED_DOWN_SHAPE:
        raise ValueError("packed down checkpoint shape is not logical512")


def _weight_loader(parameter: Any, name: str) -> Callable[..., Any]:
    loader = getattr(parameter, "weight_loader", None)
    if not callable(loader):
        raise TypeError(f"{name} has no native weight_loader")
    return loader


def load_selected_packed_rows(
    *,
    owner: Any,
    plan: SelectiveLoadPlan,
    selected_gate_up: Any,
    selected_down: Any,
    gate_up_weight_name: str,
    down_weight_name: str,
) -> tuple[ProjectionLoad, ...]:
    """Load Q32 CPU slices into a raw physical owner through native callbacks.

    ``selected_*`` must already be in physical slot order.  The callback still
    receives the original logical ID so the same immutable ``ResidentPlan``
    controls owner, loader, and the later Hybrid runner.
    """

    if not isinstance(plan, SelectiveLoadPlan):
        raise TypeError("packed loader requires a SelectiveLoadPlan")
    if _shape(selected_gate_up, "selected_gate_up") != SELECTED_GATE_UP_SHAPE:
        raise ValueError("selected gate/up tensor must contain exactly Q32 rows")
    if _shape(selected_down, "selected_down") != SELECTED_DOWN_SHAPE:
        raise ValueError("selected down tensor must contain exactly Q32 rows")
    if type(gate_up_weight_name) is not str or "gate_up_proj" not in gate_up_weight_name:
        raise ValueError("packed gate/up checkpoint identity is invalid")
    if type(down_weight_name) is not str or "down_proj" not in down_weight_name:
        raise ValueError("packed down checkpoint identity is invalid")

    w13 = getattr(owner, "w13_weight", None)
    w2 = getattr(owner, "w2_weight", None)
    layer_name = getattr(owner, "layer_name", None)
    if type(layer_name) is not str or not layer_name:
        raise ValueError("packed loader owner layer identity is missing")
    # RoutedExperts.load_weights first maps a checkpoint source name onto the
    # target parameter name, then invokes the parameter callback.  We bypass
    # only its logical-512 enumeration, so this name translation must remain.
    target_w13_name = f"{layer_name}.w13_weight"
    target_w2_name = f"{layer_name}.w2_weight"
    w13_loader = _weight_loader(w13, "w13_weight")
    w2_loader = _weight_loader(w2, "w2_weight")
    coverage: list[ProjectionLoad] = []

    for physical_slot, logical_id in planned_row_reads(plan):
        gate_up_row = selected_gate_up[physical_slot]
        gate = gate_up_row[:INTERMEDIATE_SIZE]
        up = gate_up_row[INTERMEDIATE_SIZE:]
        down = selected_down[physical_slot]
        calls = (
            (w13_loader, w13, gate, target_w13_name, "w1", "gate"),
            (w13_loader, w13, up, target_w13_name, "w3", "up"),
            (w2_loader, w2, down, target_w2_name, "w2", "down"),
        )
        for loader, parameter, source, weight_name, shard_id, projection in calls:
            success = loader(
                param=parameter,
                loaded_weight=source,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=logical_id,
                return_success=True,
            )
            if success is not True:
                raise RuntimeError(
                    f"native loader rejected resident logical expert {logical_id} "
                    f"projection {projection}"
                )
            coverage.append(ProjectionLoad(logical_id, physical_slot, projection))

    validate_projection_coverage(coverage, plan)
    return tuple(coverage)


def validate_projection_coverage(
    coverage: Iterable[ProjectionLoad], plan: SelectiveLoadPlan
) -> None:
    if not isinstance(plan, SelectiveLoadPlan):
        raise TypeError("coverage validation requires a SelectiveLoadPlan")
    entries = tuple(coverage)
    expected = {
        (logical_id, physical_slot, projection)
        for physical_slot, logical_id in planned_row_reads(plan)
        for projection in PROJECTIONS
    }
    actual = {
        (entry.logical_expert_id, entry.physical_slot, entry.projection)
        for entry in entries
        if isinstance(entry, ProjectionLoad)
    }
    if len(entries) != 96 or actual != expected:
        raise ValueError("resident projection coverage must be exact Q32 x 3")


__all__ = (
    "PACKED_DOWN_SHAPE",
    "PACKED_GATE_UP_SHAPE",
    "PROJECTIONS",
    "ProjectionLoad",
    "RESIDENT_PACKED_LOADER_API",
    "SELECTED_DOWN_SHAPE",
    "SELECTED_GATE_UP_SHAPE",
    "load_selected_packed_rows",
    "planned_row_reads",
    "validate_packed_source_metadata",
    "validate_projection_coverage",
)
