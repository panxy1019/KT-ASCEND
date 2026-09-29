"""Immutable Top-10 routing split and single-application merge."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

from round16.common.contracts import RouteBatch


LOGICAL_EXPERT_COUNT = 512
ROUTER_TOP_K = 10
COMBINE_POLICY = "routed_scaled_plus_shared"


@dataclass(frozen=True)
class RouteOccurrence:
    token_index: int
    topk_index: int
    logical_expert_id: int
    weight: float
    npu_slot: int | None
    cpu_logical_expert_id: int | None

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.token_index, self.topk_index, self.logical_expert_id)


@dataclass(frozen=True)
class HybridRoutePlan:
    routes: RouteBatch
    resident_logical_ids: tuple[int, ...]
    npu_routes: tuple[RouteOccurrence, ...]
    cpu_routes: tuple[RouteOccurrence, ...]
    logical_expert_count: int = LOGICAL_EXPERT_COUNT


def _resident_map(
    resident_logical_ids: Sequence[int], logical_expert_count: int
) -> tuple[tuple[int, ...], dict[int, int]]:
    resident = tuple(resident_logical_ids)
    if any(type(expert_id) is not int for expert_id in resident):
        raise ValueError("resident logical expert ids must be integers")
    if len(resident) != len(set(resident)):
        raise ValueError("resident logical expert ids must be unique")
    if any(expert_id < 0 or expert_id >= logical_expert_count for expert_id in resident):
        raise ValueError("resident logical expert id is out of range")
    return resident, {expert_id: slot for slot, expert_id in enumerate(resident)}


def _freeze_route_ids(
    topk_ids: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    try:
        frozen_ids = tuple(tuple(row) for row in topk_ids)
    except TypeError as error:
        raise ValueError("route logical expert ids must be integer sequences") from error
    if any(type(expert_id) is not int for row in frozen_ids for expert_id in row):
        raise ValueError("route logical expert ids must be integers")
    return frozen_ids


def _validate_frozen_route_ids(routes: RouteBatch) -> None:
    if type(routes.topk_ids) is not tuple or any(
        type(row) is not tuple for row in routes.topk_ids
    ):
        raise ValueError("frozen route logical expert ids must be deeply frozen tuples")
    if any(
        type(expert_id) is not int
        for row in routes.topk_ids
        for expert_id in row
    ):
        raise ValueError("route logical expert ids must be integers")


def plan_routes(
    topk_ids: RouteBatch | Sequence[Sequence[int]],
    topk_weights: Sequence[Sequence[float]] | None = None,
    *,
    resident_logical_ids: Sequence[int],
    valid_rows: Sequence[bool] | None = None,
    logical_expert_count: int = LOGICAL_EXPERT_COUNT,
) -> HybridRoutePlan:
    """Freeze raw routes once, then split logical IDs from physical slots."""

    if type(logical_expert_count) is not int:
        raise ValueError("logical_expert_count must be an integer")
    if logical_expert_count <= 0:
        raise ValueError("logical_expert_count must be positive")
    resident, physical_by_logical = _resident_map(
        resident_logical_ids, logical_expert_count
    )
    if isinstance(topk_ids, RouteBatch):
        if topk_weights is not None or valid_rows is not None:
            raise ValueError("a frozen RouteBatch cannot be frozen or overridden again")
        routes = topk_ids
        _validate_frozen_route_ids(routes)
    else:
        if topk_weights is None:
            raise ValueError("topk_weights are required for raw route ids")
        frozen_ids = _freeze_route_ids(topk_ids)
        routes = RouteBatch.freeze(frozen_ids, topk_weights, valid_rows=valid_rows)
    if routes.token_count and routes.top_k != ROUTER_TOP_K:
        raise ValueError(f"Round16-B requires Top-{ROUTER_TOP_K} routes")
    npu: list[RouteOccurrence] = []
    cpu: list[RouteOccurrence] = []
    for token_index, (ids, weights, valid) in enumerate(
        zip(routes.topk_ids, routes.topk_weights, routes.valid_rows)
    ):
        if not valid:
            continue
        for topk_index, (expert_id, weight) in enumerate(zip(ids, weights)):
            if expert_id == -1:
                continue
            if expert_id < 0 or expert_id >= logical_expert_count:
                raise ValueError(
                    f"logical expert id {expert_id} is outside [0, {logical_expert_count})"
                )
            slot = physical_by_logical.get(expert_id)
            if slot is None:
                cpu.append(
                    RouteOccurrence(
                        token_index,
                        topk_index,
                        expert_id,
                        weight,
                        None,
                        expert_id,
                    )
                )
            else:
                npu.append(
                    RouteOccurrence(
                        token_index,
                        topk_index,
                        expert_id,
                        weight,
                        slot,
                        None,
                    )
                )
    return HybridRoutePlan(routes, resident, tuple(npu), tuple(cpu), logical_expert_count)


def _vector_width(
    plan: HybridRoutePlan,
    npu_outputs: Mapping[tuple[int, int, int], Sequence[float]],
    cpu_outputs: Mapping[tuple[int, int, int], Sequence[float]],
    shared_output: Sequence[Sequence[float]] | None,
) -> int:
    if shared_output is not None and len(shared_output) != plan.routes.token_count:
        raise ValueError("shared output row count must match the route batch")
    candidates = list(npu_outputs.values()) + list(cpu_outputs.values())
    if shared_output is not None:
        candidates.extend(shared_output)
    if not candidates:
        return 0
    width = len(candidates[0])
    if any(len(vector) != width for vector in candidates):
        raise ValueError("all expert and shared output vectors must have equal width")
    return width


def merge_route_outputs(
    plan: HybridRoutePlan,
    npu_outputs: Mapping[tuple[int, int, int], Sequence[float]],
    cpu_outputs: Mapping[tuple[int, int, int], Sequence[float]],
    *,
    shared_output: Sequence[Sequence[float]] | None = None,
    scaling: float = 1.0,
    combine_policy: str = COMBINE_POLICY,
) -> tuple[tuple[float, ...], ...]:
    """Weight routed experts once, scale their sum, then add shared once."""

    if combine_policy != COMBINE_POLICY:
        raise ValueError(f"unsupported combine policy: {combine_policy}")
    try:
        routed_scaling_factor = float(scaling)
    except (TypeError, ValueError) as error:
        raise ValueError("routed scaling factor must be finite") from error
    if not isfinite(routed_scaling_factor):
        raise ValueError("routed scaling factor must be finite")
    expected_npu = {route.key for route in plan.npu_routes}
    expected_cpu = {route.key for route in plan.cpu_routes}
    if set(npu_outputs) != expected_npu or set(cpu_outputs) != expected_cpu:
        raise ValueError("expert outputs must match every routed occurrence exactly")
    width = _vector_width(plan, npu_outputs, cpu_outputs, shared_output)
    merged = [[0.0] * width for _ in range(plan.routes.token_count)]
    for route in (*plan.npu_routes, *plan.cpu_routes):
        branch = npu_outputs if route.npu_slot is not None else cpu_outputs
        vector = branch[route.key]
        row = merged[route.token_index]
        for index, value in enumerate(vector):
            row[index] += float(value) * route.weight
    for row in merged:
        for index, value in enumerate(row):
            row[index] = value * routed_scaling_factor
    if shared_output is not None:
        for row, shared in zip(merged, shared_output):
            for index, value in enumerate(shared):
                row[index] += float(value)
    return tuple(tuple(row) for row in merged)


__all__ = (
    "HybridRoutePlan",
    "COMBINE_POLICY",
    "LOGICAL_EXPERT_COUNT",
    "ROUTER_TOP_K",
    "RouteOccurrence",
    "merge_route_outputs",
    "plan_routes",
)
