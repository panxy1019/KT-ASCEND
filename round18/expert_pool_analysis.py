"""Deterministic offline working-set analysis for a per-layer expert pool.

The simulator counts expert transfers, not routed occurrences.  All experts
needed by one layer invocation must fit simultaneously; repeated occurrences
of the same expert in that invocation therefore cause one load at most.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Literal


PoolPolicy = Literal["static_frequency", "lru", "belady"]
DEFAULT_EXPERT_BYTES = 9_830_400  # 9.375 MiB BF16 gate/up/down payload.


@dataclass(frozen=True)
class RouteEvent:
    layer: int
    experts: tuple[int, ...]
    token: int | None = None

    def __post_init__(self) -> None:
        if type(self.layer) is not int or self.layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if not self.experts:
            raise ValueError("a route event must contain at least one expert")
        if any(type(value) is not int or value < 0 for value in self.experts):
            raise ValueError("expert IDs must be non-negative integers")


@dataclass(frozen=True)
class LayerPoolResult:
    layer: int
    events: int
    occurrences: int
    unique_requests: int
    hits: int
    miss_loads: int
    resident_final: tuple[int, ...]


@dataclass(frozen=True)
class PoolResult:
    policy: PoolPolicy
    capacity: int
    total_events: int
    total_occurrences: int
    unique_requests: int
    hits: int
    miss_loads: int
    bytes_loaded: int
    initial_resident: dict[int, tuple[int, ...]]
    per_layer: dict[int, LayerPoolResult]

    @property
    def occurrence_hit_rate(self) -> float:
        return self.hits / self.unique_requests if self.unique_requests else 0.0


def _group_by_layer(events: Iterable[RouteEvent]) -> dict[int, list[RouteEvent]]:
    grouped: dict[int, list[RouteEvent]] = defaultdict(list)
    for event in events:
        grouped[event.layer].append(event)
    return dict(sorted(grouped.items()))


def _validate_capacity(events: list[RouteEvent], capacity: int) -> None:
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    widest = max((len(set(event.experts)) for event in events), default=0)
    if widest > capacity:
        raise ValueError(
            f"event working set {widest} exceeds pool capacity {capacity}"
        )


def _simulate_static(events: list[RouteEvent], capacity: int) -> LayerPoolResult:
    counts = Counter(expert for event in events for expert in event.experts)
    resident = {
        expert
        for expert, _count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:capacity]
    }
    misses = hits = unique_requests = occurrences = 0
    for event in events:
        requested = set(event.experts)
        occurrences += len(event.experts)
        unique_requests += len(requested)
        event_hits = len(requested & resident)
        hits += event_hits
        misses += len(requested) - event_hits
    return LayerPoolResult(
        layer=events[0].layer,
        events=len(events),
        occurrences=occurrences,
        unique_requests=unique_requests,
        hits=hits,
        miss_loads=misses,
        resident_final=tuple(sorted(resident)),
    )


def _future_positions(events: list[RouteEvent]) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        for expert in set(event.experts):
            positions[expert].append(index)
    return positions


def _next_use(positions: dict[int, list[int]], expert: int, index: int) -> int:
    for candidate in positions.get(expert, ()):
        if candidate > index:
            return candidate
    return 1 << 60


def _simulate_dynamic(
    events: list[RouteEvent],
    capacity: int,
    policy: Literal["lru", "belady"],
    initial_resident: Iterable[int] = (),
) -> LayerPoolResult:
    resident = set(initial_resident)
    if len(resident) > capacity:
        raise ValueError("initial resident set exceeds pool capacity")
    last_used: dict[int, int] = {expert: -1 for expert in resident}
    positions = _future_positions(events) if policy == "belady" else {}
    misses = hits = unique_requests = occurrences = 0

    for index, event in enumerate(events):
        requested = set(event.experts)
        occurrences += len(event.experts)
        unique_requests += len(requested)
        hits += len(requested & resident)
        missing = sorted(requested - resident)
        misses += len(missing)
        for expert in missing:
            if len(resident) >= capacity:
                candidates = resident - requested
                if not candidates:
                    raise ValueError(
                        "event working set cannot be made resident at the configured capacity"
                    )
                if policy == "lru":
                    victim = min(
                        candidates, key=lambda value: (last_used.get(value, -1), value)
                    )
                else:
                    victim = max(
                        candidates,
                        key=lambda value: (_next_use(positions, value, index), value),
                    )
                resident.remove(victim)
                last_used.pop(victim, None)
            resident.add(expert)
        for expert in requested:
            last_used[expert] = index

    return LayerPoolResult(
        layer=events[0].layer,
        events=len(events),
        occurrences=occurrences,
        unique_requests=unique_requests,
        hits=hits,
        miss_loads=misses,
        resident_final=tuple(sorted(resident)),
    )


def simulate_pool(
    events: Iterable[RouteEvent],
    *,
    capacity: int,
    policy: PoolPolicy,
    expert_bytes: int = DEFAULT_EXPERT_BYTES,
    initial_resident: dict[int, Iterable[int]] | None = None,
) -> PoolResult:
    materialized = list(events)
    if policy not in ("static_frequency", "lru", "belady"):
        raise ValueError(f"unsupported pool policy: {policy}")
    if type(expert_bytes) is not int or expert_bytes <= 0:
        raise ValueError("expert_bytes must be a positive integer")
    _validate_capacity(materialized, capacity)

    initial = {
        int(layer): tuple(sorted(set(experts)))
        for layer, experts in (initial_resident or {}).items()
    }
    per_layer: dict[int, LayerPoolResult] = {}
    for layer, layer_events in _group_by_layer(materialized).items():
        if policy == "static_frequency":
            result = _simulate_static(layer_events, capacity)
        else:
            result = _simulate_dynamic(
                layer_events, capacity, policy, initial.get(layer, ())
            )
        per_layer[layer] = result

    return PoolResult(
        policy=policy,
        capacity=capacity,
        total_events=sum(item.events for item in per_layer.values()),
        total_occurrences=sum(item.occurrences for item in per_layer.values()),
        unique_requests=sum(item.unique_requests for item in per_layer.values()),
        hits=sum(item.hits for item in per_layer.values()),
        miss_loads=sum(item.miss_loads for item in per_layer.values()),
        bytes_loaded=sum(item.miss_loads for item in per_layer.values())
        * expert_bytes,
        initial_resident=initial,
        per_layer=per_layer,
    )


def frequency_resident_plan(
    events: Iterable[RouteEvent], capacity: int
) -> dict[int, tuple[int, ...]]:
    """Return the deterministic per-layer frequency warm-start plan."""

    materialized = list(events)
    _validate_capacity(materialized, capacity)
    result: dict[int, tuple[int, ...]] = {}
    for layer, layer_events in _group_by_layer(materialized).items():
        counts = Counter(expert for event in layer_events for expert in event.experts)
        result[layer] = tuple(
            expert
            for expert, _count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[:capacity]
        )
    return result


def load_round9_route_tape(path: str | Path) -> list[RouteEvent]:
    """Restore per-row logical IDs from the earlier host-only route tape.

    This loader is for historical sensitivity analysis only.  A Round18
    promotion decision still requires a fresh W1/W2/W4 trace from the frozen
    Native-Hybrid identity.
    """

    events: list[RouteEvent] = []
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("kind") != "group":
                continue
            logical = item.get("logical_experts")
            remapped = item.get("remapped_ids")
            rows = item.get("rows")
            layer = item.get("logical_layer")
            if not isinstance(logical, list) or not isinstance(remapped, list):
                raise ValueError(f"invalid route group at line {line_number}")
            if not isinstance(rows, list) or len(rows) != len(remapped):
                raise ValueError(f"route rows disagree at line {line_number}")
            for token, slot_ids in zip(rows, remapped, strict=True):
                try:
                    experts = tuple(int(logical[int(slot)]) for slot in slot_ids)
                except (IndexError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid remapped expert at line {line_number}"
                    ) from error
                events.append(RouteEvent(layer=int(layer), experts=experts, token=int(token)))
    if not events:
        raise ValueError("route tape contains no group events")
    return events


__all__ = [
    "DEFAULT_EXPERT_BYTES",
    "LayerPoolResult",
    "PoolResult",
    "RouteEvent",
    "frequency_resident_plan",
    "load_round9_route_tape",
    "simulate_pool",
]
