"""Bounded offline analysis of a fresh Round18 native route trace.

The dynamic policies operate on sequential Top-10 token rows.  Their results
are cache-policy oracles; a prefill invocation can require more distinct
experts than the configured capacity, so the results do not establish that a
native fused MoE call can run from that capacity.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Literal, Mapping


BF16_EXPERT_BYTES = 9_830_400
CAPACITIES = (32, 64, 96, 128)
EXPECTED_LAYERS = 48
EXPECTED_TOP_K = 10
EXPECTED_LOGICAL_EXPERTS = 512
MAX_ROUTE_EVENTS = 20_000
MAX_TRACE_BYTES = 256 * 1024 * 1024


Phase = Literal["Prefill", "Decode"]
Policy = Literal["static_frequency", "lru", "belady"]


@dataclass(frozen=True)
class _Invocation:
    event_index: int
    layer: int
    phase: Phase
    rows: tuple[tuple[int, ...], ...]

    @property
    def working_set(self) -> frozenset[int]:
        return frozenset(expert for row in self.rows for expert in row)


@dataclass(frozen=True)
class _Request:
    workload: str
    completion_tokens: int
    invocations: tuple[_Invocation, ...]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_trace(path: Path) -> list[dict[str, Any]]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ValueError(f"cannot stat route trace {path}: {error}") from error
    if size > MAX_TRACE_BYTES:
        raise ValueError(f"route trace exceeds {MAX_TRACE_BYTES} byte bound")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                if len(rows) >= MAX_ROUTE_EVENTS:
                    raise ValueError(
                        f"route trace exceeds {MAX_ROUTE_EVENTS} event bound"
                    )
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"route trace line {line_number} is invalid JSON"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(
                        f"route trace line {line_number} is not an object"
                    )
                if row.get("event_index") != len(rows):
                    raise ValueError("route trace event indices are not contiguous")
                rows.append(row)
    except OSError as error:
        raise ValueError(f"cannot read route trace {path}: {error}") from error
    return rows


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_invocation(raw: Mapping[str, Any], *, logical_experts: int) -> _Invocation:
    event_index = raw.get("event_index")
    if type(event_index) is not int or event_index < 0:
        raise ValueError("route event_index must be a non-negative integer")
    if raw.get("schema") != 1 or raw.get("kind") != "native_route_event":
        raise ValueError(f"route event {event_index} has an unsupported schema")
    layer = raw.get("logical_layer")
    if type(layer) is not int or not 0 <= layer < EXPECTED_LAYERS:
        raise ValueError(f"route event {event_index} has an invalid logical layer")
    if raw.get("top_k") != EXPECTED_TOP_K:
        raise ValueError("route trace changed the frozen Top-10 contract")
    token_rows = _positive_int(raw.get("token_rows"), "token_rows")
    raw_rows = raw.get("topk_ids")
    if not isinstance(raw_rows, list) or len(raw_rows) != token_rows:
        raise ValueError(f"route event {event_index} token_rows disagree with topk_ids")
    rows: list[tuple[int, ...]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, list) or len(raw_row) != EXPECTED_TOP_K:
            raise ValueError("route trace changed the frozen Top-10 contract")
        row = tuple(raw_row)
        if any(
            type(expert) is not int or not 0 <= expert < logical_experts
            for expert in row
        ):
            raise ValueError(f"route event {event_index} has an invalid expert ID")
        if len(set(row)) != EXPECTED_TOP_K:
            raise ValueError(f"route event {event_index} Top-10 row is not distinct")
        rows.append(row)
    raw_weights = raw.get("topk_weights")
    if not isinstance(raw_weights, list) or len(raw_weights) != token_rows:
        raise ValueError(f"route event {event_index} topk weight rows disagree")
    if any(not isinstance(row, list) or len(row) != EXPECTED_TOP_K for row in raw_weights):
        raise ValueError(f"route event {event_index} topk weight width disagrees")
    return _Invocation(
        event_index=event_index,
        layer=layer,
        phase="Prefill" if token_rows > 1 else "Decode",
        rows=tuple(rows),
    )


def _validate_header(result: Mapping[str, Any], event_count: int) -> None:
    if result.get("schema") != 1:
        raise ValueError("route result has an unsupported schema")
    if result.get("status") != "PASS_ROUND18_FRESH_NATIVE_ROUTE_TRACE":
        raise ValueError("route result is not a passing fresh Round18 trace")
    if result.get("logical_experts") != EXPECTED_LOGICAL_EXPERTS:
        raise ValueError("route result logical expert count is not 512")
    if result.get("top_k") != EXPECTED_TOP_K:
        raise ValueError("route result changed the frozen Top-10 contract")
    if result.get("layer_count") != EXPECTED_LAYERS:
        raise ValueError("route result layer count is not 48")
    if result.get("route_event_count") != event_count:
        raise ValueError("route result event count disagrees with JSONL")


def _load_requests(
    result: Mapping[str, Any], raw_events: list[dict[str, Any]]
) -> dict[str, _Request]:
    _validate_header(result, len(raw_events))
    raw_requests = result.get("requests")
    if not isinstance(raw_requests, list) or len(raw_requests) != 3:
        raise ValueError("route result must contain W1/W2/W4 requests")
    parsed_events = [
        _parse_invocation(event, logical_experts=EXPECTED_LOGICAL_EXPERTS)
        for event in raw_events
    ]
    requests: dict[str, _Request] = {}
    expected_start = 0
    for raw_request in raw_requests:
        if not isinstance(raw_request, dict):
            raise ValueError("route result request entry must be an object")
        workload = raw_request.get("workload")
        if workload not in ("W1", "W2", "W4") or workload in requests:
            raise ValueError("route result must contain unique W1/W2/W4 requests")
        start = raw_request.get("route_event_start")
        end = raw_request.get("route_event_end_exclusive")
        count = raw_request.get("route_event_count")
        if (
            type(start) is not int
            or type(end) is not int
            or type(count) is not int
            or start != expected_start
            or end <= start
            or end > len(parsed_events)
            or count != end - start
        ):
            raise ValueError("route result has non-contiguous request event ranges")
        invocations = tuple(parsed_events[start:end])
        layers = {event.layer for event in invocations}
        if layers != set(range(EXPECTED_LAYERS)):
            raise ValueError(f"{workload} request does not contain all 48 layers")
        for phase in ("Prefill", "Decode"):
            phase_layers = {
                event.layer for event in invocations if event.phase == phase
            }
            if phase_layers != set(range(EXPECTED_LAYERS)):
                raise ValueError(f"{workload} {phase} does not contain all 48 layers")
        observed_rows = sum(len(event.rows) for event in invocations)
        if raw_request.get("route_token_rows") != observed_rows:
            raise ValueError(f"{workload} route_token_rows disagrees with JSONL")
        completion_tokens = _positive_int(
            raw_request.get("completion_tokens"), "completion_tokens"
        )
        requests[workload] = _Request(
            workload=workload,
            completion_tokens=completion_tokens,
            invocations=invocations,
        )
        expected_start = end
    if expected_start != len(parsed_events) or set(requests) != {"W1", "W2", "W4"}:
        raise ValueError("request event ranges do not cover the complete trace")
    return requests


def _frequency_plan(
    invocations: Iterable[_Invocation], capacity: int
) -> dict[int, tuple[int, ...]]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for invocation in invocations:
        counts[invocation.layer].update(
            expert for row in invocation.rows for expert in row
        )
    return {
        layer: tuple(
            expert
            for expert, _count in sorted(
                counts[layer].items(), key=lambda item: (-item[1], item[0])
            )[:capacity]
        )
        for layer in range(EXPECTED_LAYERS)
    }


def _rows_by_layer(
    invocations: Iterable[_Invocation], phase: Phase
) -> dict[int, list[tuple[int, ...]]]:
    rows: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for invocation in invocations:
        if invocation.phase == phase:
            rows[invocation.layer].extend(invocation.rows)
    return dict(rows)


def _working_sets_by_layer(
    invocations: Iterable[_Invocation], phase: Phase
) -> dict[int, list[frozenset[int]]]:
    working_sets: dict[int, list[frozenset[int]]] = defaultdict(list)
    for invocation in invocations:
        if invocation.phase == phase:
            working_sets[invocation.layer].append(invocation.working_set)
    return dict(working_sets)


def _next_positions(rows: list[tuple[int, ...]]) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        for expert in row:
            positions[expert].append(index)
    return positions


def _next_use(positions: Mapping[int, list[int]], expert: int, index: int) -> int:
    for candidate in positions.get(expert, ()):
        if candidate > index:
            return candidate
    return 1 << 60


def _simulate_layer(
    rows: list[tuple[int, ...]],
    *,
    invocation_working_sets: list[frozenset[int]],
    resident_plan: tuple[int, ...],
    capacity: int,
    policy: Policy,
) -> tuple[int, int]:
    resident = set(resident_plan)
    misses = hits = 0
    if policy == "static_frequency":
        for requested in invocation_working_sets:
            hits += len(requested & resident)
            misses += len(requested - resident)
        return misses, hits

    last_used = {expert: -1 for expert in resident}
    positions = _next_positions(rows) if policy == "belady" else {}
    for index, row in enumerate(rows):
        requested = set(row)
        hits += len(requested & resident)
        missing = sorted(requested - resident)
        misses += len(missing)
        for expert in missing:
            if len(resident) >= capacity:
                candidates = resident - requested
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
    return misses, hits


def _policy_report(
    *,
    rows_by_layer: Mapping[int, list[tuple[int, ...]]],
    working_sets_by_layer: Mapping[int, list[frozenset[int]]],
    plan: Mapping[int, tuple[int, ...]],
    capacity: int,
    policy: Policy,
    completion_tokens: int,
    routed_model_tokens: int,
) -> dict[str, Any]:
    per_layer: dict[str, dict[str, int]] = {}
    misses = hits = 0
    for layer in range(EXPECTED_LAYERS):
        layer_misses, layer_hits = _simulate_layer(
            rows_by_layer[layer],
            invocation_working_sets=working_sets_by_layer[layer],
            resident_plan=plan[layer],
            capacity=capacity,
            policy=policy,
        )
        misses += layer_misses
        hits += layer_hits
        per_layer[str(layer)] = {
            "miss_experts": layer_misses,
            "hit_experts": layer_hits,
        }
    bf16_bytes = misses * BF16_EXPERT_BYTES
    return {
        "miss_experts": misses,
        "hit_experts": hits,
        "bf16_bytes": bf16_bytes,
        "bf16_bytes_per_generated_output_token": bf16_bytes / completion_tokens,
        "bf16_bytes_per_routed_model_token": bf16_bytes / routed_model_tokens,
        "per_layer": per_layer,
    }


def _phase_report(
    request: _Request,
    *,
    phase: Phase,
    capacity: int,
    plan: Mapping[int, tuple[int, ...]],
) -> dict[str, Any]:
    invocations = [
        invocation for invocation in request.invocations if invocation.phase == phase
    ]
    working_sets = [len(invocation.working_set) for invocation in invocations]
    rows_by_layer = _rows_by_layer(request.invocations, phase)
    working_sets_by_layer = _working_sets_by_layer(request.invocations, phase)
    layer_row_counts = {layer: len(rows) for layer, rows in rows_by_layer.items()}
    if set(layer_row_counts) != set(range(EXPECTED_LAYERS)):
        raise ValueError(f"{request.workload} {phase} lacks routed rows for a layer")
    distinct_counts = set(layer_row_counts.values())
    if len(distinct_counts) != 1:
        raise ValueError(
            f"{request.workload} {phase} routed model token count differs by layer"
        )
    routed_model_tokens = distinct_counts.pop()
    policies = {
        policy: _policy_report(
            rows_by_layer=rows_by_layer,
            working_sets_by_layer=working_sets_by_layer,
            plan=plan,
            capacity=capacity,
            policy=policy,
            completion_tokens=request.completion_tokens,
            routed_model_tokens=routed_model_tokens,
        )
        for policy in ("static_frequency", "lru", "belady")
    }
    return {
        "generated_output_tokens": request.completion_tokens,
        "routed_model_tokens": routed_model_tokens,
        "native_invocations": len(invocations),
        "invocation_working_set": {
            "fit_ratio": sum(size <= capacity for size in working_sets)
            / len(working_sets),
            "min": min(working_sets),
            "median": float(median(working_sets)),
            "max": max(working_sets),
        },
        "policies": policies,
    }


def analyze_native_route_working_set(
    route_result_path: str | Path, route_trace_path: str | Path
) -> dict[str, Any]:
    """Analyze one passing fresh W1/W2/W4 trace without touching runtime state."""

    result_path = Path(route_result_path)
    trace_path = Path(route_trace_path)
    result = _read_json_object(result_path)
    raw_events = _read_trace(trace_path)
    requests = _load_requests(result, raw_events)
    by_capacity: dict[str, Any] = {}
    for capacity in CAPACITIES:
        plan = _frequency_plan(requests["W1"].invocations, capacity)
        by_capacity[f"Q{capacity}"] = {
            "w1_static_frequency_plan": {
                str(layer): list(plan[layer]) for layer in range(EXPECTED_LAYERS)
            },
            "holdouts": {
                workload: {
                    phase: _phase_report(
                        requests[workload],
                        phase=phase,
                        capacity=capacity,
                        plan=plan,
                    )
                    for phase in ("Prefill", "Decode")
                }
                for workload in ("W2", "W4")
            },
        }
    return {
        "schema": 1,
        "status": "PASS_ROUND18_NATIVE_ROUTE_ANALYSIS",
        "source_status": result["status"],
        "development_workload": "W1",
        "holdout_workloads": ["W2", "W4"],
        "capacities": list(CAPACITIES),
        "layer_count": EXPECTED_LAYERS,
        "top_k": EXPECTED_TOP_K,
        "logical_experts": EXPECTED_LOGICAL_EXPERTS,
        "bf16_expert_bytes": BF16_EXPERT_BYTES,
        "by_capacity": by_capacity,
        "interpretation": {
            "static_frequency": (
                "Per-layer W1 development frequency plan evaluated unchanged on "
                "W2/W4 holdouts; misses are distinct nonresident experts per "
                "native invocation."
            ),
            "dynamic_initial_residency": "W1_STATIC_FREQUENCY_PLAN",
            "dynamic_policy_scope": (
                "SEQUENTIAL_TOKEN_ROW_ORACLE_NOT_NATIVE_SINGLE_CALL_RUNNABLE"
            ),
            "dynamic_policy_detail": (
                "LRU and Belady replay Top-10 token rows sequentially from the W1 "
                "plan; Belady is the optimistic sequential replacement bound. A "
                "native invocation whose working set exceeds Q cannot be made "
                "runnable by these oracle results."
            ),
            "transfer_accounting": (
                "One miss expert equals 9,830,400 BF16 payload bytes; padding, "
                "metadata, conversion, publication, and overlap are excluded."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-result", required=True, type=Path)
    parser.add_argument("--route-trace", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = analyze_native_route_working_set(args.route_result, args.route_trace)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "BF16_EXPERT_BYTES",
    "CAPACITIES",
    "analyze_native_route_working_set",
    "main",
)
