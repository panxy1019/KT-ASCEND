"""Join Round16 Host observations to native TaskQueue events and close W2 cost."""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


def load_jsonl(
    path: Path,
    *,
    allow_terminal_partial: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    terminal_partial_lines = 0
    for number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            terminal = number == len(lines) and not line.endswith(("\n", "\r"))
            recognizable = line.startswith('{"schema":1,"event":"')
            if allow_terminal_partial and terminal and recognizable:
                terminal_partial_lines = 1
                continue
            raise ValueError(f"invalid JSONL {path}:{number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at {path}:{number}")
        rows.append(value)
    if diagnostics is not None:
        diagnostics.update(
            {
                "complete_records": len(rows),
                "terminal_partial_lines": terminal_partial_lines,
                "terminal_partial_scope": (
                    "TERMINAL_FILE_TAIL_EXCLUDED_FROM_JOIN"
                    if terminal_partial_lines
                    else "NONE"
                ),
            }
        )
    return rows


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty series")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary_ns(values: Iterable[int]) -> dict[str, float | int]:
    materialized = list(values)
    return {
        "count": len(materialized),
        "median_ms": statistics.median(materialized) / 1e6,
        "p05_ms": percentile(materialized, 0.05) / 1e6,
        "p95_ms": percentile(materialized, 0.95) / 1e6,
        "mean_ms": statistics.fmean(materialized) / 1e6,
        "sum_ms": sum(materialized) / 1e6,
    }


def native_indexes(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]]]:
    enqueue = sorted((r for r in rows if r.get("event") == "enqueue"), key=lambda r: r["now_ns"])
    completed = {r["sequence"]: r for r in rows if r.get("event") == "task_complete"}
    sync_begin = {r["sequence"]: r for r in rows if r.get("event") == "sync_begin"}
    sync_end = {r["sequence"]: r for r in rows if r.get("event") == "sync_end"}
    if set(sync_begin) != set(sync_end):
        raise ValueError("native sync begin/end identity mismatch")
    sync_pairs = sorted(((sync_begin[k], sync_end[k]) for k in sync_begin), key=lambda pair: pair[0]["now_ns"])
    return enqueue, completed, sync_pairs


def containing_one(rows: list[Any], timestamps: list[int], start: int, end: int) -> Any:
    lower = bisect.bisect_left(timestamps, start)
    upper = bisect.bisect_right(timestamps, end)
    if upper - lower != 1:
        raise ValueError(
            f"expected one event inside [{start},{end}], found {upper - lower}"
        )
    return rows[lower]


def request_modes(round13_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in round13_rows:
        if row.get("status") != "ROUND13_P0_BATCH":
            continue
        result[row["request_id"]].append(row)
    for rows in result.values():
        rows.sort(key=lambda row: row["started_ns"])
    return result


def join_calls(
    observer_rows: list[dict[str, Any]],
    native_rows: list[dict[str, Any]],
    round13_rows: list[dict[str, Any]],
    *,
    request_marker: str | None = None,
) -> list[dict[str, Any]]:
    enqueues, completed, sync_pairs = native_indexes(native_rows)
    enqueue_times = [row["now_ns"] for row in enqueues]
    sync_times = [pair[0]["now_ns"] for pair in sync_pairs]
    modes = request_modes(round13_rows)
    by_parent: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in observer_rows:
        component = row.get("component")
        if component == "PARENT_ROUTED_EXPERT":
            by_parent[row["span_id"]][component] = row
        elif component in {"HOST_DISPATCH", "HOST_SYNC", "HOST_MERGE"}:
            by_parent[row["parent_span_id"]][component] = row
    joined = []
    for parent_id, components in by_parent.items():
        required = {"PARENT_ROUTED_EXPERT", "HOST_DISPATCH", "HOST_SYNC", "HOST_MERGE"}
        if set(components) != required:
            raise ValueError(f"incomplete Host expert call {parent_id}: {set(components)}")
        parent = components["PARENT_ROUTED_EXPERT"]
        if request_marker is not None and request_marker not in parent["request_id"]:
            continue
        dispatch = components["HOST_DISPATCH"]
        host_sync = components["HOST_SYNC"]
        enqueue = containing_one(
            enqueues,
            enqueue_times,
            dispatch["start_ns"],
            dispatch["end_ns"],
        )
        task = completed.get(enqueue["sequence"])
        if task is None:
            raise ValueError(f"native task {enqueue['sequence']} has no completion")
        sync_begin, sync_end = containing_one(
            sync_pairs,
            sync_times,
            host_sync["start_ns"],
            host_sync["end_ns"],
        )
        request_batches = modes.get(parent["request_id"], [])
        # A parent is contained in a scheduler batch, while the batch starts
        # before the parent.  Match by interval containment explicitly.
        batch_matches = [
            row
            for row in request_batches
            if row["started_ns"] <= parent["start_ns"] <= parent["end_ns"] <= row["ended_ns"]
        ]
        if len(batch_matches) != 1:
            raise ValueError(f"Host parent {parent_id} does not map to one scheduler batch")
        batch = batch_matches[0]
        joined.append(
            {
                "request_id": parent["request_id"],
                "batch_sequence": batch["batch_sequence"],
                "mode": batch["mode"],
                "layer": parent["layer"],
                "parent_start_ns": parent["start_ns"],
                "parent_end_ns": parent["end_ns"],
                "parent_ns": parent["duration_ns"],
                "dispatch_ns": dispatch["duration_ns"],
                "dispatch_pre_enqueue_ns": enqueue["now_ns"] - dispatch["start_ns"],
                "dispatch_post_enqueue_ns": dispatch["end_ns"] - enqueue["now_ns"],
                "queue_wait_ns": task["start_ns"] - task["admitted_ns"],
                "cpu_task_ns": task["now_ns"] - task["start_ns"],
                "cpu_task_start_ns": task["start_ns"],
                "cpu_task_end_ns": task["now_ns"],
                "native_enqueue_ns": enqueue["now_ns"],
                "host_sync_start_ns": host_sync["start_ns"],
                "host_sync_end_ns": host_sync["end_ns"],
                "native_sync_begin_ns": sync_begin["now_ns"],
                "native_sync_end_ns": sync_end["now_ns"],
                "host_sync_ns": host_sync["duration_ns"],
                "host_sync_pre_native_ns": sync_begin["now_ns"] - host_sync["start_ns"],
                "native_sync_wait_ns": sync_end["now_ns"] - sync_begin["now_ns"],
                "host_sync_post_native_ns": host_sync["end_ns"] - sync_end["now_ns"],
                "sync_pending_begin": sync_begin["pending"],
                "sync_pending_end": sync_end["pending"],
                "cpu_tail_after_host_sync_start_ns": max(0, task["now_ns"] - host_sync["start_ns"]),
                "cpu_hidden_before_host_sync_ns": max(0, host_sync["start_ns"] - task["now_ns"]),
                "host_merge_ns": components["HOST_MERGE"]["duration_ns"],
                "native_task_sequence": enqueue["sequence"],
                "native_sync_sequence": sync_begin["sequence"],
                "scheduler_batch_start_ns": batch["started_ns"],
                "scheduler_batch_end_ns": batch["ended_ns"],
                "scheduler_batch_ns": batch["duration_ns"],
            }
        )
    joined.sort(key=lambda row: row["parent_start_ns"])
    return joined


def assert_nonoverlap(rows: list[dict[str, Any]]) -> None:
    for left, right in zip(rows, rows[1:]):
        if left["parent_end_ns"] > right["parent_start_ns"]:
            raise ValueError("routed-expert parents overlap; sums are not a critical-path partition")


def analyze(joined: list[dict[str, Any]], *, request_marker: str = "W2_MEASUREMENT") -> dict[str, Any]:
    selected = [r for r in joined if request_marker in r["request_id"] and r["mode"] == "decode"]
    if not selected:
        raise ValueError("no W2 measurement Decode calls")
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_batch[row["batch_sequence"]].append(row)
    if any(len(rows) != 48 for rows in by_batch.values()):
        raise ValueError("every W2 Decode batch must cover 48 routed-expert layers")
    # Drop the first two Decode batches from steady-state summaries, while
    # retaining their raw rows.  They contain first-Decode compilation/caches.
    steady_keys = sorted(by_batch)[2:]
    steady = [row for key in steady_keys for row in by_batch[key]]
    assert_nonoverlap(sorted(steady, key=lambda row: row["parent_start_ns"]))
    metrics = [
        "parent_ns",
        "dispatch_ns",
        "dispatch_pre_enqueue_ns",
        "dispatch_post_enqueue_ns",
        "queue_wait_ns",
        "cpu_task_ns",
        "host_sync_ns",
        "host_sync_pre_native_ns",
        "native_sync_wait_ns",
        "host_sync_post_native_ns",
        "cpu_tail_after_host_sync_start_ns",
        "cpu_hidden_before_host_sync_ns",
        "host_merge_ns",
    ]
    per_call = {key: summary_ns(row[key] for row in steady) for key in metrics}
    batch_rows = []
    for key in steady_keys:
        rows = sorted(by_batch[key], key=lambda row: row["parent_start_ns"])
        assert_nonoverlap(rows)
        wall = rows[0]["scheduler_batch_ns"]
        parent = sum(row["parent_ns"] for row in rows)
        batch_rows.append(
            {
                "batch_sequence": key,
                "scheduler_wall_ns": wall,
                "routed_expert_parent_sum_ns": parent,
                "non_expert_residual_ns": wall - parent,
                "cpu_task_sum_ns_nested_with_npu": sum(row["cpu_task_ns"] for row in rows),
                "cpu_hidden_before_host_sync_sum_ns": sum(row["cpu_hidden_before_host_sync_ns"] for row in rows),
                "native_sync_wait_sum_ns": sum(row["native_sync_wait_ns"] for row in rows),
                "device_wait_before_native_sync_sum_ns": sum(row["host_sync_pre_native_ns"] for row in rows),
            }
        )
    if any(row["non_expert_residual_ns"] < 0 for row in batch_rows):
        raise ValueError("expert parent sum exceeds scheduler batch wall")
    per_token = {
        key: summary_ns(row[key] for row in batch_rows)
        for key in (
            "scheduler_wall_ns",
            "routed_expert_parent_sum_ns",
            "non_expert_residual_ns",
            "cpu_task_sum_ns_nested_with_npu",
            "cpu_hidden_before_host_sync_sum_ns",
            "native_sync_wait_sum_ns",
            "device_wait_before_native_sync_sum_ns",
        )
    }
    total_wall = sum(row["scheduler_wall_ns"] for row in batch_rows)
    total_expert = sum(row["routed_expert_parent_sum_ns"] for row in batch_rows)
    total_cpu_exposed = sum(row["native_sync_wait_sum_ns"] for row in batch_rows)
    return {
        "schema": 1,
        "status": "PASS_KT_ROOT_NATIVE_HOST_JOIN",
        "request_marker": request_marker,
        "decode_batches_total": len(by_batch),
        "decode_batches_steady": len(batch_rows),
        "calls_total": len(selected),
        "calls_steady": len(steady),
        "per_call": per_call,
        "per_decode_token": per_token,
        "critical_path_fractions": {
            "routed_expert_parent": total_expert / total_wall,
            "non_expert_residual": 1 - total_expert / total_wall,
            "cpu_exposed_native_sync_wait": total_cpu_exposed / total_wall,
            "cpu_task_sum_is_nested_not_additive": True,
        },
        "pending_at_sync": {
            "pending_zero": sum(row["sync_pending_begin"] == 0 for row in steady),
            "pending_nonzero": sum(row["sync_pending_begin"] != 0 for row in steady),
        },
        "batch_rows": batch_rows,
        "raw_joined_rows": joined,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--round13", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    native_integrity: dict[str, Any] = {}
    request_marker = "W2_MEASUREMENT"
    result = analyze(
        join_calls(
            load_jsonl(args.observer),
            load_jsonl(
                args.native,
                allow_terminal_partial=True,
                diagnostics=native_integrity,
            ),
            load_jsonl(args.round13),
            request_marker=request_marker,
        ),
        request_marker=request_marker,
    )
    result["native_trace_integrity"] = native_integrity
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
