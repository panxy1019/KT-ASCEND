"""Refine the exact W2 residual into dependency-positioned Host intervals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
from typing import Any

from .analyze_profile import percentile


ATTENTION_COMPONENTS = frozenset({"GDN", "QSA_ATTENTION"})


def _kind(interval: dict[str, Any] | None) -> str:
    if interval is None:
        return "scheduler"
    component = interval["component"]
    if component in ATTENTION_COMPONENTS:
        return "attention"
    if component == "ROUTED_EXPERT":
        return "expert"
    if component == "PLE_CONSUMER":
        return "ple"
    raise ValueError(f"unsupported top-level component {component}")


def _gap_name(
    previous: dict[str, Any] | None, following: dict[str, Any] | None
) -> str:
    before = _kind(previous)
    after = _kind(following)
    if (
        before == "attention"
        and after == "expert"
        and previous is not None
        and following is not None
        and previous["layer"] == following["layer"]
    ):
        return "attention_to_expert_same_layer"
    if (
        before == "expert"
        and after == "attention"
        and previous is not None
        and following is not None
        and previous["layer"] + 1 == following["layer"]
    ):
        return "expert_to_next_attention"
    return f"{before}_to_{after}"


def partition_timeline(
    start_ns: int, end_ns: int, intervals: list[dict[str, Any]]
) -> dict[str, Any]:
    """Partition one single-thread scheduler wall into known parents and gaps."""
    if not isinstance(start_ns, int) or not isinstance(end_ns, int) or end_ns < start_ns:
        raise ValueError("invalid scheduler interval")
    ordered = sorted(intervals, key=lambda row: (row["start_ns"], row["end_ns"]))
    cursor = start_ns
    previous = None
    gaps: dict[str, int] = defaultdict(int)
    known_ns = 0
    for interval in ordered:
        begin = interval["start_ns"]
        finish = interval["end_ns"]
        if begin < start_ns or finish > end_ns or finish < begin:
            raise ValueError("known interval is outside scheduler wall")
        if begin < cursor:
            raise ValueError("top-level interval overlap")
        if begin > cursor:
            gaps[_gap_name(previous, interval)] += begin - cursor
        known_ns += finish - begin
        cursor = finish
        previous = interval
    if cursor < end_ns:
        gaps[_gap_name(previous, None)] += end_ns - cursor
    wall_ns = end_ns - start_ns
    gap_ns = sum(gaps.values())
    if known_ns + gap_ns != wall_ns:
        raise ValueError("known intervals and gaps do not reconstruct scheduler wall")
    return {
        "wall_ns": wall_ns,
        "known_ns": known_ns,
        "gap_ns": gap_ns,
        "gaps": dict(gaps),
    }


def _summary_ms(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median_ms": statistics.median(values) / 1e6,
        "p05_ms": percentile(values, 0.05) / 1e6,
        "p95_ms": percentile(values, 0.95) / 1e6,
        "mean_ms": statistics.fmean(values) / 1e6,
        "sum_ms": sum(values) / 1e6,
    }


def build_gap_accounting(
    analysis: dict[str, Any], span_batches: list[dict[str, Any]]
) -> dict[str, Any]:
    if analysis.get("status") != "PASS_KT_ROOT_NATIVE_HOST_JOIN":
        raise ValueError("native/Host analysis is not qualified")
    selected = {
        row["batch_sequence"]: row
        for row in span_batches
        if row.get("status") == "ROUND13_P0_BATCH"
        and row.get("mode") == "decode"
        and "W2_MEASUREMENT" in row.get("request_id", "")
    }
    experts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in analysis["raw_joined_rows"]:
        if row.get("mode") == "decode" and "W2_MEASUREMENT" in row.get(
            "request_id", ""
        ):
            experts[row["batch_sequence"]].append(row)

    token_rows: list[dict[str, Any]] = []
    all_gap_names: set[str] = set()
    for source in analysis["batch_rows"]:
        sequence = source["batch_sequence"]
        batch = selected.get(sequence)
        if batch is None:
            raise ValueError(f"missing W2 span batch {sequence}")
        expert_rows = experts.get(sequence, [])
        if len(expert_rows) != 48 or len({row["layer"] for row in expert_rows}) != 48:
            raise ValueError(f"batch {sequence} does not contain 48 unique expert parents")

        components = defaultdict(list)
        for span in batch["spans"]:
            components[span["component"]].append(span)
        if len(components["GDN"]) != 36:
            raise ValueError(f"batch {sequence} does not contain 36 GDN parents")
        if len(components["QSA_ATTENTION"]) != 12:
            raise ValueError(f"batch {sequence} does not contain 12 QSA parents")
        if len(components["PLE_CONSUMER"]) != 1:
            raise ValueError(f"batch {sequence} does not contain one PLE consumer")

        intervals = [
            {
                "component": span["component"],
                "layer": span["layer"],
                "start_ns": span["start_ns"],
                "end_ns": span["end_ns"],
            }
            for name in ("GDN", "QSA_ATTENTION", "PLE_CONSUMER")
            for span in components[name]
        ]
        intervals.extend(
            {
                "component": "ROUTED_EXPERT",
                "layer": row["layer"],
                "start_ns": row["parent_start_ns"],
                "end_ns": row["parent_end_ns"],
            }
            for row in expert_rows
        )
        partition = partition_timeline(
            batch["started_ns"], batch["ended_ns"], intervals
        )
        if partition["wall_ns"] != source["scheduler_wall_ns"]:
            raise ValueError(f"batch {sequence} scheduler wall differs across observers")
        all_gap_names.update(partition["gaps"])
        token_rows.append(
            {
                "batch_sequence": sequence,
                "scheduler_wall_ns": partition["wall_ns"],
                "known_parent_ns": partition["known_ns"],
                "dependency_gap_ns": partition["gap_ns"],
                **partition["gaps"],
            }
        )

    ordered_gap_names = sorted(all_gap_names)
    for row in token_rows:
        for name in ordered_gap_names:
            row.setdefault(name, 0)
    summaries = {
        name: _summary_ms([int(row[name]) for row in token_rows])
        for name in (
            "scheduler_wall_ns",
            "known_parent_ns",
            "dependency_gap_ns",
            *ordered_gap_names,
        )
    }
    total_wall = sum(row["scheduler_wall_ns"] for row in token_rows)
    return {
        "schema": 1,
        "status": "PASS_EXACT_DEPENDENCY_POSITION_GAP_ACCOUNTING",
        "steady_decode_tokens": len(token_rows),
        "known_parent_components": [
            "GDN",
            "QSA_ATTENTION",
            "ROUTED_EXPERT",
            "PLE_CONSUMER",
        ],
        "method": (
            "sort non-overlapping top-level Host parent intervals on the same "
            "scheduler monotonic clock and name every residual interval by its "
            "preceding and following dependency boundary"
        ),
        "summaries": summaries,
        "fractions_of_scheduler_wall": {
            name: sum(row[name] for row in token_rows) / total_wall
            for name in ("known_parent_ns", "dependency_gap_ns", *ordered_gap_names)
        },
        "interpretation": {
            "attention_to_expert_same_layer": (
                "after GDN/QSA parent and before routed-expert parent in the same "
                "decoder layer: HyperConnection combine/mix, router, dispatch, "
                "shared-expert scheduling and queued NPU/Host glue"
            ),
            "expert_to_next_attention": (
                "after routed-expert parent and before the next layer attention: "
                "shared/routed combination, scaling, HyperConnection combine and "
                "the next layer pre-attention path"
            ),
            "warning": (
                "gaps are exact wall positions, not pure Python CPU time; queued "
                "NPU work may complete at a later dependency within a gap"
            ),
        },
        "token_rows": token_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    spans = [json.loads(line) for line in args.spans.read_text(encoding="utf-8").splitlines()]
    result = build_gap_accounting(analysis, spans)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, result["token_rows"])
    print(result["status"])


if __name__ == "__main__":
    main()
