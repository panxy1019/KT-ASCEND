#!/usr/bin/env python3
"""Compute Round11 Prefill-End Dynamic break-even from four unprofiled blocks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


LENGTHS = (32, 64, 128, 256)
MIGRATION_PHASES = (
    "route_statistics",
    "consumer_drain_sync",
    "state_fence_before",
    "selection",
    "migration_copy_and_sync",
    "mapping_publication",
    "state_fence_after",
    "cleanup",
    "total_exposed",
)


def _steady_intervals(row):
    arrivals = row.get("token_arrival_monotonic_ns")
    length = row.get("decode_tokens")
    if (
        length not in LENGTHS
        or not isinstance(arrivals, list)
        or len(arrivals) != length
        or any(type(value) is not int for value in arrivals)
        or any(right <= left for left, right in zip(arrivals, arrivals[1:]))
    ):
        raise ValueError("invalid token arrival series")
    # The first -> second interval contains the Prefill-End Dynamic barrier in
    # D mode. Exclude it from both modes. M records that barrier separately.
    values = [(arrivals[index] - arrivals[index - 1]) / 1e9 for index in range(2, len(arrivals))]
    if not values:
        raise ValueError("insufficient steady decode intervals")
    return values


def _validate_process(document):
    mode = document.get("mode")
    if (
        document.get("status") != "PASS_ROUND11_T4_PROCESS"
        or mode not in {"STATIC", "DYNAMIC"}
        or document.get("block") not in {"A", "B"}
        or document.get("profiler") != "OFF"
        or not isinstance(document.get("parent_identity"), str)
        or not document["parent_identity"]
    ):
        raise ValueError("invalid T4 process identity")
    cleanup = document.get("cleanup", {})
    if cleanup != {
        "guard_exit_ack": True,
        "container_removed": True,
        "physical12_after": "EMPTY",
    }:
        raise ValueError("T4 cleanup contract failed")
    rows = document.get("results")
    if not isinstance(rows, list) or sorted(row.get("decode_tokens") for row in rows) != list(LENGTHS):
        raise ValueError("T4 decode schedule mismatch")
    for row in rows:
        length = row["decode_tokens"]
        if (
            row.get("input_tokens") != 64
            or not isinstance(row.get("output_ids"), list)
            or len(row["output_ids"]) != length
            or row.get("finite_logprobs") is not True
            or row.get("finish_reason") != "length"
            or row.get("path") != "HYBRID"
            or not isinstance(row.get("e2e_seconds"), (int, float))
            or not math.isfinite(row["e2e_seconds"])
            or row["e2e_seconds"] <= 0
        ):
            raise ValueError("T4 request correctness contract failed")
        _steady_intervals(row)
        if mode == "DYNAMIC":
            migration = row.get("migration", {})
            timing = migration.get("timing_ns", {})
            if (
                migration.get("status") != "PREFILL_END_COMMIT"
                or migration.get("live_state_preserved") is not True
                or type(migration.get("changed_layers")) is not int
                or type(migration.get("copied_experts")) is not int
                or set(timing) != set(MIGRATION_PHASES)
                or any(type(timing[name]) is not int or timing[name] < 0 for name in MIGRATION_PHASES)
            ):
                raise ValueError("Dynamic migration phase contract failed")
            phase_sum = sum(timing[name] for name in MIGRATION_PHASES if name != "total_exposed")
            if phase_sum > timing["total_exposed"] * 1.02 + 5_000_000:
                raise ValueError("migration phases exceed exposed parent")
        elif "migration" in row:
            raise ValueError("Static process unexpectedly contains migration")
    return document


def calculate_break_even(documents):
    if len(documents) != 4:
        raise ValueError("T4 requires four independent model-process blocks")
    runs = [_validate_process(document) for document in documents]
    identities = {document["parent_identity"] for document in runs}
    pairs = {(document["mode"], document["block"]) for document in runs}
    if len(identities) != 1 or pairs != {
        ("STATIC", "A"),
        ("STATIC", "B"),
        ("DYNAMIC", "A"),
        ("DYNAMIC", "B"),
    }:
        raise ValueError("T4 parent or independent block coverage mismatch")

    by_mode = {mode: [run for run in runs if run["mode"] == mode] for mode in ("STATIC", "DYNAMIC")}
    for mode, mode_runs in by_mode.items():
        first, second = sorted(mode_runs, key=lambda run: run["block"])
        first_outputs = {row["decode_tokens"]: row["output_ids"] for row in first["results"]}
        second_outputs = {row["decode_tokens"]: row["output_ids"] for row in second["results"]}
        if first_outputs != second_outputs:
            raise ValueError(f"{mode} same-mode output IDs are not exact across process blocks")
    steady = {
        mode: [
            value
            for run in by_mode[mode]
            for row in run["results"]
            for value in _steady_intervals(row)
        ]
        for mode in by_mode
    }
    static = statistics.median(steady["STATIC"])
    dynamic = statistics.median(steady["DYNAMIC"])
    migration_seconds = [
        row["migration"]["timing_ns"]["total_exposed"] / 1e9
        for run in by_mode["DYNAMIC"]
        for row in run["results"]
    ]
    migration = statistics.median(migration_seconds)
    per_length = {}
    for length in LENGTHS:
        per_length[str(length)] = {
            mode.lower() + "_e2e_median_seconds": statistics.median(
                row["e2e_seconds"]
                for run in by_mode[mode]
                for row in run["results"]
                if row["decode_tokens"] == length
            )
            for mode in ("STATIC", "DYNAMIC")
        }

    confirmed_length = None
    if dynamic >= static:
        break_even = None
        disposition = "NO_FINITE_BREAK_EVEN"
    else:
        break_even = migration / (static - dynamic)
        for length in LENGTHS:
            values = per_length[str(length)]
            if length >= math.ceil(break_even) and values["dynamic_e2e_median_seconds"] < values[
                "static_e2e_median_seconds"
            ]:
                confirmed_length = length
                break
        disposition = (
            f"FINITE_BREAK_EVEN_CONFIRMED_AT_{confirmed_length}"
            if confirmed_length is not None
            else "FINITE_BREAK_EVEN_REQUIRES_REQUEST_CONFIRMATION"
        )

    phase_medians = {
        name: statistics.median(
            row["migration"]["timing_ns"][name] / 1e9
            for run in by_mode["DYNAMIC"]
            for row in run["results"]
        )
        for name in MIGRATION_PHASES
    }
    return {
        "schema": 1,
        "status": "PASS_ROUND11_T4_BREAK_EVEN",
        "parent_identity": identities.pop(),
        "process_blocks": "STATIC_A_DYNAMIC_A_DYNAMIC_B_STATIC_B",
        "decode_lengths": list(LENGTHS),
        "S_static_seconds_per_token": static,
        "D_dynamic_seconds_per_token": dynamic,
        "M_migration_exposed_seconds": migration,
        "N_break_even_tokens": break_even,
        "request_confirmed_at_tokens": confirmed_length,
        "disposition": disposition,
        "per_length_e2e": per_length,
        "migration_phase_median_seconds": phase_medians,
        "migration_phase_interpretation": (
            "All phases are exclusive subdivisions of total_exposed. "
            "The first-to-second token interval is excluded from S and D because it contains M."
        ),
        "quality": {
            "all_output_budgets_completed": True,
            "all_logprobs_finite": True,
            "all_live_state_preserved": True,
            "same_mode_output_ids_exact": True,
            "cross_mode_bit_exact_required": False,
        },
        "cleanup": "ALL_FOUR_GUARD_ACK_CONTAINER_REMOVED_PHYSICAL12_EMPTY",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-a", type=Path, required=True)
    parser.add_argument("--dynamic-a", type=Path, required=True)
    parser.add_argument("--dynamic-b", type=Path, required=True)
    parser.add_argument("--static-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    documents = [
        json.loads(path.read_text())
        for path in (args.static_a, args.dynamic_a, args.dynamic_b, args.static_b)
    ]
    result = calculate_break_even(documents)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["disposition"], flush=True)


if __name__ == "__main__":
    main()
