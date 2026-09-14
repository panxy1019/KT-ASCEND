#!/usr/bin/env python3
"""Summarize and validate opt-in Round 6 Deferred Expert JSONL traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--producer-consumer",
        action="append",
        default=[],
        help="Expected adjacent producer:consumer pair, for example 7:8.",
    )
    args = parser.parse_args()

    records = [json.loads(line) for line in args.trace.read_text().splitlines()]
    if not records:
        raise SystemExit("empty Deferred trace")

    submits: dict[int, list[dict]] = defaultdict(list)
    syncs: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        event = record.get("event")
        layer_idx = int(record["layer_idx"])
        if event == "submit":
            submits[layer_idx].append(record)
        elif event == "sync":
            syncs[layer_idx].append(record)
        else:
            raise RuntimeError(f"unknown trace event: {event!r}")

    layers = {}
    failures = []
    for layer_idx in sorted(set(submits) | set(syncs)):
        layer_submits = submits[layer_idx]
        layer_syncs = syncs[layer_idx]
        pending = sum(bool(row["outgoing_pending"]) for row in layer_submits)
        deferred_cpu = sum(int(row["deferred_cpu_routes"]) for row in layer_submits)
        false_pending = sum(
            bool(row["outgoing_pending"]) and int(row["deferred_cpu_routes"]) == 0
            for row in layer_submits
        )
        missed_pending = sum(
            not bool(row["outgoing_pending"]) and int(row["deferred_cpu_routes"]) > 0
            for row in layer_submits
        )
        allow_pending = sum(int(row["allow_pending"]) for row in layer_syncs)
        summary = {
            "submit_events": len(layer_submits),
            "sync_events": len(layer_syncs),
            "rows": sum(int(row["rows"]) for row in layer_submits),
            "immediate_cpu_routes": sum(
                int(row["immediate_cpu_routes"]) for row in layer_submits
            ),
            "deferred_cpu_routes": deferred_cpu,
            "actual_pending_events": pending,
            "incremental_from_previous_events": sum(
                bool(row["incremental_from_previous"]) for row in layer_submits
            ),
            "sync_allow_pending_events": allow_pending,
            "false_pending_events": false_pending,
            "missed_pending_events": missed_pending,
            "cpu_experts": sorted(
                {
                    int(expert_id)
                    for row in layer_submits
                    for expert_id in (
                        row["immediate_cpu_experts"] + row["deferred_cpu_experts"]
                    )
                }
            ),
        }
        layers[str(layer_idx)] = summary
        if len(layer_submits) != len(layer_syncs):
            failures.append(f"layer {layer_idx}: submit/sync count mismatch")
        if false_pending:
            failures.append(f"layer {layer_idx}: {false_pending} false pending events")
        if missed_pending:
            failures.append(f"layer {layer_idx}: {missed_pending} missed pending events")
        if pending != allow_pending:
            failures.append(f"layer {layer_idx}: pending/sync allowance mismatch")

    pairs = {}
    for value in args.producer_consumer:
        producer_text, consumer_text = value.split(":", 1)
        producer, consumer = int(producer_text), int(consumer_text)
        if consumer != producer + 1:
            raise SystemExit(f"non-adjacent pair is invalid: {value}")
        producer_pending = layers[str(producer)]["actual_pending_events"]
        consumer_incremental = layers[str(consumer)][
            "incremental_from_previous_events"
        ]
        exact = producer_pending == consumer_incremental
        pairs[value] = {
            "producer_pending_events": producer_pending,
            "consumer_incremental_events": consumer_incremental,
            "exact": exact,
        }
        if not exact:
            failures.append(f"pair {value}: producer/consumer mismatch")

    payload = {
        "schema_version": 1,
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "record_count": len(records),
        "layers": layers,
        "producer_consumer_pairs": pairs,
        "failures": failures,
        "pass": not failures,
    }
    payload["sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
