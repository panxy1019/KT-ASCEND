#!/usr/bin/env python3
"""Validate Round 6.1 deferred buffer immutability and trace ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tensor_checks = []
    for before in sorted(args.stage_dir.glob("cpu_layer??_*_call????.pt")):
        name = before.name
        if "_after_sync_" in name or "_output_" in name:
            continue
        stage = name.split("_call", 1)[0]
        tensor_name = stage.split("_", 2)[2]
        if tensor_name not in {
            "input_hidden",
            "immediate_ids",
            "deferred_ids",
            "routing_weights",
        }:
            continue
        after = before.with_name(
            name.replace(f"_{tensor_name}_call", f"_{tensor_name}_after_sync_call")
        )
        if not after.exists():
            continue
        lhs_payload = torch.load(before, map_location="cpu", weights_only=True)
        rhs_payload = torch.load(after, map_location="cpu", weights_only=True)
        lhs = lhs_payload["tensor"] if isinstance(lhs_payload, dict) else lhs_payload
        rhs = rhs_payload["tensor"] if isinstance(rhs_payload, dict) else rhs_payload
        tensor_checks.append(
            {
                "before": before.name,
                "after": after.name,
                "shape": list(lhs.shape),
                "dtype": str(lhs.dtype),
                "exact": bool(torch.equal(lhs, rhs)),
            }
        )

    events = [json.loads(line) for line in args.trace.read_text().splitlines() if line]
    sequences = [int(event["trace_sequence"]) for event in events]
    sequence_monotonic = sequences == sorted(sequences) and len(sequences) == len(set(sequences))
    target_submits = [event for event in events if event.get("event") == "submit" and event.get("rows") == 54]
    target_sequences = [event["trace_sequence"] for event in target_submits]
    producer = next(
        (event for event in target_submits if event.get("layer_idx") == 7), None
    )
    consumer = next(
        (event for event in target_submits if event.get("layer_idx") == 8), None
    )
    consumed = next(
        (
            event
            for event in events
            if event.get("event") == "previous_pending_consumed"
            and event.get("producer_layer") == 7
            and event.get("consumer_layer") == 8
        ),
        None,
    )
    producer_sync_complete = next(
        (
            event
            for event in events
            if event.get("event") == "sync_complete"
            and event.get("layer_idx") == 7
            and producer
            and event["trace_sequence"] > producer["trace_sequence"]
        ),
        None,
    )
    pending_cleared = any(
        event.get("event") == "pending_state"
        and event.get("layer_idx") == 8
        and event.get("pending_layers") == []
        and consumed
        and event["trace_sequence"] > consumed["trace_sequence"]
        for event in events
    )
    ordering_pass = bool(
        producer
        and consumer
        and consumed
        and producer_sync_complete
        and producer["trace_sequence"] < producer_sync_complete["trace_sequence"]
        < consumer["trace_sequence"] < consumed["trace_sequence"]
        and pending_cleared
    )
    result = {
        "schema_version": "round6.1-runtime-controls-v1",
        "tensor_immutability": {
            "checks": tensor_checks,
            "count": len(tensor_checks),
            "all_exact": bool(tensor_checks) and all(item["exact"] for item in tensor_checks),
        },
        "trace": {
            "event_count": len(events),
            "sequence_monotonic": sequence_monotonic,
            "target_submit_sequences": target_sequences,
            "producer_submit": producer,
            "producer_sync_complete": producer_sync_complete,
            "consumer_submit": consumer,
            "previous_pending_consumed": consumed,
            "pending_cleared": pending_cleared,
            "ordering_pass": ordering_pass,
        },
    }
    result["pass"] = (
        result["tensor_immutability"]["all_exact"]
        and sequence_monotonic
        and ordering_pass
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": result["pass"], "tensor_checks": len(tensor_checks), "ordering_pass": ordering_pass}))


if __name__ == "__main__":
    main()
