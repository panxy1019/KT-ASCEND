#!/usr/bin/env python3
"""Rank Round11 GDN, QSA, and PLE costs without adding nested parents."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


GDN_SAMPLE_LAYERS = (0, 24, 46)
QSA_SAMPLE_LAYERS = (3, 23, 47)
GDN_LAYER_COUNT = 36
QSA_LAYER_COUNT = 12
DEVICE_COMPONENTS = frozenset(("GDN", "QSA_ATTENTION", "QSA_INDEXER", "PLE_CONSUMER"))
PLE_COMPONENTS = frozenset(("PLE_PREPARE", "PLE_EXPOSED_WAIT", "PLE_CONSUMER"))
EXPECTED_BINDINGS = {
    "GDN": list(GDN_SAMPLE_LAYERS),
    "QSA_ATTENTION": list(QSA_SAMPLE_LAYERS),
    "QSA_INDEXER": list(QSA_SAMPLE_LAYERS),
    "PLE": [1],
}
EXPECTED_CLEANUP = {
    "guard_exit_ack": True,
    "container_removed": True,
    "physical12_after": "EMPTY",
}


def _finite_positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {label}")
    return float(value)


def _validate_span(row):
    try:
        component = row["component"]
        layer = row["layer"]
        start = row["start_ns"]
        end = row["end_ns"]
    except KeyError as error:
        raise ValueError("incomplete B-route span") from error
    if (not isinstance(component, str) or not component or type(layer) is not int
            or type(start) is not int or type(end) is not int or end < start
            or row.get("duration_ns") != end - start
            or row.get("observer_errors") != []):
        raise ValueError("invalid B-route span")
    if component in DEVICE_COMPONENTS:
        if row.get("device_status") != "COMPLETE":
            raise ValueError("required device timing is incomplete")
        _finite_positive(row.get("device_ms"), "device timing")
    elif row.get("device_ms") is not None:
        raise ValueError("host-only span unexpectedly has device timing")
    return row


def _component_rows(batch, component):
    return [row for row in batch["spans"] if row["component"] == component]


def _validate_batch(batch, request_ids):
    if (batch.get("request_id") not in request_ids
            or batch.get("mode") not in {"extend", "decode"}
            or type(batch.get("batch_sequence")) is not int
            or batch["batch_sequence"] < 0
            or batch.get("dropped_spans") != 0
            or batch.get("observer_errors") != []
            or not isinstance(batch.get("spans"), list)):
        raise ValueError("invalid B-route batch coverage")
    batch["spans"] = [_validate_span(row) for row in batch["spans"]]
    for component, layers in (("GDN", GDN_SAMPLE_LAYERS),
                              ("QSA_ATTENTION", QSA_SAMPLE_LAYERS),
                              ("QSA_INDEXER", QSA_SAMPLE_LAYERS)):
        rows = _component_rows(batch, component)
        if sorted(row["layer"] for row in rows) != list(layers):
            raise ValueError(f"{component} sample coverage mismatch")
    for component in PLE_COMPONENTS:
        rows = _component_rows(batch, component)
        if len(rows) != 1 or rows[0]["layer"] != 1:
            raise ValueError(f"{component} sample coverage mismatch")
    parents = {row["layer"]: row for row in _component_rows(batch, "QSA_ATTENTION")}
    children = {row["layer"]: row for row in _component_rows(batch, "QSA_INDEXER")}
    for layer in QSA_SAMPLE_LAYERS:
        parent, child = parents[layer], children[layer]
        if not (parent["start_ns"] <= child["start_ns"] <= child["end_ns"] <= parent["end_ns"]):
            raise ValueError("QSA indexer is not nested in its attention parent")
        if child["device_ms"] > parent["device_ms"] * 1.02 + 0.05:
            raise ValueError("QSA child device timing exceeds parent")
    return batch


def _validate_process(document):
    if (document.get("status") != "PASS_ROUND11_T5_PROCESS"
            or document.get("block") not in {"A", "B"}
            or document.get("profiler") != "BOUNDED_OBSERVER_ON"
            or not isinstance(document.get("parent_identity"), str)
            or not document["parent_identity"]
            or document.get("bindings") != EXPECTED_BINDINGS):
        raise ValueError("invalid T5 process identity")
    if document.get("cleanup") != EXPECTED_CLEANUP:
        raise ValueError("T5 cleanup contract failed")
    requests = document.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("T5 measurement requests missing")
    by_id = {}
    for row in requests:
        request_id = row.get("request_id")
        if (not isinstance(request_id, str) or not request_id or request_id in by_id
                or row.get("phase") != "measurement"
                or type(row.get("input_tokens")) is not int or row["input_tokens"] < 1
                or type(row.get("decode_tokens")) is not int or row["decode_tokens"] < 1
                or row.get("quality_pass") is not True):
            raise ValueError("invalid T5 request contract")
        _finite_positive(row.get("e2e_seconds"), "request E2E")
        by_id[request_id] = row
    batches = document.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("T5 batches missing")
    validated = [_validate_batch(batch, set(by_id)) for batch in batches]
    for request_id in by_id:
        modes = {batch["mode"] for batch in validated if batch["request_id"] == request_id}
        if modes != {"extend", "decode"}:
            raise ValueError("T5 request lacks extend/decode batch coverage")
    return by_id, validated


def _batch_cost(batch):
    def device(component):
        return statistics.median(row["device_ms"] for row in _component_rows(batch, component))

    gdn = device("GDN") * GDN_LAYER_COUNT
    qsa = device("QSA_ATTENTION") * QSA_LAYER_COUNT
    indexer = device("QSA_INDEXER") * QSA_LAYER_COUNT
    ple_prepare = sum(row["duration_ns"] for row in _component_rows(batch, "PLE_PREPARE")) / 1e6
    ple_wait = sum(row["duration_ns"] for row in _component_rows(batch, "PLE_EXPOSED_WAIT")) / 1e6
    ple_consumer = sum(row["device_ms"] for row in _component_rows(batch, "PLE_CONSUMER"))
    return {
        "mode": batch["mode"],
        "GDN": gdn,
        "QSA": qsa,
        "QSA_INDEXER": indexer,
        "QSA_EXCLUSIVE": max(0.0, qsa - indexer),
        "PLE_PREPARE_HOST": ple_prepare,
        "PLE_EXPOSED_WAIT": ple_wait,
        "PLE_CONSUMER": ple_consumer,
    }


def rank_b_route_hotspots(documents):
    if len(documents) != 2:
        raise ValueError("T5 ranking requires two independent profile processes")
    processes = []
    for document in documents:
        requests, batches = _validate_process(document)
        processes.append((document, requests, batches))
    if {document["block"] for document, unused, unused2 in processes} != {"A", "B"}:
        raise ValueError("T5 independent block coverage mismatch")
    parents = {document["parent_identity"] for document, unused, unused2 in processes}
    if len(parents) != 1:
        raise ValueError("T5 parent identity mismatch")

    request_rows = []
    for document, requests, batches in processes:
        for request_id, request in requests.items():
            costs = [_batch_cost(batch) for batch in batches if batch["request_id"] == request_id]
            total = {name: sum(row[name] for row in costs) for name in (
                "GDN", "QSA", "QSA_INDEXER", "QSA_EXCLUSIVE", "PLE_PREPARE_HOST",
                "PLE_EXPOSED_WAIT", "PLE_CONSUMER")}
            e2e_ms = request["e2e_seconds"] * 1000.0
            per_mode = {
                mode: {name: sum(row[name] for row in costs if row["mode"] == mode)
                       for name in total}
                for mode in ("extend", "decode")
            }
            request_rows.append({
                "block": document["block"],
                "request_id": request_id,
                "e2e_ms": e2e_ms,
                "attributable_ms": total,
                "fractions": {
                    "GDN": total["GDN"] / e2e_ms,
                    "QSA": total["QSA"] / e2e_ms,
                    "PLE_CONSUMER": total["PLE_CONSUMER"] / e2e_ms,
                    "PLE_EXPOSED_WAIT": total["PLE_EXPOSED_WAIT"] / e2e_ms,
                },
                "per_mode_ms": per_mode,
            })

    def med(path):
        return statistics.median(path(row) for row in request_rows)

    modules = {
        "GDN": {
            "attributable_ms": med(lambda row: row["attributable_ms"]["GDN"]),
            "critical_fraction": med(lambda row: row["fractions"]["GDN"]),
        },
        "QSA": {
            "attributable_ms": med(lambda row: row["attributable_ms"]["QSA"]),
            "critical_fraction": med(lambda row: row["fractions"]["QSA"]),
            "indexer_device_ms": med(lambda row: row["attributable_ms"]["QSA_INDEXER"]),
            "exclusive_remainder_ms": med(lambda row: row["attributable_ms"]["QSA_EXCLUSIVE"]),
        },
        "PLE": {
            "consumer_device_ms": med(lambda row: row["attributable_ms"]["PLE_CONSUMER"]),
            "critical_fraction": med(lambda row: row["fractions"]["PLE_CONSUMER"]),
            "prepare_host_ms": med(lambda row: row["attributable_ms"]["PLE_PREPARE_HOST"]),
            "exposed_wait_ms": med(lambda row: row["attributable_ms"]["PLE_EXPOSED_WAIT"]),
            "exposed_wait_fraction": med(lambda row: row["fractions"]["PLE_EXPOSED_WAIT"]),
        },
    }
    eligible = {}
    for name in ("GDN", "QSA"):
        if modules[name]["critical_fraction"] >= 0.10:
            eligible[name] = modules[name]["critical_fraction"]
    if modules["PLE"]["critical_fraction"] >= 0.10 or modules["PLE"]["exposed_wait_fraction"] >= 0.05:
        eligible["PLE"] = max(modules["PLE"]["critical_fraction"],
                              modules["PLE"]["exposed_wait_fraction"])
    candidate = max(sorted(eligible), key=lambda name: eligible[name]) if eligible else None
    disposition = (f"B_ROUTE_CANDIDATE_{candidate}"
                   if candidate is not None else "NO_B_ROUTE_PERFORMANCE_CANDIDATE")
    return {
        "schema": 1,
        "status": "PASS_ROUND11_T5_HOTSPOT_RANKING",
        "parent_identity": parents.pop(),
        "process_blocks": "A_B",
        "sampled_layers": {
            "GDN": list(GDN_SAMPLE_LAYERS),
            "QSA": list(QSA_SAMPLE_LAYERS),
            "PLE": [1],
        },
        "actual_layer_counts": {"GDN": GDN_LAYER_COUNT, "QSA": QSA_LAYER_COUNT, "PLE": 1},
        "module_medians": modules,
        "request_details": request_rows,
        "candidate": candidate,
        "disposition": disposition,
        "screen": {
            "attributable_critical_fraction": 0.10,
            "exposed_wait_fraction": 0.05,
            "eligible": eligible,
        },
        "accounting": (
            "QSA_ATTENTION is the parent and already contains QSA_INDEXER. The indexer is "
            "reported as a child breakdown and is never added to the QSA total. PLE exposed "
            "wait is reported separately from device duration. Sampled layer medians are "
            "scaled by the frozen 36-GDN/12-QSA model census."
        ),
        "limitations": [
            "bounded intrusive observer; excluded from unprofiled performance",
            "device event durations are durations only and do not establish cross-stream absolute overlap",
            "sampled layers are real early/middle/late providers; whole-model cost is a frozen-census estimate",
        ],
        "cleanup": "BOTH_GUARD_ACK_CONTAINER_REMOVED_PHYSICAL12_EMPTY",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-a", type=Path, required=True)
    parser.add_argument("--block-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = rank_b_route_hotspots([
        json.loads(args.block_a.read_text()),
        json.loads(args.block_b.read_text()),
    ])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["disposition"], flush=True)


if __name__ == "__main__":
    main()
