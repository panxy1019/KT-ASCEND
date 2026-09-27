#!/usr/bin/env python3
"""Validate two Round13 P0 blocks, rank costs, and freeze fresh placements."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from round13_placement import build_profiles, profile_receipt


SAMPLES = {
    "GDN": (0, 24, 46),
    "QSA_ATTENTION": (3, 23, 47),
    "QSA_INDEXER": (3, 23, 47),
    "PLE_PREPARE": (1,),
    "PLE_EXPOSED_WAIT": (1,),
    "PLE_CONSUMER": (1,),
    "ROUTED_EXPERT": (0, 24, 47),
    "CPU_SUBMIT": (0, 24, 47),
    "CPU_RESULT_WAIT": (0, 24, 47),
    "NPU_RESIDENT": (0, 24, 47),
}
DEVICE_COMPONENTS = frozenset((
    "GDN", "QSA_ATTENTION", "QSA_INDEXER", "PLE_CONSUMER",
    "ROUTED_EXPERT", "NPU_RESIDENT",
))
SCALE = {
    "GDN": 36,
    "QSA_ATTENTION": 12,
    "QSA_INDEXER": 12,
    "ROUTED_EXPERT": 48,
    "CPU_SUBMIT": 48,
    "CPU_RESULT_WAIT": 48,
    "NPU_RESIDENT": 48,
    "PLE_PREPARE": 1,
    "PLE_EXPOSED_WAIT": 1,
    "PLE_CONSUMER": 1,
}
EXPECTED_CLEANUP = {
    "guard_exit_ack": True,
    "container_removed": True,
    "physical12_after": "EMPTY",
}


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {label}")
    return float(value)


def _validate_span(row):
    component = row.get("component")
    if component not in SAMPLES or row.get("layer") not in SAMPLES[component]:
        raise ValueError("unexpected P0 component/layer")
    start, end = row.get("start_ns"), row.get("end_ns")
    if (type(start) is not int or type(end) is not int or end < start or
            row.get("duration_ns") != end - start or
            row.get("observer_errors") != [] or row.get("outcome") != "return"):
        raise ValueError("invalid P0 span")
    if component in DEVICE_COMPONENTS:
        if row.get("device_status") != "COMPLETE":
            raise ValueError("P0 device span is incomplete")
        _positive(row.get("device_ms"), "device duration")
    elif row.get("device_status") != "NOT_APPLICABLE" or row.get("device_ms") is not None:
        raise ValueError("P0 host span has device timing")
    return row


def _validate_batch(batch, request_ids):
    if (batch.get("request_id") not in request_ids or
            batch.get("mode") not in {"extend", "decode"} or
            batch.get("status") != "ROUND13_P0_BATCH" or
            batch.get("dropped_spans") != 0 or
            batch.get("observer_errors") != []):
        raise ValueError("invalid P0 batch")
    spans = [_validate_span(row) for row in batch.get("spans", [])]
    by = {}
    for row in spans:
        key = (row["component"], row["layer"])
        if key in by:
            raise ValueError("duplicate P0 span")
        by[key] = row
    expected = {(component, layer) for component, layers in SAMPLES.items()
                for layer in layers}
    if set(by) != expected:
        raise ValueError("P0 batch coverage mismatch")
    for parent, child in (("QSA_ATTENTION", "QSA_INDEXER"),):
        for layer in SAMPLES[parent]:
            p, c = by[(parent, layer)], by[(child, layer)]
            if not (p["start_ns"] <= c["start_ns"] <= c["end_ns"] <= p["end_ns"]):
                raise ValueError("QSA child is not nested in its parent")
    for layer in SAMPLES["ROUTED_EXPERT"]:
        parent = by[("ROUTED_EXPERT", layer)]
        for child in ("CPU_SUBMIT", "CPU_RESULT_WAIT", "NPU_RESIDENT"):
            row = by[(child, layer)]
            if not (parent["start_ns"] <= row["start_ns"] <= row["end_ns"] <= parent["end_ns"]):
                raise ValueError("Expert child is not nested in its parent")
    return batch


def _validate_process(document):
    if (document.get("status") != "PASS_ROUND13_P0_PROCESS" or
            document.get("block") not in {"A", "B"} or
            document.get("profiler") != "BOUNDED_OBSERVER_ON" or
            document.get("profile_threshold") != 513 or
            document.get("default_threshold") != 129 or
            document.get("generation_attempts") != 7 or
            document.get("automatic_retries") != 0 or
            document.get("cleanup") != EXPECTED_CLEANUP):
        raise ValueError("invalid P0 process identity")
    requests = document.get("requests")
    if (not isinstance(requests, list) or len(requests) != 2 or
            {row.get("case_id") for row in requests} != {"edge128", "edge512"}):
        raise ValueError("P0 W2/W3 measurement coverage mismatch")
    by_id = {}
    for row in requests:
        if (row.get("phase") != "measurement" or row.get("path") != "HYBRID" or
                row.get("finite_logprobs") is not True):
            raise ValueError("invalid P0 request")
        _positive(row.get("e2e_seconds"), "P0 E2E")
        _positive(row.get("ttft_seconds"), "P0 TTFT")
        _positive(row.get("tpot_seconds"), "P0 TPOT")
        by_id[row["request_id"]] = row
    batches = [_validate_batch(row, set(by_id)) for row in document.get("batches", [])]
    for request_id in by_id:
        if {row["mode"] for row in batches if row["request_id"] == request_id} != {"extend", "decode"}:
            raise ValueError("P0 request lacks phase coverage")
    return by_id, batches


def _batch_cost(batch):
    def value(component):
        rows = [row for row in batch["spans"] if row["component"] == component]
        values = [(row["device_ms"] if component in DEVICE_COMPONENTS
                   else row["duration_ns"] / 1e6) for row in rows]
        return statistics.median(values) * SCALE[component]
    costs = {name: value(name) for name in SAMPLES}
    costs["QSA_EXCLUSIVE"] = max(0.0, costs["QSA_ATTENTION"] - costs["QSA_INDEXER"])
    costs["EXPERT_EXCLUSIVE_DEVICE"] = max(
        0.0, costs["ROUTED_EXPERT"] - costs["NPU_RESIDENT"])
    return costs


def _merge_route_counts(documents):
    counts = [[0] * 512 for _ in range(48)]
    per_mode = {mode: [[0] * 512 for _ in range(48)]
                for mode in ("extend", "decode")}
    for document in documents:
        routes = document.get("dev_routes", {})
        if routes.get("status") != "ROUND13_FRESH_DEV_ROUTE_COUNTS":
            raise ValueError("P0 route counts missing")
        for mode in per_mode:
            values = routes.get("counts", {}).get(mode)
            if not isinstance(values, list) or len(values) != 48:
                raise ValueError("P0 route count shape mismatch")
            for layer in range(48):
                if len(values[layer]) != 512:
                    raise ValueError("P0 route expert count shape mismatch")
                for expert in range(512):
                    value = values[layer][expert]
                    if type(value) is not int or value < 0:
                        raise ValueError("P0 route count value invalid")
                    per_mode[mode][layer][expert] += value
                    counts[layer][expert] += value
    return counts, per_mode


def analyze(documents):
    if len(documents) != 2:
        raise ValueError("P0 requires two independent S0 blocks")
    validated = []
    for document in documents:
        requests, batches = _validate_process(document)
        validated.append((document, requests, batches))
    if {row[0]["block"] for row in validated} != {"A", "B"}:
        raise ValueError("P0 block coverage mismatch")
    if len({row[0]["parent_identity"] for row in validated}) != 1:
        raise ValueError("P0 parent identity mismatch")

    rows = []
    for document, requests, batches in validated:
        for request_id, request in requests.items():
            mode_costs = {}
            for mode in ("extend", "decode"):
                selected = [_batch_cost(batch) for batch in batches
                            if batch["request_id"] == request_id and batch["mode"] == mode]
                mode_costs[mode] = {
                    name: sum(cost[name] for cost in selected)
                    for name in selected[0]
                }
            phase_ms = {
                "extend": request["ttft_seconds"] * 1000.0,
                "decode": max(1e-9, request["e2e_seconds"] - request["ttft_seconds"]) * 1000.0,
            }
            fractions = {
                mode: {
                    "GDN": mode_costs[mode]["GDN"] / phase_ms[mode],
                    "QSA": mode_costs[mode]["QSA_ATTENTION"] / phase_ms[mode],
                    "PLE_CONSUMER": mode_costs[mode]["PLE_CONSUMER"] / phase_ms[mode],
                    "PLE_EXPOSED_WAIT": mode_costs[mode]["PLE_EXPOSED_WAIT"] / phase_ms[mode],
                    "EXPERT": mode_costs[mode]["ROUTED_EXPERT"] / phase_ms[mode],
                    "CPU_EXPOSED_WAIT": mode_costs[mode]["CPU_RESULT_WAIT"] / phase_ms[mode],
                }
                for mode in ("extend", "decode")
            }
            rows.append({
                "block": document["block"],
                "case_id": request["case_id"],
                "request_id": request_id,
                "phase_wall_ms": phase_ms,
                "attributable_ms": mode_costs,
                "fractions": fractions,
            })

    def block_support(module, *, wait=False):
        threshold = 0.05 if wait else 0.10
        support = {}
        for block in ("A", "B"):
            values = []
            for row in rows:
                if row["block"] != block:
                    continue
                for mode in ("extend", "decode"):
                    values.append(row["fractions"][mode][module])
            support[block] = max(values) if values else None
        return {"threshold": threshold, "per_block_max_fraction": support,
                "both_blocks": all(value is not None and value >= threshold
                                   for value in support.values())}

    screen = {
        "GDN": block_support("GDN"),
        "QSA": block_support("QSA"),
        "PLE": block_support("PLE_CONSUMER"),
        "PLE_EXPOSED_WAIT": block_support("PLE_EXPOSED_WAIT", wait=True),
        "EXPERT": block_support("EXPERT"),
        "CPU_EXPOSED_WAIT_FOR_C": block_support("CPU_EXPOSED_WAIT"),
    }
    counts, per_mode = _merge_route_counts(documents)
    profiles = build_profiles(counts)
    profile_document = {
        "schema": 2,
        "status": "FROZEN_ROUND13_PLACEMENTS_FROM_FRESH_P0_DEV",
        "dev_source": "TWO_INDEPENDENT_S0_BLOCKS_DEV_ONLY_D2H",
        "frequency_tie_break": "COUNT_DESC_LAYER_ASC_EXPERT_ASC",
        "random_algorithm": "SHA256_COUNTER_PERMUTATION_V1",
        "main_random_seed": 42,
        "profiles": {name: profile_receipt(name, profile, counts)
                     for name, profile in profiles.items()},
        "fresh_dev_counts": counts,
        "fresh_dev_counts_by_mode": per_mode,
    }
    eligible = [name for name in ("GDN", "QSA", "PLE")
                if screen[name]["both_blocks"]]
    result = {
        "schema": 1,
        "status": "PASS_ROUND13_P0_FRESH_ATTRIBUTION",
        "parent_identity": validated[0][0]["parent_identity"],
        "process_blocks": ["A", "B"],
        "requests": rows,
        "screen": screen,
        "b_candidate_pool": eligible,
        "b_preliminary_disposition": (
            "HOTSPOT_CANDIDATE_PRESENT" if eligible else
            "NO_B_ROUTE_PERFORMANCE_CANDIDATE_BY_COMPLETE_ATTRIBUTION"
        ),
        "c_cpu_wait_trigger": screen["CPU_EXPOSED_WAIT_FOR_C"]["both_blocks"],
        "accounting": (
            "QSA_ATTENTION and ROUTED_EXPERT are parents. Child spans are breakdowns "
            "and are never added to their parent. Device event durations are queried "
            "at later scheduler boundaries without synchronization."
        ),
        "placement_profile_status": profile_document["status"],
    }
    return result, profile_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-a", type=Path, required=True)
    parser.add_argument("--block-b", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--placements", type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists() or args.placements.exists():
        raise FileExistsError("P0 analysis outputs must be fresh")
    result, placements = analyze([
        json.loads(args.block_a.read_text()),
        json.loads(args.block_b.read_text()),
    ])
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    args.placements.write_text(json.dumps(placements, indent=2) + "\n")
    print(result["status"], result["b_preliminary_disposition"], flush=True)


if __name__ == "__main__":
    main()
