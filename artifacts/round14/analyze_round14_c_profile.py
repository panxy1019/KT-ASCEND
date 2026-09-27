#!/usr/bin/env python3
"""Validate two fresh Round14 C S0 profiles and rank attributable costs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


EXPECTED = ((274, "A"), (275, "B"))
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
        raise RuntimeError(f"invalid {label}")
    return float(value)


def _validate_span(span):
    component = span.get("component")
    if component not in SAMPLES or span.get("layer") not in SAMPLES[component]:
        raise RuntimeError("unexpected Round14 C component/layer span")
    start, end = span.get("start_ns"), span.get("end_ns")
    expected_device = "COMPLETE" if component in DEVICE_COMPONENTS else "NOT_APPLICABLE"
    if (
        type(start) is not int or type(end) is not int or end < start
        or span.get("duration_ns") != end - start
        or span.get("outcome") != "return"
        or span.get("observer_errors") != []
        or span.get("device_status") != expected_device
    ):
        raise RuntimeError("invalid Round14 C profile span")
    if component in DEVICE_COMPONENTS:
        _positive(span.get("device_ms"), "device span")
    elif span.get("device_ms") is not None:
        raise RuntimeError("host Round14 C span unexpectedly has device time")
    return span


def _validate_batch(batch, request_ids):
    if (
        batch.get("status") != "ROUND13_P0_BATCH"
        or batch.get("request_id") not in request_ids
        or batch.get("mode") not in {"extend", "decode"}
        or type(batch.get("batch_sequence")) is not int
        or batch.get("dropped_spans") != 0
        or batch.get("observer_errors") != []
    ):
        raise RuntimeError("Round14 C batch contract failed")
    spans = [_validate_span(span) for span in batch.get("spans", [])]
    by = {(span["component"], span["layer"]): span for span in spans}
    expected = {
        (component, layer) for component, layers in SAMPLES.items() for layer in layers
    }
    if len(by) != len(spans) or set(by) != expected:
        raise RuntimeError("Round14 C batch sample coverage mismatch")
    for layer in SAMPLES["QSA_ATTENTION"]:
        parent = by[("QSA_ATTENTION", layer)]
        child = by[("QSA_INDEXER", layer)]
        if not (
            parent["start_ns"] <= child["start_ns"]
            <= child["end_ns"] <= parent["end_ns"]
        ):
            raise RuntimeError("Round14 C QSA child is not nested in parent")
    for layer in SAMPLES["ROUTED_EXPERT"]:
        parent = by[("ROUTED_EXPERT", layer)]
        for component in ("CPU_SUBMIT", "CPU_RESULT_WAIT", "NPU_RESIDENT"):
            child = by[(component, layer)]
            if not (
                parent["start_ns"] <= child["start_ns"]
                <= child["end_ns"] <= parent["end_ns"]
            ):
                raise RuntimeError("Round14 C Expert child is not nested in parent")
    return batch


def _validate_process(document):
    identity = (document.get("version"), document.get("block"))
    if (
        identity not in EXPECTED
        or document.get("status") != "PASS_ROUND14_C_PROFILE_PROCESS"
        or document.get("parent") != "ROUND13_FINAL_S0_UNIFORM_Q32_PROMOTED_QSA"
        or document.get("profile_threshold") != 513
        or document.get("default_threshold") != 129
        or document.get("profiler") != "BOUNDED_OBSERVER_ON"
        or document.get("generation_budget") != 4
        or document.get("generation_attempts") != 4
        or document.get("abort_controls") != 0
        or document.get("automatic_retries") != 0
        or document.get("cleanup") != EXPECTED_CLEANUP
    ):
        raise RuntimeError("Round14 C process contract failed")
    results = document.get("results")
    if (
        not isinstance(results, list) or len(results) != 4
        or [row.get("ordinal") for row in results] != [1, 2, 3, 4]
        or [row.get("workload") for row in results]
            != ["WARMUP", "W2", "W3", "DRAIN"]
    ):
        raise RuntimeError("Round14 C request schedule mismatch")
    measurements = [row for row in results if row.get("phase") == "measurement"]
    if [row.get("workload") for row in measurements] != ["W2", "W3"]:
        raise RuntimeError("Round14 C measurement coverage mismatch")
    request_ids = set()
    for row in results:
        if (
            row.get("finite_logprobs") is not True
            or row.get("finish_reason") != "length"
            or row.get("completion_tokens") != row.get("decode_tokens", row.get("completion_tokens"))
        ):
            raise RuntimeError("Round14 C output contract failed")
        _positive(row.get("ttft_seconds"), "TTFT")
        _positive(row.get("e2e_seconds"), "E2E")
        qsa = row.get("qsa", {})
        if (
            qsa.get("prefill_strategy") != "VECTORIZED"
            or qsa.get("decode_strategy") != "DIRECT_BF16_ROW"
            or qsa.get("actual_completion_tokens") != row.get("completion_tokens")
            or qsa.get("first_completion_token_from_prefill_logits") is not True
        ):
            raise RuntimeError("Round14 C QSA execution contract failed")
        if row["workload"] in {"W2", "W3"} and row.get("output_ids", [])[:2] != [3897, 248046]:
            raise RuntimeError("Round14 C ACK prefix changed")
        request_ids.add(row["request_id"])
    measurement_ids = {row["request_id"] for row in measurements}
    batches = [
        _validate_batch(batch, measurement_ids) for batch in document.get("batches", [])
    ]
    identities = set()
    modes = {request_id: set() for request_id in measurement_ids}
    for batch in batches:
        identity = (batch["request_id"], batch["mode"], batch["batch_sequence"])
        if identity in identities:
            raise RuntimeError("duplicate Round14 C batch identity")
        identities.add(identity)
        modes[batch["request_id"]].add(batch["mode"])
    if not batches or any(value != {"extend", "decode"} for value in modes.values()):
        raise RuntimeError("Round14 C profile lacks extend/decode coverage")
    return measurements, batches


def _batch_cost(batch):
    costs = {}
    for component in SAMPLES:
        rows = [span for span in batch["spans"] if span["component"] == component]
        values = [
            span["device_ms"] if component in DEVICE_COMPONENTS
            else span["duration_ns"] / 1e6
            for span in rows
        ]
        costs[component] = statistics.median(values) * SCALE[component]
    costs["QSA_EXCLUSIVE"] = max(
        0.0, costs["QSA_ATTENTION"] - costs["QSA_INDEXER"]
    )
    costs["EXPERT_EXCLUSIVE_DEVICE"] = max(
        0.0, costs["ROUTED_EXPERT"] - costs["NPU_RESIDENT"]
    )
    return costs


def analyze(documents):
    if len(documents) != 2:
        raise ValueError("Round14 C attribution requires two fresh S0 blocks")
    observed = [(row.get("version"), row.get("block")) for row in documents]
    if tuple(observed) != EXPECTED:
        raise RuntimeError(f"Round14 C process order/identity mismatch: {observed}")
    if len({row.get("fixture_sha256") for row in documents}) != 1:
        raise RuntimeError("Round14 C fixture identity drift")
    rows = []
    for document in documents:
        measurements, batches = _validate_process(document)
        for request in measurements:
            rid = request["request_id"]
            mode_costs = {}
            for mode in ("extend", "decode"):
                selected = [
                    _batch_cost(batch) for batch in batches
                    if batch["request_id"] == rid and batch["mode"] == mode
                ]
                mode_costs[mode] = {
                    component: sum(item[component] for item in selected)
                    for component in selected[0]
                }
            phase_wall = {
                "extend": request["ttft_seconds"] * 1000.0,
                "decode": max(
                    1e-9, request["e2e_seconds"] - request["ttft_seconds"]
                ) * 1000.0,
            }
            fractions = {
                mode: {
                    component: value / phase_wall[mode]
                    for component, value in mode_costs[mode].items()
                }
                for mode in ("extend", "decode")
            }
            rows.append({
                "block": document["block"],
                "version": document["version"],
                "workload": request["workload"],
                "request_id": rid,
                "phase_wall_ms": phase_wall,
                "attributable_ms": mode_costs,
                "fractions": fractions,
                "profile_batch_count": {
                    mode: sum(
                        batch["request_id"] == rid and batch["mode"] == mode
                        for batch in batches
                    ) for mode in ("extend", "decode")
                },
            })

    definitions = {
        "GDN": ("GDN", 0.10),
        "QSA": ("QSA_ATTENTION", 0.10),
        "PLE": ("PLE_CONSUMER", 0.10),
        "EXPERT": ("ROUTED_EXPERT", 0.10),
        "PLE_EXPOSED_WAIT": ("PLE_EXPOSED_WAIT", 0.05),
        "CPU_EXPOSED_WAIT": ("CPU_RESULT_WAIT", 0.05),
    }
    screen = {}
    eligible = []
    ranking = []
    for name, (component, threshold) in definitions.items():
        phase_support = {}
        eligible_phases = []
        for mode in ("extend", "decode"):
            values = {
                block: max(
                    row["fractions"][mode][component]
                    for row in rows if row["block"] == block
                )
                for block in ("A", "B")
            }
            phase_support[mode] = values
            if all(value >= threshold for value in values.values()):
                eligible_phases.append(mode)
            ranking.append({
                "module": name,
                "component": component,
                "phase": mode,
                "median_fraction_across_blocks_and_workloads": statistics.median(
                    row["fractions"][mode][component] for row in rows
                ),
                "threshold": threshold,
                "both_blocks": mode in eligible_phases,
            })
        screen[name] = {
            "component": component,
            "threshold": threshold,
            "per_block_phase_max_fraction": phase_support,
            "eligible_phases": eligible_phases,
            "both_blocks": bool(eligible_phases),
        }
        if eligible_phases and name in {"GDN", "QSA", "PLE", "EXPERT"}:
            eligible.append(name)
    ranking.sort(
        key=lambda row: row["median_fraction_across_blocks_and_workloads"],
        reverse=True,
    )
    return {
        "schema": 1,
        "status": "PASS_ROUND14_C_FRESH_ATTRIBUTION",
        "process_order": [row["version"] for row in documents],
        "fixture_sha256": documents[0]["fixture_sha256"],
        "requests": rows,
        "screen": screen,
        "eligible_hotspots": eligible,
        "ranking": ranking,
        "candidate_gate": "PROFILE_COMPLETE_REQUIRES_ONE_HOTSPOT_CODE_AUDIT",
        "accounting": (
            "QSA_ATTENTION and ROUTED_EXPERT are parents; QSA_INDEXER, CPU_SUBMIT, "
            "CPU_RESULT_WAIT, and NPU_RESIDENT are child breakdowns and are never "
            "added to their parents. Device event time and host exposed waits remain "
            "separate clock domains."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-a", type=Path, required=True)
    parser.add_argument("--block-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = analyze([
        json.loads(args.block_a.read_text()),
        json.loads(args.block_b.read_text()),
    ])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["status"], report["eligible_hotspots"], flush=True)


if __name__ == "__main__":
    main()
