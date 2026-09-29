"""Pure analysis helpers for Round17 complete-parent profile evidence."""

from __future__ import annotations

from collections import defaultdict
import statistics

from .attribution import exclusive_account


QSA_CHILDREN = frozenset({
    "QSA_INPUT_PROJ",
    "QSA_QK_NORM",
    "QSA_ROPE",
    "QSA_INDEXER",
    "QSA_INDEX_PROJ",
    "QSA_INDEX_Q_NORM",
    "QSA_INDEX_K_NORM",
    "QSA_SPARSE_CORE",
    "QSA_OUTPUT_PROJ",
})


def select_steady_decode(rows):
    selected = [
        row for row in rows
        if row.get("mode") == "decode"
        and ":W2_MEASUREMENT" in row.get("request_id", "")
    ]
    selected.sort(key=lambda row: row["batch_sequence"])
    if len(selected) < 3:
        raise ValueError("W2 measurement lacks three Decode batches")
    return selected[1:-1]


def qsa_exclusive_ns(spans):
    total = 0
    parents = [span for span in spans if span["component"] == "QSA_ATTENTION"]
    for parent in parents:
        children = [
            (child["start_ns"], child["end_ns"])
            for child in spans
            if child["component"] in QSA_CHILDREN
            and child["layer"] == parent["layer"]
            and parent["start_ns"] <= child["start_ns"]
            and child["end_ns"] <= parent["end_ns"]
        ]
        total += exclusive_account(
            (parent["start_ns"], parent["end_ns"]), children
        )["exclusive_ns"]
    return total


def glue_intervals_ns(spans):
    attentions = {
        span["layer"]: span for span in spans
        if span["component"] in {"GDN", "QSA_ATTENTION"}
    }
    experts = {
        span["layer"]: span for span in spans
        if span["component"] == "ROUTED_EXPERT"
    }
    decoder = {
        span["layer"]: span for span in spans
        if span["component"] == "DECODER_LAYER"
    }
    common = sorted(set(attentions) & set(experts))
    if not common:
        raise ValueError("batch lacks attention/expert dependency spans")
    attention_to_expert = sum(
        max(0, experts[layer]["start_ns"] - attentions[layer]["end_ns"])
        for layer in common
    )
    expert_to_next = 0
    for layer in common:
        if layer + 1 in attentions:
            expert_to_next += max(
                0, attentions[layer + 1]["start_ns"] - experts[layer]["end_ns"]
            )
    last = common[-1]
    last_tail = 0
    if last in decoder:
        last_tail = max(0, decoder[last]["end_ns"] - experts[last]["end_ns"])
    return {
        "attention_to_expert_ns": attention_to_expert,
        "expert_to_next_attention_ns": expert_to_next,
        "last_expert_to_layer_end_ns": last_tail,
    }


def _summary(values):
    values = list(values)
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def analyze_profile(rows):
    steady = select_steady_decode(rows)
    host_by_component = defaultdict(list)
    device_by_component = defaultdict(list)
    qsa_exclusive = []
    glue = defaultdict(list)
    scheduler_wall = []
    span_counts = defaultdict(list)
    for batch in steady:
        host_totals = defaultdict(float)
        device_totals = defaultdict(float)
        counts = defaultdict(int)
        for span in batch["spans"]:
            component = span["component"]
            host_totals[component] += span["duration_ns"] / 1_000_000
            counts[component] += 1
            if span.get("device_status") == "COMPLETE":
                device_totals[component] += span["device_ms"]
        for component, value in host_totals.items():
            host_by_component[component].append(value)
        for component, value in device_totals.items():
            device_by_component[component].append(value)
        for component, value in counts.items():
            span_counts[component].append(value)
        qsa_exclusive.append(qsa_exclusive_ns(batch["spans"]) / 1_000_000)
        for name, value in glue_intervals_ns(batch["spans"]).items():
            glue[name].append(value / 1_000_000)
        scheduler_wall.append(batch["duration_ns"] / 1_000_000)
    return {
        "schema": 1,
        "status": "PASS_ROUND17_PROFILE_ANALYSIS",
        "request_id": steady[0]["request_id"],
        "steady_decode_batches": len(steady),
        "excluded_decode_batches": 2,
        "scheduler_wall_ms_per_token": _summary(scheduler_wall),
        "host_ms_per_token": {
            name: _summary(values) for name, values in sorted(host_by_component.items())
        },
        "device_ms_per_token": {
            name: _summary(values) for name, values in sorted(device_by_component.items())
        },
        "span_count_per_token": {
            name: _summary(values) for name, values in sorted(span_counts.items())
        },
        "qsa_uninstrumented_host_ms_per_token": _summary(qsa_exclusive),
        "glue_ms_per_token": {
            name: _summary(values) for name, values in sorted(glue.items())
        },
        "clock_rule": "HOST_AND_DEVICE_REPORTED_SEPARATELY_NOT_ADDED",
    }
