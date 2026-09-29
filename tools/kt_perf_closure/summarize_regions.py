"""Summarize bounded Round13 samples without double-counting nested spans."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any

from .analyze_profile import percentile


EXPECTED_TOTAL_LAYERS = {"GDN": 36, "QSA_ATTENTION": 12}
NESTED_IN = {
    "QSA_INDEXER": "QSA_ATTENTION",
    "CPU_SUBMIT": "ROUTED_EXPERT",
    "CPU_RESULT_WAIT": "ROUTED_EXPERT",
    "NPU_RESIDENT": "ROUTED_EXPERT",
}


def summary_ms(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty series")
    return {
        "count": len(values),
        "median_ms": statistics.median(values),
        "p05_ms": percentile(values, 0.05),
        "p95_ms": percentile(values, 0.95),
        "mean_ms": statistics.fmean(values),
    }


def analyze_sampled_spans(
    rows: list[dict[str, Any]],
    *,
    request_marker: str = "W2_MEASUREMENT",
    warm_decode_batches: int = 2,
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if request_marker in row.get("request_id", "") and row.get("mode") == "decode"
    ]
    selected.sort(key=lambda row: row["batch_sequence"])
    selected = selected[warm_decode_batches:]
    if not selected:
        raise ValueError("no steady sampled Decode batches")
    if any(row.get("dropped_spans") for row in selected):
        raise ValueError("sample observer dropped spans")
    if any(row.get("observer_errors") for row in selected):
        raise ValueError("sample observer recorded errors")

    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_token: dict[str, list[float]] = defaultdict(list)
    component_names = sorted({
        span["component"] for row in selected for span in row.get("spans", [])
    })
    for row in selected:
        token_values: dict[str, float] = defaultdict(float)
        for span in row.get("spans", []):
            by_component[span["component"]].append(span)
            token_values[span["component"]] += span["duration_ns"] / 1e6
        for component in component_names:
            per_token[component].append(token_values[component])

    components: dict[str, Any] = {}
    for component in component_names:
        spans = by_component[component]
        host_values = [span["duration_ns"] / 1e6 for span in spans]
        device_values = [
            float(span["device_ms"])
            for span in spans
            if span.get("device_status") == "COMPLETE" and span.get("device_ms") is not None
        ]
        layers = sorted({int(span["layer"]) for span in spans})
        components[component] = {
            "sampled_layers": layers,
            "host_per_call": summary_ms(host_values),
            "host_sampled_sum_per_token": summary_ms(per_token[component]),
            "device_per_call": summary_ms(device_values) if device_values else None,
            "nested_in": NESTED_IN.get(component),
        }

    scaled: dict[str, Any] = {}
    for component, total_layers in EXPECTED_TOTAL_LAYERS.items():
        if component not in components:
            continue
        sampled_layers = len(components[component]["sampled_layers"])
        if sampled_layers == 0:
            raise ValueError(f"{component} has no sampled layer")
        scale = total_layers / sampled_layers
        values = [value * scale for value in per_token[component]]
        scaled[component] = {
            **summary_ms(values),
            "scale": scale,
            "sampled_layer_count": sampled_layers,
            "total_layer_count": total_layers,
            "scope": "NON_ADDITIVE_SAMPLED_EXTRAPOLATION",
        }

    return {
        "schema": 1,
        "status": "PASS_BOUNDED_REGION_SUMMARY",
        "request_marker": request_marker,
        "decode_batches_steady": len(selected),
        "components": components,
        "scaled_nonadditive_estimates_ms_per_token": scaled,
        "scaled_estimates_are_nonadditive": True,
        "warning": (
            "Scaled GDN/QSA samples are ranges for ranking only; QSA_INDEXER is "
            "nested in QSA_ATTENTION and exact Expert/non-Expert partitions come "
            "from the direct full-model Host trace."
        ),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.spans.read_text().splitlines()]
    result = analyze_sampled_spans(rows)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
