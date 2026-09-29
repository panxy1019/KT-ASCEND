"""Build an exact, non-overlapping W2 Host-parent cost partition."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any

from .analyze_profile import percentile


TOP_LEVEL_COMPONENTS = ("GDN", "QSA_ATTENTION", "PLE_CONSUMER")


def _summary_ms(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median_ms": statistics.median(values) / 1e6,
        "p05_ms": percentile(values, 0.05) / 1e6,
        "p95_ms": percentile(values, 0.95) / 1e6,
        "mean_ms": statistics.fmean(values) / 1e6,
        "sum_ms": sum(values) / 1e6,
    }


def build_accounting(
    analysis: dict[str, Any], spans: list[dict[str, Any]]
) -> dict[str, Any]:
    if analysis.get("status") != "PASS_KT_ROOT_NATIVE_HOST_JOIN":
        raise ValueError("native/Host analysis is not qualified")
    selected: dict[int, dict[str, Any]] = {}
    for record in spans:
        if (
            record.get("status") == "ROUND13_P0_BATCH"
            and record.get("mode") == "decode"
            and "W2_MEASUREMENT" in record.get("request_id", "")
        ):
            selected[record["batch_sequence"]] = record

    rows: list[dict[str, int]] = []
    for source in analysis["batch_rows"]:
        sequence = source["batch_sequence"]
        record = selected.get(sequence)
        if record is None:
            raise ValueError(f"missing W2 span batch {sequence}")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for span in record["spans"]:
            grouped[span["component"]].append(span)
        if len(grouped["GDN"]) != 36:
            raise ValueError(f"batch {sequence} does not contain 36 GDN parents")
        if len(grouped["QSA_ATTENTION"]) != 12:
            raise ValueError(f"batch {sequence} does not contain 12 QSA parents")
        if len(grouped["PLE_CONSUMER"]) != 1:
            raise ValueError(f"batch {sequence} does not contain one PLE consumer")
        if len({row["layer"] for row in grouped["GDN"]}) != 36:
            raise ValueError(f"batch {sequence} repeats a GDN layer")
        if len({row["layer"] for row in grouped["QSA_ATTENTION"]}) != 12:
            raise ValueError(f"batch {sequence} repeats a QSA layer")

        gdn = sum(row["duration_ns"] for row in grouped["GDN"])
        qsa = sum(row["duration_ns"] for row in grouped["QSA_ATTENTION"])
        ple = sum(row["duration_ns"] for row in grouped["PLE_CONSUMER"])
        expert = source["routed_expert_parent_sum_ns"]
        wall = source["scheduler_wall_ns"]
        classified = gdn + qsa + ple + expert
        residual = wall - classified
        if residual < 0:
            raise ValueError(f"top-level parents exceed scheduler wall in batch {sequence}")
        rows.append(
            {
                "batch_sequence": sequence,
                "scheduler_wall_ns": wall,
                "gdn_ns": gdn,
                "qsa_attention_ns": qsa,
                "routed_expert_ns": expert,
                "ple_consumer_ns": ple,
                "other_dense_dispatch_ns": residual,
                "cpu_task_ns_nested": source["cpu_task_sum_ns_nested_with_npu"],
                "cpu_exposed_wait_ns": source["native_sync_wait_sum_ns"],
            }
        )

    keys = (
        "scheduler_wall_ns",
        "gdn_ns",
        "qsa_attention_ns",
        "routed_expert_ns",
        "ple_consumer_ns",
        "other_dense_dispatch_ns",
        "cpu_task_ns_nested",
        "cpu_exposed_wait_ns",
    )
    total_wall = sum(row["scheduler_wall_ns"] for row in rows)
    fractions = {
        key: sum(row[key] for row in rows) / total_wall
        for key in (
            "gdn_ns",
            "qsa_attention_ns",
            "routed_expert_ns",
            "ple_consumer_ns",
            "other_dense_dispatch_ns",
            "cpu_exposed_wait_ns",
        )
    }
    return {
        "schema": 1,
        "status": "PASS_EXACT_W2_HOST_PARENT_ACCOUNTING",
        "scope": "EXACT_ALL_LAYER_HOST_PARENT_PARTITION",
        "request_marker": "W2_MEASUREMENT",
        "steady_decode_tokens": len(rows),
        "coverage": {
            "gdn_layers_per_token": 36,
            "qsa_layers_per_token": 12,
            "routed_expert_layers_per_token": 48,
            "ple_consumers_per_token": 1,
        },
        "summaries": {key: _summary_ms([row[key] for row in rows]) for key in keys},
        "fractions_of_scheduler_wall": fractions,
        "nested_not_additive": {
            "cpu_task_ns_nested": "inside routed_expert_ns",
            "qsa_indexer": "inside qsa_attention_ns and intentionally excluded",
            "ple_prepare": "asynchronous producer and intentionally excluded",
            "cpu_exposed_wait_ns": "inside routed_expert_ns",
        },
        "other_dense_dispatch_definition": (
            "scheduler wall minus exact GDN, QSA attention, routed-expert, and PLE "
            "consumer Host parents; includes dense projections, router/norm, sampler, "
            "scheduler gaps, and observer overhead not otherwise classified"
        ),
        "token_rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, int]]) -> None:
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
    result = build_accounting(analysis, spans)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_csv, result["token_rows"])
    print(result["status"])


if __name__ == "__main__":
    main()
