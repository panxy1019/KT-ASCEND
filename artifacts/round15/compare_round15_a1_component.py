#!/usr/bin/env python3
"""Freeze the strict two-block Round15 QSA component disposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_ROWS = {(layer, context) for layer in (3, 23, 47) for context in (32, 64, 128, 256)}


def summarize(document):
    rows = document.get("full_set_rows", [])
    row_keys = {(row.get("layer"), row.get("context_tokens")) for row in rows}
    numerical = all(
        row.get("selected_set_exact") is True
        and isinstance(row.get("downstream_relative_l2"), (int, float))
        and row["downstream_relative_l2"] <= 0.001
        and len(row.get("repeat_hashes", [])) == 3
        and len(set(row.get("repeat_hashes", []))) == 1
        for row in rows
    )
    performance = (
        len(rows) == 12
        and row_keys == EXPECTED_ROWS
        and all(
            isinstance(row.get("candidate_over_baseline"), (int, float))
            and 0 < row["candidate_over_baseline"] <= 0.85
            for row in rows
        )
    )
    return {
        "block": document.get("block"),
        "status": document.get("status"),
        "candidate": document.get("selected_component_candidate"),
        "source_identity": document.get("source_identity"),
        "row_count": len(rows),
        "row_keys": sorted([list(key) for key in row_keys]),
        "ratios": [row.get("candidate_over_baseline") for row in rows],
        "worst_ratio": max(
            (row.get("candidate_over_baseline", float("inf")) for row in rows),
            default=None,
        ),
        "max_downstream_relative_l2": max(
            (row.get("downstream_relative_l2", float("inf")) for row in rows),
            default=None,
        ),
        "numerical_pass": numerical,
        "performance_pass": performance,
        "lifecycle_pass": document.get("lifecycle_verdict", {}).get("pass") is True,
        "process_component_gate": document.get("component_gate_pass") is True,
        "peak_npu_allocated": document.get("peak_npu_allocated"),
        "lightning_disposition": document.get("lightning_disposition"),
    }


def compare(documents):
    blocks = [summarize(document) for document in documents]
    by_block = {row["block"]: row for row in blocks}
    if set(by_block) != {"A", "B"} or len(blocks) != 2:
        return {
            "schema": 1,
            "status": "ROUND15_A1_QSA_COMPONENT_PARTIAL",
            "blocks": blocks,
            "request_smoke_gate_open": False,
        }
    source_match = by_block["A"]["source_identity"] == by_block["B"]["source_identity"]
    candidate_match = (
        by_block["A"]["candidate"]
        == by_block["B"]["candidate"]
        == "FULL_SET_CAUSAL_SHORT_CONTEXT"
    )
    correctness = all(
        row["status"] == "PASS_ROUND15_A1_QSA_COMPONENT_PROCESS"
        and row["row_count"] == 12
        and row["numerical_pass"]
        and row["lifecycle_pass"]
        and row["process_component_gate"]
        and isinstance(row["peak_npu_allocated"], int)
        and row["peak_npu_allocated"] <= 10240 * 1024**2
        for row in by_block.values()
    )
    performance = all(row["performance_pass"] for row in by_block.values())
    passed = correctness and performance and source_match and candidate_match
    return {
        "schema": 1,
        "status": (
            "PASS_ROUND15_A1_QSA_TWO_BLOCK_COMPONENT_GATE"
            if passed
            else "ROUND15_A1_QSA_COMPONENT_NO_PROMOTION"
        ),
        "candidate_family": "QSA_INDEX_SELECTION",
        "selected_candidate": "FULL_SET_CAUSAL_SHORT_CONTEXT" if candidate_match else None,
        "blocks": [by_block["A"], by_block["B"]],
        "source_identity_match": source_match,
        "candidate_identity_match": candidate_match,
        "correctness_pass": correctness,
        "all_24_full_parent_rows_improve_15pct": performance,
        "request_gate": "DECODE_W2",
        "request_smoke_gate_open": passed,
        "threshold": (
            "both independent blocks: 12/12 full index-selection parent host medians "
            "candidate/baseline <=0.85; selected set exact; downstream rel-L2 <=0.001; "
            "repeat/lifecycle/resource/source identity pass"
        ),
        "lightning_disposition": (
            "SEMANTICALLY_EXPRESSIBLE_OUTSIDE_CURRENT_REQUEST_RANGE; "
            "not selected because <=256-token qualified requests select every completed group"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs=2, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare([json.loads(path.read_text()) for path in args.results])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["request_smoke_gate_open"], flush=True)


if __name__ == "__main__":
    main()
