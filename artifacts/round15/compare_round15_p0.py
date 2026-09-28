#!/usr/bin/env python3
"""Strictly compare the two fresh Round15 P0 profile blocks."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iso(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def metrics(document):
    return {
        row["workload"]: {
            "ttft_seconds": row["ttft_seconds"],
            "tpot_seconds": row["tpot_seconds"],
            "e2e_seconds": row["e2e_seconds"],
            "completion_tokens": row["completion_tokens"],
        }
        for row in document["results"]
    }


def profile(document):
    result = {}
    for workload in ("W1", "W2"):
        batches = [
            batch for batch in document["batches"]
            if batch["request_id"].endswith(workload.lower())
        ]
        by_mode = {}
        for mode in ("extend", "decode"):
            selected = [batch for batch in batches if batch["mode"] == mode]
            components = defaultdict(list)
            for batch in selected:
                if batch.get("dropped_spans") != 0 or batch.get("observer_errors"):
                    raise RuntimeError(f"{workload}/{mode}: incomplete observer batch")
                for span in batch["spans"]:
                    if span.get("outcome") != "return" or span.get("observer_errors"):
                        raise RuntimeError(f"{workload}/{mode}: incomplete observer span")
                    components[span["component"]].append(span)
            rows = {}
            for component, spans in sorted(components.items()):
                host = [span["duration_ns"] / 1e6 for span in spans]
                device = [span["device_ms"] for span in spans if span["device_ms"] is not None]
                rows[component] = {
                    "observed_calls": len(spans),
                    "host_median_ms": statistics.median(host),
                    "host_total_ms": sum(host),
                    "device_median_ms": statistics.median(device) if device else None,
                    "device_total_ms": sum(device) if device else None,
                }
            by_mode[mode] = {"observed_batches": len(selected), "components": rows}
        result[workload] = by_mode
    return result


def validate_identity(a, b):
    fixed = (
        "stage", "parent", "profile_threshold", "default_threshold",
        "fixture_sha256", "generation_budget", "graph",
    )
    for key in fixed:
        if a.get(key) != b.get(key):
            raise RuntimeError(f"P0 identity mismatch: {key}")
    if (a.get("version"), a.get("block")) != (335, "A"):
        raise RuntimeError("P0 A identity mismatch")
    if (b.get("version"), b.get("block")) != (338, "B"):
        raise RuntimeError("P0 B identity mismatch")
    for label, document in (("A", a), ("B", b)):
        if (
            document.get("generation_attempts") != 4
            or document.get("abort_controls") != 0
            or document.get("automatic_retries") != 0
            or document.get("graph") is not False
        ):
            raise RuntimeError(f"P0 {label} budget/mode mismatch")
        if [row["workload"] for row in document["results"]] != [
            "WARMUP", "W1", "W2", "DRAIN"
        ]:
            raise RuntimeError(f"P0 {label} workload mismatch")
        if any(row.get("finite_logprobs") is not True for row in document["results"]):
            raise RuntimeError(f"P0 {label} non-finite output")
        for row in document["results"]:
            qsa = row.get("qsa", {})
            if (
                qsa.get("prefill_strategy") != "VECTORIZED"
                or qsa.get("decode_strategy") != "DIRECT_BF16_ROW"
            ):
                raise RuntimeError(f"P0 {label} QSA identity mismatch")
    if [row["output_ids"] for row in a["results"]] != [
        row["output_ids"] for row in b["results"]
    ]:
        raise RuntimeError("P0 cross-block token output mismatch")
    if [row["path"] for row in a["results"]] != [row["path"] for row in b["results"]]:
        raise RuntimeError("P0 cross-block Hybrid path mismatch")
    if a["five_bucket_memory"]["buckets"] != b["five_bucket_memory"]["buckets"]:
        raise RuntimeError("P0 five-bucket storage identity mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--b-supervisor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a = json.loads(args.a.read_text())
    b = json.loads(args.b.read_text())
    supervisor = json.loads(args.b_supervisor.read_text())
    validate_identity(a, b)
    if (
        b.get("status") != "PASS_ROUND15_P0_PROFILE_PROCESS"
        or b.get("profiler") != "BOUNDED_EXISTING_EVENT_OBSERVER"
        or supervisor.get("status") != "PASS_ROUND15_P0_PROFILE_PROCESS"
        or supervisor.get("owned_container_removed") is not True
        or supervisor.get("owned_exit_state", {}).get("OOMKilled") is not False
        or supervisor.get("physical12_after") != "REGISTERED_TRAINING_ONLY"
    ):
        raise RuntimeError("P0 B process/cleanup contract mismatch")
    state = supervisor["owned_exit_state"]
    seconds = (iso(state["FinishedAt"]) - iso(state["StartedAt"])).total_seconds()
    result = {
        "schema": 1,
        "status": "PASS_ROUND15_P0_TWO_FRESH_BLOCKS",
        "identity": {
            "parent": b["parent"],
            "fixture_sha256": b["fixture_sha256"],
            "threshold": b["default_threshold"],
            "graph": False,
            "qsa_prefill": "VECTORIZED",
            "qsa_decode": "DIRECT_BF16_ROW",
            "cross_block_output_ids_exact": True,
            "cross_block_hybrid_path_exact": True,
            "five_bucket_storage_exact": True,
        },
        "blocks": {
            "A_v335_salvaged": {
                "metrics": metrics(a),
                "profile": profile(a),
                "boundary": "raw measurements salvaged; original control FAIL preserved",
            },
            "B_v338": {
                "metrics": metrics(b),
                "profile": profile(b),
                "runtime_seconds": seconds,
                "npu_device_hours": seconds / 3600.0,
                "cleanup": b["cleanup"],
            },
        },
        "hotspot_disposition": {
            "decode": (
                "QSA_ATTENTION is the largest observed per-sampled-call device span; "
                "QSA_INDEXER is second among the QSA subpaths and is the first new A1 candidate."
            ),
            "prefill": (
                "QSA_ATTENTION and QSA_INDEXER dominate the observed sampled device spans; "
                "GDN RMSNorm-only work is a smaller fraction of the complete promoted GDN parent."
            ),
            "no_parent_child_sum": True,
            "selection": "SCREEN_A1_LIGHTNING_INDEXER_THEN_A2_RMSNORM_IF_A1_INAPPLICABLE",
            "request_gate_if_a1_advances": "DECODE_W2",
        },
        "evidence": {
            "a_raw_sha256": sha256(args.a),
            "b_final_sha256": sha256(args.b),
            "b_supervisor_sha256": sha256(args.b_supervisor),
        },
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
