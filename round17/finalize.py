"""Deterministic Round17 result aggregation from archived JSON evidence."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summarize_monitor(lines: Iterable[str]) -> dict[str, float | int]:
    rows = [json.loads(line) for line in lines if line.strip()]
    samples = [row for row in rows if "hbm_used_mib" in row]
    if not samples:
        raise ValueError("monitor contains no resource samples")
    host_peaks = []
    for row in samples:
        raw = row.get("owned_cgroup", {}).get("memory.max_usage_in_bytes")
        if raw is not None:
            host_peaks.append(int(raw))
    baseline = min(int(row["hbm_used_mib"]) for row in samples)
    whole = max(int(row["hbm_used_mib"]) for row in samples)
    return {
        "whole_card_hbm_peak_mib": whole,
        "observed_board_baseline_mib": baseline,
        "estimated_owned_hbm_peak_mib": whole - baseline,
        "host_cgroup_peak_gib": max(host_peaks) / 1024**3,
        "samples": len(samples),
    }


def baseline_block(result: dict[str, Any], monitor_lines: Iterable[str]) -> dict[str, Any]:
    if result.get("status") != "PASS_ROUND16_S0_REQUEST_PERFORMANCE_BLOCK":
        raise ValueError("fresh S0 result is not PASS")
    if result.get("profiler") != "OFF" or result.get("generation_attempts") != 10:
        raise ValueError("fresh S0 request contract changed")
    performance = [row for row in result["results"] if row["kind"] == "performance"]
    if len(performance) != 9:
        raise ValueError("fresh S0 performance schedule is incomplete")
    if any(row["completion_tokens"] != row["decode_tokens"] for row in performance):
        raise ValueError("fixed-length performance output is incomplete")
    return {
        "block": result["block"],
        "profiler": result["profiler"],
        "generation_attempts": result["generation_attempts"],
        "summaries": result["summaries"],
        "resources": summarize_monitor(monitor_lines),
    }


def aggregate_baseline(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if len(blocks) != 2:
        raise ValueError("Round17 baseline requires two independent blocks")
    workloads: dict[str, Any] = {}
    for workload in ("W1", "W2", "W4"):
        rows = [block["summaries"][workload] for block in blocks]
        workloads[workload] = {
            "block_values": rows,
            "median_of_block_medians": {
                "ttft_seconds": median(row["median_ttft_seconds"] for row in rows),
                "tpot_seconds": median(row["median_tpot_seconds"] for row in rows),
                "e2e_seconds": median(row["median_e2e_seconds"] for row in rows),
                "serial_output_tokens_per_second": median(
                    row["serial_output_tokens_per_second"] for row in rows
                ),
                "requests_per_second": median(row["requests_per_second"] for row in rows),
            },
        }
    return {"schema": 1, "status": "PASS_ROUND17_FRESH_S0", "blocks": blocks, "workloads": workloads}


def qsa_ceiling(profile: dict[str, Any]) -> dict[str, float]:
    host = profile["host_ms_per_token"]
    device = profile["device_ms_per_token"]
    parent_host = host["QSA_ATTENTION"]["median"]
    parent_device = device["QSA_ATTENTION"]["median"]
    sparse_host = host["QSA_SPARSE_CORE"]["median"]
    sparse_device = device["QSA_SPARSE_CORE"]["median"]
    return {
        "qsa_parent_host_ms_per_token": parent_host,
        "qsa_parent_device_ms_per_token": parent_device,
        "indexer_host_ms_per_token": host["QSA_INDEXER"]["median"],
        "sparse_core_host_ms_per_token": sparse_host,
        "sparse_core_device_ms_per_token": sparse_device,
        "perfect_sparse_core_removal_host_percent": 100 * sparse_host / parent_host,
        "perfect_sparse_core_removal_device_percent": 100 * sparse_device / parent_device,
        "qsa_uninstrumented_host_ms_per_token": profile[
            "qsa_uninstrumented_host_ms_per_token"
        ]["median"],
    }
