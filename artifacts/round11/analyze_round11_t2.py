#!/usr/bin/env python3
"""Strictly compose the three frozen Round11 T2 attribution processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CAPTURE_SHA256 = "c837641468b9db93c422a3610af7a7ecc1f14d4bbb359841d69ac0dfd655dc6e"
BANK_SHA256 = "e9d6b27f352d501a9ed3737ec5d3ab140b9d1f5bda22d2e36ef9ef0bcc45b08b"
OUTPUT_SHA256 = "f5e189843493b9f42fcf724d9487ec5e1b9c630a6b0e780079cbc2bf6053f7c4"
BASE_NATIVE_SHA256 = "614ae60f70ff70d1e47d7fd32e857f887f874b6248eb332172c8a637cb471a6d"
ATTRIBUTION_NATIVE_SHA256 = "dff773ecabb3c7419afd8e3586e3c09934ebd9f23c7d9604eca5fa1218b39183"
ROLE_VERSIONS = {"baseline": 113, "counters": 114, "attribution": 115}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _median(result, name):
    item = result["parent_and_exclusive_us"][name]
    if item.get("count") != 3000:
        raise ValueError(f"{name} timing coverage mismatch")
    value = item.get("median_us")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} median is invalid")
    return float(value)


def _load_run(path: Path, role: str):
    result_path = path / "result.json"
    controller_path = path / "controller.json"
    cleanup_path = path / "post-cleanup-proc-mem.txt"
    result = json.loads(result_path.read_text())
    controller = json.loads(controller_path.read_text())
    expected_version = ROLE_VERSIONS[role]
    if (
        controller.get("status") != "PASS_ROUND11_T2_COMPONENT_CONTROLLER"
        or controller.get("role") != role
        or controller.get("version") != expected_version
        or controller.get("result_sha256") != sha256(result_path)
    ):
        raise ValueError(f"{role} controller/result hash or identity mismatch")
    exit_ack = controller.get("guard_exit_ack", {})
    if (
        not isinstance(exit_ack, dict)
        or not str(exit_ack.get("action", "")).startswith("OWNED_EXITED")
        or controller.get("owned_container_removed") is not True
        or controller.get("physical12_after") != "EMPTY"
        or controller.get("oom_killed") is not False
        or "No process in device" not in cleanup_path.read_text()
    ):
        raise ValueError(f"{role} cleanup contract mismatch")
    if (
        result.get("status") != "PASS_ROUND11_T2_DEEP_ATTRIBUTION"
        or result.get("scope") != "ROUND11_T2_REAL_LAYER0_C128_MXFP4_DEEP_ATTRIBUTION"
        or result.get("iterations") != 3000
        or result.get("warmups") != 5
        or result.get("tokens") != 3
        or result.get("numa_nodes") != [2, 3, 4, 7]
        or result.get("cpu_workers") != 128
        or result.get("threadpool_count") != 4
        or result.get("capture_sha256") != CAPTURE_SHA256
        or result.get("bank_sha256") != BANK_SHA256
        or result.get("output_sha256") != OUTPUT_SHA256
        or result.get("repeat_hash_exact") is not True
    ):
        raise ValueError(f"{role} result correctness/fixture contract mismatch")
    expected_native = ATTRIBUTION_NATIVE_SHA256 if role == "attribution" else BASE_NATIVE_SHA256
    expected_binary_role = (
        "ATTRIBUTION_ONLY_TASK_QUEUE_TIMESTAMPS"
        if role == "attribution"
        else "FROZEN_HFINAL_NATIVE"
    )
    if (
        result.get("native_sha256") != expected_native
        or result.get("binary_role") != expected_binary_role
    ):
        raise ValueError(f"{role} native identity mismatch")
    expected_tp_calls = 3 * (result["iterations"] + result["warmups"])
    expected_group_calls = result["iterations"] + result["warmups"]
    phase = result.get("native_phases", {})
    partitions = phase.get("tp_partitions", {})
    if (
        set(partitions) != {"0", "1", "2", "3"}
        or any(row.get("calls") != expected_tp_calls for row in partitions.values())
        or phase.get("merge", {}).get("calls") != expected_group_calls
    ):
        raise ValueError(f"{role} native phase coverage mismatch")
    for name in (
        "total_parent",
        "submit_parent",
        "pinned_gather_and_metadata",
        "enqueue_and_task_construction_exclusive",
        "sync_parent",
        "exposed_queue_and_compute_wait_exclusive",
        "result_publication",
    ):
        _median(result, name)
    return {
        "result": result,
        "controller": controller,
        "source_sha256": {
            "result": sha256(result_path),
            "controller": sha256(controller_path),
            "post_cleanup_proc_mem": sha256(cleanup_path),
        },
    }


def _thread_summary(result):
    rows = result.get("thread_accounting", {}).get("rows", [])
    target_ids = set(result.get("thread_accounting", {}).get("compute_thread_ids", []))
    active = [
        row
        for row in rows
        if float(row.get("sched_runtime_ns", 0)) > 0
        and (not target_ids or row.get("tid") in target_ids)
    ]
    runtimes = [float(row["sched_runtime_ns"]) for row in active]
    waits = [float(row.get("sched_wait_ns", 0)) for row in active]
    if not runtimes:
        return {"status": "UNKNOWN_NO_ACTIVE_THREAD_ROWS"}
    mean = sum(runtimes) / len(runtimes)
    variance = sum((value - mean) ** 2 for value in runtimes) / len(runtimes)
    return {
        "status": "MEASURED_PROC_THREAD_ACCOUNTING",
        "active_thread_rows": len(active),
        "expected_compute_threads": 128,
        "counter_scope": "NAMED_NUMA_COMPUTE_THREADS_ONLY",
        "total_sched_runtime_ns": sum(runtimes),
        "total_sched_wait_ns": sum(waits),
        "runtime_max_to_min_ratio": max(runtimes) / max(min(runtimes), 1.0),
        "runtime_coefficient_of_variation": math.sqrt(variance) / max(mean, 1.0),
        "context_switches": "SEE_PERF_COUNTERS_IF_AVAILABLE",
    }


def _counter_summary(result):
    counters = result.get("hardware_counters", {})
    status = counters.get("status", "UNKNOWN")
    report = {
        "status": status,
        "llc_misses": "UNKNOWN",
        "memory_bandwidth": "UNKNOWN",
        "remote_numa_accesses": "UNKNOWN",
        "source_boundaries": {
            "llc": counters.get("llc_misses", "UNKNOWN"),
            "memory_bandwidth": counters.get("memory_controller_bandwidth", "UNKNOWN"),
            "remote_numa": counters.get("remote_numa_accesses", "UNKNOWN"),
        },
    }
    summary = counters.get("summary")
    if not isinstance(summary, dict):
        report["open_errors"] = counters.get("open_errors", [])
        return report
    totals = summary.get("scaled_totals", {})
    groups = result["iterations"]
    for name in (
        "cycles",
        "instructions",
        "cache_references",
        "cache_misses",
        "context_switches",
        "cpu_migrations",
    ):
        value = totals.get(name)
        report[f"{name}_per_group"] = None if value is None else value / groups
    report["ipc"] = summary.get("ipc")
    report["generic_cache_miss_ratio"] = summary.get("generic_cache_miss_ratio")
    return report


def compose_t2(run_paths):
    if set(run_paths) != set(ROLE_VERSIONS):
        raise ValueError("T2 requires baseline, counters, and attribution run paths")
    runs = {role: _load_run(Path(run_paths[role]), role) for role in ROLE_VERSIONS}
    baseline = runs["baseline"]["result"]
    counters = runs["counters"]["result"]
    attribution = runs["attribution"]["result"]
    queue = attribution.get("queue_wait_separation")
    if not isinstance(queue, dict) or queue.get("calls") != 3005:
        raise ValueError("attribution-only queue separation coverage mismatch")

    total = _median(attribution, "total_parent")
    prepare = _median(attribution, "pinned_gather_and_metadata")
    enqueue = _median(attribution, "enqueue_and_task_construction_exclusive")
    exposed = _median(attribution, "exposed_queue_and_compute_wait_exclusive")
    publication = _median(attribution, "result_publication")
    queue_wait = float(queue["queue_admission_wait_us_per_group"])
    task_execute = float(queue["task_execute_us_per_group"])
    native = float(attribution["native_phases"]["native_compute_plus_merge_us_per_group"])
    task_residual = task_execute - native
    sync_residual = exposed - queue_wait - task_execute
    top_residual = total - prepare - enqueue - exposed - publication
    tolerance_us = max(5.0, total * 0.02)
    if min(task_residual, sync_residual, top_residual) < -tolerance_us:
        raise ValueError("nested attribution produced a materially negative residual")

    attribution_overhead = total / _median(baseline, "total_parent")
    counter_overhead = _median(counters, "total_parent") / _median(baseline, "total_parent")
    decomposition = {
        "total_parent": total,
        "pinned_gather_and_metadata": prepare,
        "enqueue_and_task_construction_exclusive": enqueue,
        "exposed_queue_and_compute_wait_exclusive": exposed,
        "result_publication": publication,
        "top_level_rounding_residual": top_residual,
        "queue_admission_wait": queue_wait,
        "task_execute_parent": task_execute,
        "sync_bridge_residual": sync_residual,
        "native_parallel_critical_compute_plus_merge": native,
        "task_execute_residual_after_native": task_residual,
        "native_tp_partition_imbalance_ratio": attribution["native_phases"][
            "tp_partition_imbalance_ratio"
        ],
        "note": (
            "TP partitions execute in parallel; only the critical partition plus merge is used. "
            "Nested parent spans are presented as a hierarchy and are not summed twice."
        ),
    }
    rankable = {
        "pinned_gather_and_metadata": prepare,
        "enqueue_and_task_construction": enqueue,
        "queue_admission_wait": queue_wait,
        "native_parallel_compute_and_merge": native,
        "task_execute_residual": max(task_residual, 0.0),
        "sync_bridge_residual": max(sync_residual, 0.0),
        "result_publication": publication,
    }
    ranking = [
        {"component": key, "median_us": value, "share_of_total": value / total}
        for key, value in sorted(rankable.items(), key=lambda item: item[1], reverse=True)
    ]
    instrumentation_quality = (
        "LOW_PERTURBATION"
        if attribution_overhead <= 1.05
        else "MATERIAL_PERTURBATION_USE_BASELINE_FOR_ABSOLUTE_TIME"
    )
    report = {
        "schema": 1,
        "status": "PASS_ROUND11_T2_DEEP_ATTRIBUTION",
        "fixture": {
            "capture_sha256": CAPTURE_SHA256,
            "bank_sha256": BANK_SHA256,
            "output_sha256": OUTPUT_SHA256,
            "tokens": 3,
            "iterations_per_process": 3000,
            "numa_nodes": [2, 3, 4, 7],
            "cpu_workers": 128,
            "threadpool_count": 4,
        },
        "run_sources": {
            role: {
                "version": ROLE_VERSIONS[role],
                "binary_role": runs[role]["result"]["binary_role"],
                "native_sha256": runs[role]["result"]["native_sha256"],
                "source_sha256": runs[role]["source_sha256"],
            }
            for role in ROLE_VERSIONS
        },
        "baseline_total_median_us": _median(baseline, "total_parent"),
        "counter_run_total_median_us": _median(counters, "total_parent"),
        "attribution_total_median_us": total,
        "instrumentation_overhead_ratio": attribution_overhead,
        "counter_overhead_ratio": counter_overhead,
        "instrumentation_quality": instrumentation_quality,
        "attribution_decomposition_us_per_group": decomposition,
        "exclusive_hotspot_ranking": ranking,
        "thread_accounting": _thread_summary(attribution),
        "hardware_counters": _counter_summary(counters),
        "known_unknowns": [
            "LLC-specific misses remain UNKNOWN because only generic cache events are mapped.",
            "Memory-controller bandwidth remains UNKNOWN without a validated uncore event mapping.",
            "Remote NUMA accesses remain UNKNOWN without a validated PMU/uncore event mapping.",
        ],
        "cleanup": {
            "all_three_guard_exit_ack": True,
            "all_three_containers_removed": True,
            "physical12_after_each": "EMPTY",
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--counters", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compose_t2(
        {
            "baseline": args.baseline,
            "counters": args.counters,
            "attribution": args.attribution,
        }
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
