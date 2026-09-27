#!/usr/bin/env python3
"""Strict Round14 A K/N/N/K system-deployment comparator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


EXPECTED = (
    (235, "KT", "A"),
    (251, "NATIVE_TP8", "A"),
    (252, "NATIVE_TP8", "B"),
    (238, "KT", "B"),
)
WORKLOADS = ("W0", "W1", "W2", "W3", "W4")


def _read_jsonl(path: Path) -> list[dict]:
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"unterminated monitor JSONL: {path}")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _resource_summary(rows: list[dict]) -> dict:
    samples = [row for row in rows if "host_mem_available_kib" in row]
    if not samples:
        raise RuntimeError("Round14 A monitor has no resource samples")
    native = any(isinstance(row.get("devices"), dict) for row in samples)
    if native:
        simultaneous = [
            sum(int(device["used_mib"]) for device in row["devices"].values())
            for row in samples if isinstance(row.get("devices"), dict)
        ]
        per_device = {
            str(device): max(
                int(row["devices"][str(device)]["used_mib"])
                for row in samples if str(device) in row.get("devices", {})
            )
            for device in range(8)
        }
        owned_hbm = [
            sum(int(device.get("owned_process_mib", 0)) for device in row["devices"].values())
            for row in samples if isinstance(row.get("devices"), dict)
        ]
    else:
        simultaneous = [int(row["hbm_used_mib"]) for row in samples if "hbm_used_mib" in row]
        per_device = {"12": max(simultaneous)}
        owned_hbm = [int(row.get("owned_process_mib", 0)) for row in samples]
    if not simultaneous:
        raise RuntimeError("Round14 A monitor lacks HBM samples")
    cgroup = []
    for row in samples:
        values = row.get("owned_cgroup", {})
        for name in ("memory.peak", "memory.max_usage_in_bytes", "memory.current", "memory.usage_in_bytes"):
            value = values.get(name)
            if value is not None:
                cgroup.append(int(value))
    numa_nodes = {"2", "3", "4", "7"}
    min_numa = {}
    for node in numa_nodes:
        values = [
            int(row["numa_meminfo_kib"][node]["MemFree"])
            for row in samples
            if node in row.get("numa_meminfo_kib", {})
        ]
        min_numa[node] = min(values) if values else None
    return {
        "physical_devices": list(range(8)) if native else [12],
        "simultaneous_whole_card_hbm_peak_mib": max(simultaneous),
        "per_device_whole_card_hbm_peak_mib": per_device,
        "simultaneous_owned_process_hbm_peak_mib": max(owned_hbm) if owned_hbm else None,
        "host_cgroup_peak_bytes": max(cgroup) if cgroup else None,
        "minimum_host_mem_available_kib": min(int(row["host_mem_available_kib"]) for row in samples),
        "minimum_numa_memfree_kib": min_numa,
        "sample_count": len(samples),
    }


def _cleanup_passes(final: dict) -> bool:
    cleanup = final.get("cleanup", {})
    if cleanup.get("container_removed") is not True:
        return False
    backend = final.get("backend")
    if backend == "KT":
        return (
            cleanup.get("guard_exit_ack") is True
            and cleanup.get("physical12_after") == "EMPTY"
        )
    if backend == "NATIVE_TP8":
        ack = cleanup.get("guard_exit_ack")
        if not isinstance(ack, dict) or ack.get("action") not in {
            "OWNED_EXITED", "OWNED_EXITED_DURING_SAMPLE",
        }:
            return False
        if ack.get("action") == "OWNED_EXITED":
            state = ack.get("state", {})
            if state.get("Running") is not False or state.get("Pid") != 0:
                return False
        stopped = final.get("resource", {}).get("physical0_7_processes_after_stop")
        return (
            isinstance(stopped, dict)
            and set(stopped) == {str(device) for device in range(8)}
            and all(value == {} for value in stopped.values())
        )
    return False


def summarize_process(final_path: Path, monitor_path: Path) -> dict:
    final = json.loads(Path(final_path).read_text())
    if (
        final.get("status") != "PASS_ROUND14_A_MODEL_PROCESS"
        or final.get("generation_attempts") != 24
        or final.get("abort_controls") != 0
        or final.get("automatic_retries") != 0
        or final.get("profiler") != "OFF"
        or not _cleanup_passes(final)
    ):
        raise RuntimeError("Round14 A process/final/cleanup contract failed")
    results = final.get("results", [])
    quality = [row for row in results if row.get("phase") == "quality"]
    if len(quality) != 4 or any(row.get("quality_pass") is not True for row in quality):
        raise RuntimeError("Round14 A quality contract failed")
    workloads = {}
    for workload in WORKLOADS:
        warmup = [row for row in results if row.get("workload") == workload and row.get("phase") == "warmup"]
        measured = [row for row in results if row.get("workload") == workload and row.get("phase") == "measurement"]
        if len(warmup) != 1 or len(measured) != 3:
            raise RuntimeError(f"Round14 A {workload} repeat count mismatch")
        if any(row.get("finite_logprobs") is not True or row.get("finish_reason") != "length" for row in measured):
            raise RuntimeError(f"Round14 A {workload} output quality mismatch")
        window = final.get("measurement_windows", {}).get(workload)
        if not isinstance(window, dict) or window.get("successful_requests") != 3:
            raise RuntimeError(f"Round14 A {workload} full window missing")
        completion = sum(int(row["completion_tokens"]) for row in measured)
        if completion != window.get("actual_output_tokens"):
            raise RuntimeError(f"Round14 A {workload} token accounting mismatch")
        tpots = [float(row["tpot_seconds"]) for row in measured if row.get("tpot_seconds") is not None]
        workloads[workload] = {
            "e2e_median_seconds": statistics.median(float(row["e2e_seconds"]) for row in measured),
            "ttft_median_seconds": statistics.median(float(row["ttft_seconds"]) for row in measured),
            "tpot_median_seconds": statistics.median(tpots) if tpots else None,
            "actual_output_tokens": completion,
            "serial_output_tokens_per_second": float(window["serial_output_tokens_per_second"]),
            "requests_per_second": float(window["requests_per_second"]),
            "measurement_window_seconds": float(window["seconds"]),
            "errors": 0,
        }
    return {
        "version": final["version"],
        "backend": final["backend"],
        "block": final["block"],
        "quality_requests_passed": len(quality),
        "workloads": workloads,
        "resources": _resource_summary(_read_jsonl(monitor_path)),
    }


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def compare(inputs) -> dict:
    if len(inputs) != 4:
        raise ValueError("Round14 A comparator requires four process pairs")
    summaries = [summarize_process(final, monitor) for final, monitor in inputs]
    observed = [(row["version"], row["backend"], row["block"]) for row in summaries]
    if tuple(observed) != EXPECTED:
        raise RuntimeError(f"Round14 A process order/identity mismatch: {observed}")
    by_version = {str(row["version"]): row for row in summaries}
    blocks = {}
    for block, k_version, n_version in (("A", 235, 251), ("B", 238, 252)):
        kt = by_version[str(k_version)]
        native = by_version[str(n_version)]
        block_rows = {}
        for workload in WORKLOADS:
            k = kt["workloads"][workload]
            n = native["workloads"][workload]
            block_rows[workload] = {
                "kt_version": k_version,
                "native_version": n_version,
                "native_over_kt_e2e": _ratio(n["e2e_median_seconds"], k["e2e_median_seconds"]),
                "native_over_kt_ttft": _ratio(n["ttft_median_seconds"], k["ttft_median_seconds"]),
                "native_over_kt_tpot": _ratio(n["tpot_median_seconds"], k["tpot_median_seconds"]),
                "native_over_kt_serial_throughput": _ratio(
                    n["serial_output_tokens_per_second"], k["serial_output_tokens_per_second"]
                ),
                "native_over_kt_requests_per_second": _ratio(
                    n["requests_per_second"], k["requests_per_second"]
                ),
                "kt": k,
                "native": n,
            }
        blocks[block] = block_rows
    return {
        "schema": 1,
        "status": "PASS_ROUND14_A_SYSTEM_COMPARISON",
        "process_order": [row["version"] for row in summaries],
        "claim_scope": "SYSTEM_DEPLOYMENT_COMPARISON_NOT_PURE_KT_CAUSAL",
        "differences": ["NPU_COUNT_1_VS_8", "MXFP4_PLUS_BF16_VS_BF16", "PROMOTED_QSA_VS_OLD_QSA", "KT_HETEROGENEOUS_VS_NATIVE_PURE_NPU"],
        "processes": by_version,
        "blocks": blocks,
        "w2_native_serial_output_tokens_per_second": [
            by_version["251"]["workloads"]["W2"]["serial_output_tokens_per_second"],
            by_version["252"]["workloads"]["W2"]["serial_output_tokens_per_second"],
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("k_a", "n_a", "n_b", "k_b"):
        parser.add_argument(f"--{name.replace('_', '-')}-final", type=Path, required=True)
        parser.add_argument(f"--{name.replace('_', '-')}-monitor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = [
        (args.k_a_final, args.k_a_monitor),
        (args.n_a_final, args.n_a_monitor),
        (args.n_b_final, args.n_b_monitor),
        (args.k_b_final, args.k_b_monitor),
    ]
    report = compare(inputs)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["status"], *report["w2_native_serial_output_tokens_per_second"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
