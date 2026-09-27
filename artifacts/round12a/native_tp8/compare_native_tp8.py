#!/usr/bin/env python3
"""Compare four independent native TP8 replicas in frozen A/B/B/A order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


ORDER = [(180, "A"), (181, "B"), (182, "B"), (183, "A")]
EXPECTED = [3897, 248046]
OLD_QSA_SHA256 = "76e9f34066f0b42f42545427967c0b789d5c32031dc5c085e230efaf95e3c54c"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, version: int, block: str) -> dict:
    document = json.loads(path.read_text())
    measurements = [row for row in document.get("results", []) if row.get("phase") == "measurement"]
    if (
        document.get("status") != "PASS_ROUND12A_NATIVE_TP8_MODEL_PROCESS"
        or document.get("version") != version or document.get("block") != block
        or document.get("mode") != "NATIVE_TP8" or len(measurements) != 3
        or document.get("runtime", {}).get("qsa_sha256") != OLD_QSA_SHA256
        or document.get("runtime", {}).get("tp_size") != 8
        or any(row.get("output_ids") != EXPECTED or row.get("quality_pass") is not True for row in document.get("results", []))
        or sorted(map(int, document.get("rank_evidence", {}).get("ranks", []))) != list(range(8))
    ):
        raise RuntimeError(f"invalid native TP8 final result: {path}")
    e2e = [float(row["e2e_seconds"]) for row in measurements]
    ttft = [float(row["ttft_seconds"]) for row in measurements]
    tpot = [float(row["tpot_seconds"]) for row in measurements]
    if any(not math.isfinite(value) or value <= 0 for value in e2e + ttft + tpot):
        raise RuntimeError("native TP8 measurement timing is invalid")
    resource = document["resource"]
    peaks = {str(key): int(value) for key, value in resource["per_device_peak_hbm_mib"].items()}
    if set(peaks) != {str(i) for i in range(8)} or any(value > 56832 for value in peaks.values()):
        raise RuntimeError("native TP8 HBM receipt is incomplete")
    return {
        "version": version, "block": block, "path": str(path), "sha256": sha256(path),
        "median_e2e_seconds": statistics.median(e2e),
        "median_ttft_seconds": statistics.median(ttft),
        "median_tpot_seconds": statistics.median(tpot),
        "samples_e2e_seconds": e2e, "per_device_peak_hbm_mib": peaks,
        "host_cgroup_peak_bytes": resource.get("host_cgroup_peak_bytes"),
        "min_host_mem_available_kib": int(resource["min_host_mem_available_kib"]),
        "min_numa_memfree_kib": {
            str(key): int(value) for key, value in resource["min_numa_memfree_kib"].items()
        },
    }


def compare(paths: list[Path]) -> dict:
    if len(paths) != 4:
        raise ValueError("native TP8 comparator requires four independent processes")
    rows = [_load(Path(path), version, block) for path, (version, block) in zip(paths, ORDER)]
    medians = [row["median_e2e_seconds"] for row in rows]
    pair_ratios = {
        "v181_over_v180": medians[1] / medians[0],
        "v182_over_v183": medians[2] / medians[3],
    }
    timing_stable = max(medians) / min(medians) <= 1.05 and all(0.95 <= value <= 1.05 for value in pair_ratios.values())
    cross_device_spread = {
        str(device): max(row["per_device_peak_hbm_mib"][str(device)] for row in rows)
        - min(row["per_device_peak_hbm_mib"][str(device)] for row in rows)
        for device in range(8)
    }
    within_run_imbalance = [
        max(row["per_device_peak_hbm_mib"].values()) - min(row["per_device_peak_hbm_mib"].values())
        for row in rows
    ]
    host_peaks = [row["host_cgroup_peak_bytes"] for row in rows]
    host_known = all(isinstance(value, int) and value > 0 for value in host_peaks)
    host_ratio = max(host_peaks) / min(host_peaks) if host_known else None
    min_host_available = [row["min_host_mem_available_kib"] for row in rows]
    min_numa_free = {
        node: [row["min_numa_memfree_kib"][node] for row in rows]
        for node in ("2", "3", "4", "7")
    }
    memory_safety = (
        min(min_host_available) >= 64 * 1024**2
        and all(min(values) >= 8 * 1024**2 for values in min_numa_free.values())
    )
    resource_stable = (
        max(cross_device_spread.values()) <= 1024
        and max(within_run_imbalance) <= 2048
        and host_known and host_ratio <= 1.05 and memory_safety
    )
    qualified = timing_stable and resource_stable
    return {
        "schema": 1,
        "status": (
            "PASS_ROUND12A_NATIVE_TP8_BASELINE_ABBA"
            if qualified else "ROUND12A_NATIVE_TP8_COMPLETED_WITH_INSTABILITY"
        ),
        "interpretation": "A/B labels are identical-code independent replicas, not implementation variants",
        "comparison_scope": "NATIVE_TP8_OLD_QSA_REPEATABILITY_NOT_KT_VS_NATIVE_SPEEDUP",
        "order": [f"v{version} {block}" for version, block in ORDER],
        "quality": {"all_exact": True, "expected_output_ids": EXPECTED, "old_qsa_sha256": OLD_QSA_SHA256},
        "performance": {
            "timing_stable_within_5_percent": timing_stable,
            "process_median_e2e_seconds": {f"v{row['version']}": row["median_e2e_seconds"] for row in rows},
            "pair_ratios": pair_ratios, "max_over_min": max(medians) / min(medians),
        },
        "resource": {
            "stable": resource_stable, "cross_run_per_device_spread_mib": cross_device_spread,
            "within_run_device_imbalance_mib": within_run_imbalance,
            "host_cgroup_peak_bytes": host_peaks, "host_peak_max_over_min": host_ratio,
            "min_host_mem_available_kib": min_host_available,
            "min_numa_memfree_kib": min_numa_free,
            "runtime_memory_safety_floors_pass": memory_safety,
        },
        "runs": rows,
        "qualified": qualified,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v180", type=Path, required=True)
    parser.add_argument("--v181", type=Path, required=True)
    parser.add_argument("--v182", type=Path, required=True)
    parser.add_argument("--v183", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare([args.v180, args.v181, args.v182, args.v183])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
