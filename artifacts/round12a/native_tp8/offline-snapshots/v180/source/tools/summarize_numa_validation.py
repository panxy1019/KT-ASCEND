"""Reduce a monitored NUMA trial to a source-bound, reviewable receipt."""

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .numa_policy import conservative_available_kib, runtime_pressure
except ImportError:  # Direct script execution from tools/.
    from numa_policy import conservative_available_kib, runtime_pressure


GIB = 1024 ** 3
KIB_PER_GIB = 1024 ** 2


def _owned_anon_bytes(raw):
    for line in raw.splitlines():
        if line.startswith("anon="):
            return {
                int(key[1:]): int(value) * 4096
                for key, value in (field.split("=", 1) for field in line.split()[1:])
            }
    raise ValueError("owned cgroup NUMA statistics omit anonymous pages")


def _event_counts(raw):
    return {key: int(value) for key, value in
            (line.split() for line in raw.splitlines() if line.strip())}


def summarize(records, nodes, *, driver_status):
    if not records or not nodes or any(type(node) is not int or node < 0 for node in nodes):
        raise ValueError("monitor records and numeric NUMA nodes are required")
    numa_records = [row for row in records if isinstance(row.get("numa_meminfo_kib"), dict)]
    cgroup_records = [row for row in numa_records
                      if isinstance(row.get("owned_cgroup"), dict)
                      and "memory.numa_stat" in row["owned_cgroup"]]
    failures = set()
    missing_nodes = [node for node in nodes if not all(str(node) in row["numa_meminfo_kib"]
                                                       for row in numa_records)]
    if not numa_records or missing_nodes:
        failures.add("incomplete NUMA sampling")
    if not cgroup_records:
        failures.add("incomplete owned cgroup sampling")

    node_summary = {}
    if numa_records and not missing_nodes:
        anon_records = [_owned_anon_bytes(row["owned_cgroup"]["memory.numa_stat"])
                        for row in cgroup_records]
        for node in nodes:
            samples = [row["numa_meminfo_kib"][str(node)] for row in numa_records]
            node_summary[str(node)] = {
                "minimum_memfree_gib": min(sample["MemFree"] for sample in samples) / KIB_PER_GIB,
                "minimum_conservative_available_gib": min(
                    conservative_available_kib(sample) for sample in samples
                ) / KIB_PER_GIB,
                "peak_owned_anon_gib": (
                    max((sample.get(node, 0) for sample in anon_records), default=0) / GIB
                ),
            }

    failcnt = 0
    event_peaks = {name: 0 for name in ("high", "oom", "oom_kill")}
    for row in cgroup_records:
        cgroup = row["owned_cgroup"]
        failcnt = max(failcnt, int(cgroup.get("memory.failcnt", "0")))
        events = _event_counts(cgroup.get("memory.events", ""))
        for name in event_peaks:
            event_peaks[name] = max(event_peaks[name], events.get(name, 0))
        pressure = runtime_pressure(
            {int(node): sample for node, sample in row["numa_meminfo_kib"].items()
             if int(node) in nodes},
            cgroup,
        )
        if pressure:
            failures.add(pressure)

    if failures:
        status = "FAIL_MONITORED_MXFP4_NUMA_VALIDATION"
    elif driver_status != "PASS_TWO_DRIVERS":
        status = "INCOMPLETE_MONITORED_MXFP4_NUMA_VALIDATION"
    else:
        status = "PASS_MONITORED_MXFP4_NUMA_VALIDATION"

    return {
        "schema": 1,
        "status": status,
        "scope": "MXFP4_C128_Q32_LAYERWISE_PREFILL_END_DYNAMIC_SHORT_SMOKE",
        "driver_status": driver_status,
        "samples": len(records),
        "cgroup_samples": len(cgroup_records),
        "nodes": node_summary,
        "peak_owned_cgroup_usage_gib": max(
            (int(row["owned_cgroup"]["memory.usage_in_bytes"]) for row in cgroup_records),
            default=0,
        ) / GIB,
        "peak_owned_cgroup_recorded_gib": max(
            (int(row["owned_cgroup"]["memory.max_usage_in_bytes"]) for row in cgroup_records),
            default=0,
        ) / GIB,
        "peak_owned_hbm_gib": max(
            (int(row.get("hbm_used_mib") or 0)
             - int(row.get("estimated_nonowned_board_mib") or 0) for row in records),
            default=0,
        ) / 1024,
        "owned_cgroup_failcnt_peak": failcnt,
        "owned_cgroup_event_peaks": event_peaks,
        "failures": sorted(failures),
        "qualification": "ONE_MONITORED_FRAMEWORK_SMOKE_NOT_UNIVERSAL_CAPACITY_PROOF",
    }


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--nodes", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    driver = json.loads(args.driver.read_text())
    result = summarize(_read_jsonl(args.monitor), args.nodes,
                       driver_status=driver.get("status"))
    result["monitor_sha256"] = hashlib.sha256(args.monitor.read_bytes()).hexdigest()
    result["driver_sha256"] = hashlib.sha256(args.driver.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
