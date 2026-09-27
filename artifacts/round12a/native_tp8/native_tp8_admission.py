#!/usr/bin/env python3
"""Two-round CPU/NUMA/physical0-7 admission for native TP8."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


NODES = (2, 3, 4, 7)
PHYSICAL_DEVICES = tuple(range(8))
TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
MIN_IDLE = 0.80
MIN_NODE_FREE_KIB = 48 * 1024**2
MIN_SYSTEM_AVAILABLE_KIB = 704 * 1024**2
MAX_BASELINE_HBM_MIB = 8192


def command(args, timeout=30):
    return subprocess.check_output(args, text=True, timeout=timeout)


def training_identity(training_id=TRAINING_ID):
    if re.fullmatch(r"[0-9a-f]{64}", training_id) is None:
        raise ValueError("training identity must be a full Docker ID")
    documents = json.loads(command(["docker", "inspect", training_id]))
    if len(documents) != 1 or documents[0]["Id"] != training_id or not documents[0]["State"]["Running"]:
        raise RuntimeError("registered training container identity is absent or stopped")
    pid = int(documents[0]["State"]["Pid"])
    cgroup = Path(f"/proc/{pid}/cgroup").read_text()
    if not any(line.split(":", 2)[-1] == f"/docker/{training_id}" for line in cgroup.splitlines()):
        raise RuntimeError("registered training PID cgroup does not match its full ID")
    return {"id": training_id, "pid": pid, "cgroup": cgroup, "running": True}


def cpulist(value):
    result = []
    for item in value.strip().split(","):
        if not item:
            continue
        first, sep, last = item.partition("-")
        result.extend(range(int(first), int(last if sep else first) + 1))
    return result


def cpu_stats():
    return {
        int(values[0][3:]): list(map(int, values[1:]))
        for line in Path("/proc/stat").read_text().splitlines()
        if (values := line.split())[0].startswith("cpu") and values[0] != "cpu"
    }


def meminfo(path="/proc/meminfo"):
    return {
        values[0].rstrip(":"): int(values[1])
        for line in Path(path).read_text().splitlines()
        if len(values := line.split()) >= 2
    }


def parse_numa_meminfo(raw):
    result = {}
    for line in raw.splitlines():
        values = line.split()
        if len(values) >= 4 and values[0] == "Node" and values[1].isdigit():
            result[values[2].rstrip(":")] = int(values[3])
    return result


def parse_first8_memory(raw: str) -> dict[int, dict]:
    result = {}
    for line in raw.splitlines():
        match = re.match(
            r"\|\s*(\d+)\s+(\d+)\s*\|\s*(0000:[0-9A-Fa-f]{2}:00\.0)\s*\|.*?(\d+)\s*/\s*(65536)\s*\|?\s*$",
            line,
        )
        if match:
            physical = int(match.group(2))
            if physical in PHYSICAL_DEVICES:
                result[physical] = {
                    "card": physical // 2,
                    "chip": physical % 2,
                    "pci": match.group(3),
                    "used_mib": int(match.group(4)),
                    "total_mib": int(match.group(5)),
                }
    if set(result) != set(PHYSICAL_DEVICES):
        raise RuntimeError("physical0-7 HBM rows are incomplete")
    return result


def parse_first8_processes(raw: str) -> dict[int, dict[int, int]]:
    result = {device: {} for device in PHYSICAL_DEVICES}
    for line in raw.splitlines():
        match = re.match(
            r"\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|[^|]*\|\s*(\d+)\s*\|",
            line,
        )
        if match:
            card, chip, pid, memory = map(int, match.groups())
            physical = card * 2 + chip
            if physical in result:
                result[physical][pid] = memory
    return result


def keyed(mapping, key):
    """Read a numeric-keyed row before or after JSON serialization."""
    if key in mapping:
        return mapping[key]
    return mapping[str(key)]


def _project_containers():
    rows = []
    for container_id in command(["docker", "ps", "-q"]).split():
        document = json.loads(command(["docker", "inspect", container_id]))[0]
        name = document["Name"].lstrip("/")
        if name.startswith("kt-round12") or name.startswith("kt-round11") or name.startswith("kt-round10"):
            rows.append({"id": document["Id"], "name": name})
    return rows


def _topology():
    allowed = os.sched_getaffinity(0)
    cores, reserved = {}, set()
    containers = []
    for container_id in command(["docker", "ps", "-q"]).split():
        doc = json.loads(command(["docker", "inspect", container_id]))[0]
        cpus = doc["HostConfig"]["CpusetCpus"]
        containers.append({"id": doc["Id"], "name": doc["Name"], "cpuset": cpus})
        reserved.update(cpulist(cpus))
    for node_path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        node = int(node_path.name[4:])
        for cpu in cpulist((node_path / "cpulist").read_text()):
            if cpu not in allowed:
                continue
            topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            key = (int((topo / "physical_package_id").read_text()), int((topo / "core_id").read_text()))
            row = cores.setdefault(key, {"node": node, "siblings": []})
            row["siblings"].append(cpu)
    values = sorted(cores.values(), key=lambda r: (r["node"], min(r["siblings"])))
    for row in values:
        row["excluded_explicit_container_affinity"] = bool(set(row["siblings"]) & reserved)
    return allowed, values, containers


def conservative_mean(cores, idle):
    values = [min(idle[str(cpu)] for cpu in row["siblings"]) for row in cores]
    if not values:
        raise RuntimeError("native TP8 has no candidate physical cores")
    return sum(values) / len(values)


def sample(output: Path):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if "110.123.0.3" not in command(["hostname", "-I"]).split():
        raise RuntimeError("wrong host")
    allowed, cores, containers = _topology()
    document = {
        "schema": 1, "started_at": time.time(), "host": command(["hostname", "-I"]).strip(),
        "cores": cores, "containers": containers,
        "training_start": training_identity(),
        "pci_numa": {
            str(device): Path(f"/sys/bus/pci/devices/{pci}/numa_node").read_text().strip()
            for device, pci in enumerate((
                "0000:9d:00.0", "0000:9f:00.0", "0000:99:00.0", "0000:9b:00.0",
                "0000:95:00.0", "0000:97:00.0", "0000:91:00.0", "0000:93:00.0",
            ))
        },
        "policy": "2x60s; 6x10s each; physical0-7 empty; no other KT project container; CPU/NUMA Round11 contract",
        "windows": [],
    }
    previous = cpu_stats()
    for index in range(6):
        time.sleep(10)
        current = cpu_stats()
        idle = {
            str(cpu): (current[cpu][3] - previous[cpu][3]) / max(1, sum(current[cpu][:8]) - sum(previous[cpu][:8]))
            for cpu in allowed if cpu in current and cpu in previous
        }
        previous = current
        raw = command(["npu-smi", "info"])
        row = {
            "index": index, "time": time.time(), "idle": idle,
            "system_meminfo": meminfo(),
            "numa_meminfo": {
                str(node): parse_numa_meminfo(Path(f"/sys/devices/system/node/node{node}/meminfo").read_text())
                for node in NODES
            },
            "first8_memory": parse_first8_memory(raw),
            "first8_processes": parse_first8_processes(raw),
            "project_containers": _project_containers(),
        }
        document["windows"].append(row)
        output.write_text(json.dumps(document, indent=2) + "\n")
        selected = [r for r in cores if r["node"] in NODES and not r["excluded_explicit_container_affinity"]]
        print("WINDOW", index + 1, round(conservative_mean(selected, idle), 6), flush=True)
    document["ended_at"] = time.time()
    document["training_end"] = training_identity()
    output.write_text(json.dumps(document, indent=2) + "\n")
    return document


def validate_documents(documents):
    if len(documents) != 2:
        raise RuntimeError("exactly two native TP8 admission rounds required")
    means, selected_cpus = [], None
    min_free = {str(node): float("inf") for node in NODES}
    hbm_baselines = {str(device): [] for device in PHYSICAL_DEVICES}
    for document in documents:
        if len(document.get("windows", [])) != 6 or document.get("ended_at", 0) - document.get("started_at", 0) < 59:
            raise RuntimeError("each native TP8 admission round must cover 60 seconds")
        cores = [r for r in document["cores"] if r["node"] in NODES and not r["excluded_explicit_container_affinity"]]
        if len(cores) != 160:
            raise RuntimeError("exact Round11 160-core set is unavailable")
        cpus = [min(r["siblings"]) for r in cores]
        if selected_cpus is None:
            selected_cpus = cpus
        elif selected_cpus != cpus:
            raise RuntimeError("CPU identity changed between native TP8 rounds")
        for key in ("training_start", "training_end"):
            training = document.get(key, {})
            if training.get("id") != TRAINING_ID or training.get("running") is not True:
                raise RuntimeError("registered training container identity changed")
            if not any(
                line.split(":", 2)[-1] == f"/docker/{TRAINING_ID}"
                for line in training.get("cgroup", "").splitlines()
            ):
                raise RuntimeError("registered training cgroup changed")
        for window in document["windows"]:
            if window.get("project_containers"):
                raise RuntimeError("native TP8 cannot overlap another KT performance container")
            mean = conservative_mean(cores, window["idle"])
            means.append(mean)
            if mean <= MIN_IDLE:
                raise RuntimeError("native TP8 CPU idle gate failed")
            if window["system_meminfo"].get("MemAvailable", 0) < MIN_SYSTEM_AVAILABLE_KIB:
                raise RuntimeError("native TP8 system MemAvailable gate failed")
            for node in NODES:
                free = window["numa_meminfo"][str(node)].get("MemFree", 0)
                min_free[str(node)] = min(min_free[str(node)], free / 1024**2)
                if free < MIN_NODE_FREE_KIB:
                    raise RuntimeError(f"native TP8 NUMA{node} MemFree gate failed")
            if any(keyed(window["first8_processes"], device) for device in PHYSICAL_DEVICES):
                raise RuntimeError("physical0-7 has an existing device process")
            memory = window["first8_memory"]
            for device in PHYSICAL_DEVICES:
                row = keyed(memory, device)
                if row["total_mib"] != 65536 or row["used_mib"] > MAX_BASELINE_HBM_MIB:
                    raise RuntimeError(f"physical{device} HBM admission failed")
                hbm_baselines[str(device)].append(row["used_mib"])
    return {
        "status": "PASS_NATIVE_TP8_ADMISSION", "ended_at": max(d["ended_at"] for d in documents),
        "verified_at": time.time(), "windows": len(means), "selected_physical_cores": 160,
        "min_mean_idle": min(means), "max_mean_idle": max(means), "window_means": means,
        "min_numa_free_gib": min_free, "cpus": selected_cpus,
        "physical_devices": list(PHYSICAL_DEVICES), "hbm_baselines_mib": hbm_baselines,
        "overlap_policy": "NO_OTHER_KT_PERFORMANCE_CONTAINER",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command_name", required=True)
    sampling = sub.add_parser("sample")
    sampling.add_argument("--output", type=Path, required=True)
    validation = sub.add_parser("validate")
    validation.add_argument("--discovery", type=Path, required=True)
    validation.add_argument("--validation", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command_name == "sample":
        sample(args.output)
        return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    docs = [json.loads(args.discovery.read_text()), json.loads(args.validation.read_text())]
    result = validate_documents(docs)
    result["raw_sha256"] = {
        args.discovery.name: hashlib.sha256(args.discovery.read_bytes()).hexdigest(),
        args.validation.name: hashlib.sha256(args.validation.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
