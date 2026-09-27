"""Two-round Round10 CPU/NUMA/HBM/identity admission sampler and validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import time


NODES = (2, 3, 4, 7)
TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
MIN_IDLE = 0.80
MIN_NODE_FREE_KIB = 48 * 1024**2
MIN_SYSTEM_AVAILABLE_KIB = 704 * 1024**2


def cpulist(value):
    result = []
    for item in value.strip().split(","):
        if not item:
            continue
        first, separator, last = item.partition("-")
        result.extend(range(int(first), int(last if separator else first) + 1))
    return result


def cpu_stats():
    return {
        int(values[0][3:]): list(map(int, values[1:]))
        for line in pathlib.Path("/proc/stat").read_text().splitlines()
        if (values := line.split())[0].startswith("cpu") and values[0] != "cpu"
    }


def meminfo(path="/proc/meminfo"):
    return {
        values[0].rstrip(":"): int(values[1])
        for line in pathlib.Path(path).read_text().splitlines()
        if len(values := line.split()) >= 2
    }


def parse_numa_meminfo(raw):
    result = {}
    for line in raw.splitlines():
        values = line.split()
        if len(values) >= 4 and values[0] == "Node" and values[1].isdigit():
            result[values[2].rstrip(":")] = int(values[3])
    return result


def command(args, timeout=30):
    return subprocess.check_output(args, text=True, timeout=timeout)


def physical12_processes(raw):
    return {
        int(match.group(1)): int(match.group(2))
        for line in raw.splitlines()
        if (
            match := re.match(
                r"\|\s*6\s+0\s*\|\s*(\d+)\s*\|[^|]*\|\s*(\d+)\s*\|",
                line,
            )
        )
    }


def npu_memory(raw):
    row = next(line for line in raw.splitlines() if "0000:85:00.0" in line)
    used, total = map(int, re.findall(r"(\d+)\s*/\s*(\d+)", row)[-1])
    return used, total


def training_identity(training_id):
    if re.fullmatch(r"[0-9a-f]{64}", training_id) is None:
        raise ValueError("training identity must be a full Docker ID")
    documents = json.loads(command(["docker", "inspect", training_id]))
    if len(documents) != 1 or documents[0]["Id"] != training_id or not documents[0]["State"]["Running"]:
        raise RuntimeError("training container identity is absent or stopped")
    pid = int(documents[0]["State"]["Pid"])
    cgroup = pathlib.Path(f"/proc/{pid}/cgroup").read_text()
    if f"/docker/{training_id}" not in cgroup:
        raise RuntimeError("training PID cgroup does not match the full container ID")
    return {"id": training_id, "pid": pid, "cgroup": cgroup, "running": True}


def topology_and_containers():
    allowed = os.sched_getaffinity(0)
    cores = {}
    nodes = {}
    for node_path in sorted(pathlib.Path("/sys/devices/system/node").glob("node[0-9]*")):
        node = int(node_path.name[4:])
        nodes[node] = {"distance": (node_path / "distance").read_text().strip()}
        for cpu in cpulist((node_path / "cpulist").read_text()):
            if cpu not in allowed:
                continue
            topo = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            key = (
                int((topo / "physical_package_id").read_text()),
                int((topo / "core_id").read_text()),
            )
            row = cores.setdefault(
                key,
                {"node": node, "socket": key[0], "core": key[1], "siblings": []},
            )
            if row["node"] != node:
                raise RuntimeError("one physical core spans NUMA nodes")
            row["siblings"].append(cpu)
    containers = []
    reserved = set()
    for container_id in command(["docker", "ps", "-q"]).split():
        document = json.loads(command(["docker", "inspect", container_id]))[0]
        cpuset = document["HostConfig"]["CpusetCpus"]
        containers.append(
            {
                "id": document["Id"],
                "name": document["Name"],
                "cpuset": cpuset,
                "mems": document["HostConfig"]["CpusetMems"],
            }
        )
        reserved.update(cpulist(cpuset))
    core_list = sorted(cores.values(), key=lambda row: (row["node"], min(row["siblings"])))
    for row in core_list:
        row["excluded_explicit_container_affinity"] = bool(set(row["siblings"]) & reserved)
    return allowed, core_list, nodes, containers


def sample(output, training_id=TRAINING_ID):
    output = pathlib.Path(output)
    if output.exists():
        raise FileExistsError(output)
    if "110.123.0.3" not in command(["hostname", "-I"]).split():
        raise RuntimeError("wrong host")
    allowed, cores, nodes, containers = topology_and_containers()
    start_training = training_identity(training_id)
    document = {
        "schema": 1,
        "started_at": time.time(),
        "host": command(["hostname", "-I"]).strip(),
        "cores": cores,
        "nodes": nodes,
        "containers": containers,
        "pci_numa": pathlib.Path("/sys/bus/pci/devices/0000:85:00.0/numa_node").read_text().strip(),
        "training_start": start_training,
        "policy": "6x10s; per-core idle=min(all SMT siblings); exact v90 physical cores; every window strictly >80%; node MemFree>=48GiB; system MemAvailable>=704GiB; physical12 unknown PID fail-closed",
        "windows": [],
    }
    previous = cpu_stats()
    for index in range(6):
        time.sleep(10)
        current = cpu_stats()
        idle = {
            str(cpu): (current[cpu][3] - previous[cpu][3])
            / max(1, sum(current[cpu][:8]) - sum(previous[cpu][:8]))
            for cpu in allowed
            if cpu in current and cpu in previous
        }
        previous = current
        npu_raw = command(["npu-smi", "info"])
        processes = physical12_processes(npu_raw)
        process_cgroups = {}
        for pid in processes:
            try:
                process_cgroups[str(pid)] = pathlib.Path(f"/proc/{pid}/cgroup").read_text()
            except OSError as error:
                raise RuntimeError("physical12 process changed during admission") from error
        used, total = npu_memory(npu_raw)
        row = {
            "index": index,
            "time": time.time(),
            "idle": idle,
            "system_meminfo": meminfo(),
            "numa_meminfo": {
                str(node): parse_numa_meminfo(
                    pathlib.Path(f"/sys/devices/system/node/node{node}/meminfo").read_text()
                )
                for node in nodes
            },
            "physical12_processes": {str(pid): memory for pid, memory in processes.items()},
            "physical12_process_cgroups": process_cgroups,
            "npu_used_mib": used,
            "npu_total_mib": total,
        }
        document["windows"].append(row)
        output.write_text(json.dumps(document, indent=2) + "\n")
        selected = [item for item in cores if item["node"] in NODES and not item["excluded_explicit_container_affinity"]]
        print("WINDOW", index + 1, round(conservative_mean(selected, idle), 6), flush=True)
    document["ended_at"] = time.time()
    document["training_end"] = training_identity(training_id)
    output.write_text(json.dumps(document, indent=2) + "\n")
    return document


def conservative_mean(cores, idle):
    if not cores:
        raise RuntimeError("no candidate cores")
    values = [min(idle[str(cpu)] for cpu in row["siblings"]) for row in cores]
    return sum(values) / len(values)


def _cgroup_matches(cgroup, training_id):
    return any(
        line.split(":", 2)[-1] == f"/docker/{training_id}"
        for line in cgroup.splitlines()
    )


def validate_documents(documents):
    if len(documents) != 2:
        raise RuntimeError("exactly two admission rounds required")
    means = []
    min_free = {str(node): float("inf") for node in NODES}
    selected_cpus = None
    training_end = None
    for round_index, document in enumerate(documents):
        if len(document.get("windows", [])) != 6:
            raise RuntimeError("each admission round requires six 10-second windows")
        if document.get("ended_at", 0) - document.get("started_at", 0) < 59:
            raise RuntimeError("admission round shorter than 60 seconds")
        cores = [
            row
            for row in document.get("cores", [])
            if row.get("node") in NODES and not row.get("excluded_explicit_container_affinity")
        ]
        if len(cores) != 160:
            raise RuntimeError("exact v90 160-core candidate set unavailable after affinity exclusions")
        cpus = [min(row["siblings"]) for row in cores]
        if selected_cpus is None:
            selected_cpus = cpus
        elif cpus != selected_cpus:
            raise RuntimeError("candidate CPU identity changed between rounds")
        for key in ("training_start", "training_end"):
            training = document.get(key, {})
            if training.get("id") != TRAINING_ID or training.get("running") is not True:
                raise RuntimeError("training container identity changed")
            if not _cgroup_matches(training.get("cgroup", ""), TRAINING_ID):
                raise RuntimeError("training cgroup identity changed")
        training_end = document["training_end"]
        for window in document["windows"]:
            mean = conservative_mean(cores, window["idle"])
            means.append(mean)
            if mean <= MIN_IDLE:
                raise RuntimeError("candidate CPU idle gate failed")
            available = window.get("system_meminfo", {}).get("MemAvailable", 0)
            if available < MIN_SYSTEM_AVAILABLE_KIB:
                raise RuntimeError("system MemAvailable gate failed")
            for node in NODES:
                free = window.get("numa_meminfo", {}).get(str(node), {}).get("MemFree", 0)
                min_free[str(node)] = min(min_free[str(node)], free / 1024**2)
                if free < MIN_NODE_FREE_KIB:
                    raise RuntimeError(f"NUMA{node} MemFree gate failed")
            if window.get("npu_total_mib") != 65536:
                raise RuntimeError("physical12 total HBM identity changed")
            for pid, memory in window.get("physical12_processes", {}).items():
                cgroup = window.get("physical12_process_cgroups", {}).get(str(pid), "")
                if not _cgroup_matches(cgroup, TRAINING_ID):
                    raise RuntimeError(f"unknown physical12 process {pid}")
                if memory > 15360:
                    raise RuntimeError("training process exceeds shared HBM allowance")
    return {
        "status": "PASS",
        "ended_at": max(document["ended_at"] for document in documents),
        "verified_at": time.time(),
        "windows": len(means),
        "selected_physical_cores": len(selected_cpus),
        "min_mean_idle": min(means),
        "max_mean_idle": max(means),
        "window_means": means,
        "min_numa_free_gib": min_free,
        "training_id": TRAINING_ID,
        "training_pid": training_end["pid"],
        "training_cgroup": training_end["cgroup"],
        "cpus": selected_cpus,
        "policy": "min SMT sibling idle; each 10s window strictly >80%; 2x60s; exact v90 CPU identity; 48GiB/node; 704GiB system available; unknown physical12 PID fail-closed",
    }


def synthetic_document_for_test():
    cores = []
    idle = {}
    cpu = 1
    for node in NODES:
        for _ in range(40):
            cores.append(
                {
                    "node": node,
                    "siblings": [cpu],
                    "excluded_explicit_container_affinity": False,
                }
            )
            idle[str(cpu)] = 0.99
            cpu += 1
    training = {
        "id": TRAINING_ID,
        "pid": 1,
        "cgroup": f"1:name=systemd:/docker/{TRAINING_ID}\n",
        "running": True,
    }
    window = {
        "idle": idle,
        "system_meminfo": {"MemAvailable": MIN_SYSTEM_AVAILABLE_KIB + 1},
        "numa_meminfo": {
            str(node): {"MemFree": MIN_NODE_FREE_KIB + 1} for node in NODES
        },
        "physical12_processes": {},
        "physical12_process_cgroups": {},
        "npu_total_mib": 65536,
        "npu_used_mib": 0,
    }
    return {
        "started_at": 1,
        "ended_at": 61,
        "cores": cores,
        "windows": [json.loads(json.dumps(window)) for _ in range(6)],
        "training_start": training,
        "training_end": training,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sampling = sub.add_parser("sample")
    sampling.add_argument("--output", type=pathlib.Path, required=True)
    sampling.add_argument("--training-id", default=TRAINING_ID)
    validation = sub.add_parser("validate")
    validation.add_argument("--discovery", type=pathlib.Path, required=True)
    validation.add_argument("--validation", type=pathlib.Path, required=True)
    validation.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.command == "sample":
        sample(args.output, args.training_id)
        print("ROUND10_ADMISSION_SAMPLE_COMPLETE", args.output, flush=True)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    documents = [json.loads(args.discovery.read_text()), json.loads(args.validation.read_text())]
    result = validate_documents(documents)
    result["raw_sha256"] = {
        args.discovery.name: hashlib.sha256(args.discovery.read_bytes()).hexdigest(),
        args.validation.name: hashlib.sha256(args.validation.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
