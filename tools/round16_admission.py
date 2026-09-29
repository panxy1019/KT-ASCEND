#!/usr/bin/env python3
"""Capture and validate an explicit Round16 NPU/NUMA admission target.

This program is read-only with respect to Docker, processes, NPU state and
memory.  ``sample`` captures one policy-sized round; callers must run it twice and
then use ``validate``.  Merely passing ``precheck`` never substitutes for the
two required rounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

# When executed as ``python tools/round16_admission.py``, Python places the
# ``tools`` directory rather than its parent on ``sys.path``.  Add exactly the
# repository root containing this script so the sibling ``round16`` package is
# resolved without requiring an ambient PYTHONPATH.  Normal module imports do
# not take this branch.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from round16.common.parameterized_admission import (
    AdmissionPolicy,
    AdmissionResult,
    validate_admission_documents,
)
from round16.common.resource_target import (
    NpuDeviceSnapshot,
    NpuTarget,
    parse_npu_smi,
    select_safe_target,
)


def command(args: Sequence[str], *, timeout: int = 60) -> str:
    return subprocess.check_output(args, text=True, timeout=timeout)


def policy_from_payload(payload: Mapping[str, Any]) -> AdmissionPolicy:
    if payload.get("schema") != 1:
        raise ValueError("unsupported policy schema")
    target_payload = payload.get("target")
    if not isinstance(target_payload, Mapping):
        raise ValueError("policy has no explicit target")
    nodes = target_payload.get("numa_nodes")
    if not isinstance(nodes, list):
        raise ValueError("policy has no explicit NUMA profile")
    target = NpuTarget(
        physical_id=target_payload.get("physical_id"),
        card_id=target_payload.get("card_id"),
        chip_id=target_payload.get("chip_id"),
        pci_bdf=target_payload.get("pci_bdf"),
        numa_nodes=tuple(nodes),
    )
    return AdmissionPolicy(
        target=target,
        training_container_id=payload.get("training_container_id"),
        expected_host_ip=payload.get("expected_host_ip"),
        expected_physical_cores=payload.get("expected_physical_cores", 160),
        rounds=payload.get("rounds", 2),
        windows_per_round=payload.get("windows_per_round", 6),
        minimum_round_seconds=payload.get("minimum_round_seconds", 60.0),
        minimum_idle_exclusive=payload.get("minimum_idle_exclusive", 0.80),
        minimum_node_free_kib=payload.get("minimum_node_free_kib", 48 * 1024**2),
        minimum_system_available_kib=payload.get(
            "minimum_system_available_kib", 704 * 1024**2
        ),
        expected_hbm_total_mib=payload.get("expected_hbm_total_mib", 65536),
        training_process_limit_mib=payload.get(
            "training_process_limit_mib", 15360
        ),
    )


def read_policy(path: Path) -> AdmissionPolicy:
    return policy_from_payload(json.loads(path.read_text()))


def _visible_device_values(document: Mapping[str, Any]) -> list[str]:
    config = document.get("Config")
    if not isinstance(config, Mapping):
        raise RuntimeError("container inspect has no Config")
    environment = config.get("Env")
    if environment is None:
        return []
    if not isinstance(environment, list):
        raise RuntimeError("container environment is malformed")
    values: list[str] = []
    device_environment_keys = {
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
    }
    for item in environment:
        if not isinstance(item, str):
            raise RuntimeError("container environment entry is malformed")
        key, separator, value = item.partition("=")
        if separator and key in device_environment_keys:
            values.append(value)
    distinct_values = tuple(dict.fromkeys(values))
    if len(distinct_values) > 1:
        raise RuntimeError("container has conflicting Ascend device visibility")
    return list(distinct_values)


def claims_from_container_documents(
    documents: Sequence[Mapping[str, Any]],
) -> dict[int, str]:
    claims: dict[int, str] = {}
    for document in documents:
        owner = document.get("Name")
        if not isinstance(owner, str) or not owner:
            raise RuntimeError("container inspect has no owner name")
        values = _visible_device_values(document)
        if not values:
            continue
        raw = values[0]
        fields = raw.split(",")
        if not fields or any(not field.isdigit() for field in fields):
            raise RuntimeError(f"malformed ASCEND device claim for {owner}")
        for field in fields:
            physical_id = int(field)
            if not 0 <= physical_id <= 15:
                raise RuntimeError(f"malformed ASCEND device claim for {owner}")
            if physical_id in claims:
                raise RuntimeError(
                    f"multiple containers claim physical device {physical_id}"
                )
            claims[physical_id] = owner
    return claims


def running_container_documents() -> list[Mapping[str, Any]]:
    identifiers = command(["docker", "ps", "-q"]).split()
    if not identifiers:
        return []
    documents = json.loads(command(["docker", "inspect", *identifiers]))
    if not isinstance(documents, list):
        raise RuntimeError("docker inspect did not return a list")
    return documents


def _cpulist(value: str) -> list[int]:
    result: list[int] = []
    for item in value.strip().split(","):
        if not item:
            continue
        first, separator, last = item.partition("-")
        result.extend(range(int(first), int(last if separator else first) + 1))
    return result


def _cpu_stats() -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] == "cpu" or not fields[0].startswith("cpu"):
            continue
        result[int(fields[0][3:])] = [int(value) for value in fields[1:]]
    return result


def _meminfo(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2:
            result[fields[0].rstrip(":")] = int(fields[1])
    return result


def _numa_meminfo(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "Node" and fields[1].isdigit():
            result[fields[2].rstrip(":")] = int(fields[3])
    return result


def _training_identity(container_id: str) -> dict[str, object]:
    if len(container_id) != 64:
        raise RuntimeError("training container identity is not full length")
    documents = json.loads(command(["docker", "inspect", container_id]))
    if (
        not isinstance(documents, list)
        or len(documents) != 1
        or documents[0].get("Id") != container_id
        or documents[0].get("State", {}).get("Running") is not True
    ):
        raise RuntimeError("training container identity is absent or stopped")
    pid = int(documents[0]["State"]["Pid"])
    cgroup = Path(f"/proc/{pid}/cgroup").read_text()
    return {"id": container_id, "pid": pid, "cgroup": cgroup, "running": True}


def _topology_and_containers() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    allowed = os.sched_getaffinity(0)
    container_documents = running_container_documents()
    reserved: set[int] = set()
    containers: list[dict[str, object]] = []
    for document in container_documents:
        host_config = document.get("HostConfig", {})
        cpuset = host_config.get("CpusetCpus", "")
        if not isinstance(cpuset, str):
            raise RuntimeError("container CPU affinity is malformed")
        if cpuset:
            reserved.update(_cpulist(cpuset))
        containers.append(
            {
                "id": document.get("Id"),
                "name": document.get("Name"),
                "cpuset": cpuset,
                "mems": host_config.get("CpusetMems", ""),
            }
        )

    cores: dict[tuple[int, int], dict[str, object]] = {}
    for node_path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        node = int(node_path.name[4:])
        for cpu in _cpulist((node_path / "cpulist").read_text()):
            if cpu not in allowed:
                continue
            topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            key = (
                int((topology / "physical_package_id").read_text()),
                int((topology / "core_id").read_text()),
            )
            row = cores.setdefault(
                key,
                {
                    "node": node,
                    "socket": key[0],
                    "core": key[1],
                    "siblings": [],
                },
            )
            if row["node"] != node:
                raise RuntimeError("one physical CPU core spans NUMA nodes")
            row["siblings"].append(cpu)

    rows = sorted(cores.values(), key=lambda row: (row["node"], min(row["siblings"])))
    for row in rows:
        row["siblings"] = sorted(row["siblings"])
        row["excluded_explicit_container_affinity"] = bool(
            set(row["siblings"]) & reserved
        )
    return rows, containers


def _target_payload(target: NpuTarget) -> dict[str, object]:
    return {
        "physical_id": target.physical_id,
        "card_id": target.card_id,
        "chip_id": target.chip_id,
        "pci_bdf": target.pci_bdf,
        "numa_nodes": list(target.numa_nodes),
    }


def _capture_target(
    policy: AdmissionPolicy,
) -> tuple[dict[str, object], Mapping[int, NpuDeviceSnapshot]]:
    npu_raw = command(["npu-smi", "info"])
    devices = parse_npu_smi(npu_raw)
    device = devices.get(policy.target.physical_id)
    if device is None:
        raise RuntimeError("target disappeared from npu-smi")
    containers = running_container_documents()
    claims = claims_from_container_documents(containers)
    processes: list[dict[str, object]] = []
    for process in device.processes:
        try:
            cgroup = Path(f"/proc/{process.pid}/cgroup").read_text()
        except OSError as error:
            raise RuntimeError("target process changed during admission") from error
        processes.append(
            {
                "pid": process.pid,
                "name": process.name,
                "memory_mib": process.memory_mib,
                "cgroup": cgroup,
            }
        )
    payload = {
        "npu_smi_sha256": hashlib.sha256(npu_raw.encode()).hexdigest(),
        "target_device": {
            "physical_id": device.physical_id,
            "card_id": device.card_id,
            "chip_id": device.chip_id,
            "pci_bdf": device.pci_bdf,
            "hbm_used_mib": device.hbm_used_mib,
            "hbm_total_mib": device.hbm_total_mib,
            "processes": processes,
        },
        "device_claims": {str(key): value for key, value in sorted(claims.items())},
    }
    return payload, devices


def _target_window(policy: AdmissionPolicy) -> dict[str, object]:
    payload, _devices = _capture_target(policy)
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sample(output: Path, policy: AdmissionPolicy) -> Mapping[str, Any]:
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError(output)
    host_ips = command(["hostname", "-I"]).split()
    if policy.expected_host_ip not in host_ips:
        raise RuntimeError("wrong host")
    cores, containers = _topology_and_containers()
    document: dict[str, Any] = {
        "schema": 1,
        "started_at": time.time(),
        "host_ips": host_ips,
        "target": _target_payload(policy.target),
        "cores": cores,
        "containers_at_start": containers,
        "training_start": _training_identity(policy.training_container_id),
        "windows": [],
        "policy": {
            "rounds": policy.rounds,
            "windows_per_round": policy.windows_per_round,
            "window_seconds": 10,
            "idle": "mean(min(all SMT siblings)) strictly greater than 0.80",
            "minimum_node_free_kib": policy.minimum_node_free_kib,
            "minimum_system_available_kib": policy.minimum_system_available_kib,
            "external_target_claim": "fail-closed",
        },
    }
    previous = _cpu_stats()
    for index in range(policy.windows_per_round):
        time.sleep(10)
        current = _cpu_stats()
        idle = {
            str(cpu): (current[cpu][3] - previous[cpu][3])
            / max(1, sum(current[cpu][:8]) - sum(previous[cpu][:8]))
            for cpu in os.sched_getaffinity(0)
            if cpu in current and cpu in previous
        }
        previous = current
        target_evidence = _target_window(policy)
        window = {
            "index": index,
            "time": time.time(),
            "idle": idle,
            "system_meminfo": _meminfo(Path("/proc/meminfo")),
            "numa_meminfo": {
                str(node): _numa_meminfo(
                    Path(f"/sys/devices/system/node/node{node}/meminfo")
                )
                for node in policy.target.numa_nodes
            },
            **target_evidence,
        }
        document["windows"].append(window)
        _write_json_atomic(output, document)
        print(
            f"ROUND16_ADMISSION_WINDOW {index + 1}/{policy.windows_per_round}",
            flush=True,
        )
    document["ended_at"] = time.time()
    document["training_end"] = _training_identity(policy.training_container_id)
    _write_json_atomic(output, document)
    return document


def precheck(policy: AdmissionPolicy) -> Mapping[str, Any]:
    target_evidence, devices = _capture_target(policy)
    device_payload = target_evidence["target_device"]

    def belongs_to_training(cgroup: object) -> bool:
        if not isinstance(cgroup, str):
            return False
        for line in cgroup.splitlines():
            path = line.split(":", 2)[-1]
            if path == f"/docker/{policy.training_container_id}":
                return True
            if path.endswith(f"/docker-{policy.training_container_id}.scope"):
                return True
        return False

    allowed_training_pids = {
        process["pid"]
        for process in device_payload["processes"]
        if belongs_to_training(process["cgroup"])
    }
    decision = select_safe_target(
        devices,
        claims={int(key): value for key, value in target_evidence["device_claims"].items()},
        allowed_training_pids=allowed_training_pids,
        candidates=(policy.target,),
        expected_total_mib=policy.expected_hbm_total_mib,
        training_process_limit_mib=policy.training_process_limit_mib,
    )
    return {
        "schema": 1,
        "checked_at": time.time(),
        "target": _target_payload(policy.target),
        "ready_for_two_round_sampling": decision.ready,
        "rejections": {
            str(key): list(value) for key, value in decision.rejections.items()
        },
        "evidence": target_evidence,
        "note": (
            "precheck never substitutes for two complete policy-sized rounds"
        ),
    }


def _result_payload(result: AdmissionResult) -> dict[str, object]:
    return {
        "schema": 1,
        "status": result.status,
        "target": _target_payload(result.target),
        "windows": result.windows,
        "selected_physical_cores": result.selected_physical_cores,
        "selected_parent_cpus": list(result.selected_parent_cpus),
        "min_mean_idle": result.min_mean_idle,
        "max_mean_idle": result.max_mean_idle,
        "window_means": list(result.window_means),
        "min_numa_free_gib": {
            str(key): value for key, value in result.min_numa_free_gib.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    precheck_parser = subparsers.add_parser("precheck")
    precheck_parser.add_argument("--policy", type=Path, required=True)
    precheck_parser.add_argument("--output", type=Path, required=True)
    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--policy", type=Path, required=True)
    sample_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--policy", type=Path, required=True)
    validate_parser.add_argument("--round1", type=Path, required=True)
    validate_parser.add_argument("--round2", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = read_policy(args.policy)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.operation == "precheck":
        _write_json_atomic(args.output, precheck(policy))
        return
    if args.operation == "sample":
        sample(args.output, policy)
        return

    documents = [json.loads(args.round1.read_text()), json.loads(args.round2.read_text())]
    result = _result_payload(validate_admission_documents(documents, policy))
    result["raw_sha256"] = {
        args.round1.name: hashlib.sha256(args.round1.read_bytes()).hexdigest(),
        args.round2.name: hashlib.sha256(args.round2.read_bytes()).hexdigest(),
    }
    _write_json_atomic(args.output, result)


if __name__ == "__main__":
    main()
