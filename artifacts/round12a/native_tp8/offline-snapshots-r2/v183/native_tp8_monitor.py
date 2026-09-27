#!/usr/bin/env python3
"""Eight-device ownership/HBM/Host guard for one native TP8 container."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_tp8_admission import parse_first8_memory, parse_first8_processes
from numa_policy import parse_node, runtime_pressure


def owned_container_name(name: str) -> bool:
    return re.fullmatch(r"kt-round12a-native-tp8-[ab]-v(?:180|181|182|183)", name) is not None


def command(args, timeout=20):
    return subprocess.check_output(args, text=True, timeout=timeout)


def process_in_container(pid: int, container_id: str) -> bool:
    raw = Path(f"/proc/{pid}/cgroup").read_text()
    return any(line.split(":", 2)[-1] == f"/docker/{container_id}" for line in raw.splitlines())


def owned_processes(name: str, waiting: bool):
    if waiting:
        return set()
    try:
        raw = command(["docker", "top", name, "-eo", "pid"])
    except subprocess.CalledProcessError:
        return None
    return {int(line.strip()) for line in raw.splitlines()[1:] if line.strip().isdigit()}


def _append(path: Path, item: dict):
    with path.open("a") as stream:
        stream.write(json.dumps(item) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=21600)
    parser.add_argument("--numa-nodes", nargs="*", type=int, default=[2, 3, 4, 7])
    args = parser.parse_args()
    if not owned_container_name(args.container):
        raise ValueError("native TP8 guard container identity rejected")
    if args.output.exists() or args.ready_file.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    baselines = {device: [] for device in range(8)}
    container_id = None

    def inspect():
        return json.loads(command(["docker", "inspect", args.container]))[0]

    def stop_owned(reason):
        _append(args.output, {"time": time.time(), "action": "STOP_OWNED", "reason": reason})
        subprocess.run(["docker", "stop", "-t", "10", args.container], timeout=30, check=False)
        terminal = inspect()
        if terminal["State"]["Running"]:
            raise RuntimeError("native TP8 guard stop did not terminate container")
        _append(args.output, {"time": time.time(), "action": "OWNED_EXITED", "container_id": terminal["Id"]})

    while True:
        try:
            state = inspect()
            container_id = state["Id"] if container_id is None else container_id
            if state["Id"] != container_id:
                raise RuntimeError("native TP8 container identity changed")
            waiting = state["State"]["Status"] == "created"
            if waiting and time.monotonic() - started > 90:
                raise RuntimeError("native TP8 container was not started after guard arm")
            if not state["State"]["Running"] and not waiting:
                _append(args.output, {"time": time.time(), "action": "OWNED_EXITED", "state": state["State"]})
                return 0
            raw = command(["npu-smi", "info"])
            memory = parse_first8_memory(raw)
            processes = parse_first8_processes(raw)
            owned = owned_processes(args.container, waiting)
            if owned is None:
                _append(args.output, {"time": time.time(), "action": "OWNED_EXITED_DURING_SAMPLE"})
                return 0
            device_rows = {}
            for device in range(8):
                outsiders = set(processes[device]) - owned
                if outsiders:
                    raise RuntimeError(f"physical{device} has unowned processes {sorted(outsiders)}")
                owned_mib = sum(processes[device].values())
                if waiting:
                    baselines[device].append(memory[device]["used_mib"])
                    baselines[device] = baselines[device][-3:]
                device_rows[str(device)] = {
                    **memory[device], "processes": processes[device], "owned_process_mib": owned_mib,
                    "baseline_samples_mib": list(baselines[device]),
                }
                if memory[device]["used_mib"] > 56832:
                    raise RuntimeError(f"physical{device} HBM reserve fell below 8704 MiB")
            mem = {
                values[0].rstrip(":"): int(values[1])
                for line in Path("/proc/meminfo").read_text().splitlines()
                if len(values := line.split()) >= 2
            }
            item = {
                "time": time.time(), "container_status": state["State"]["Status"],
                "devices": device_rows, "host_mem_available_kib": mem["MemAvailable"],
                "host_swap_free_kib": mem["SwapFree"],
                "numa_meminfo_kib": {
                    str(node): parse_node(
                        Path(f"/sys/devices/system/node/node{node}/meminfo").read_text()
                    )
                    for node in args.numa_nodes
                },
            }
            pressure = Path("/proc/pressure/memory")
            item["memory_pressure"] = (
                pressure.read_text() if pressure.exists() else "UNAVAILABLE_KERNEL_PSI"
            )
            if not waiting:
                pid = int(state["State"]["Pid"])
                cg = Path(f"/proc/{pid}/cgroup").read_text()
                if f"/docker/{container_id}" not in cg:
                    raise RuntimeError("native TP8 root PID cgroup identity changed")
                unified = next((line.split(":", 2)[2] for line in cg.splitlines() if line.startswith("0::")), None)
                legacy = next((line.split(":", 2)[2] for line in cg.splitlines() if "memory" in line.split(":", 2)[1].split(",")), None)
                relative = legacy or unified
                root = Path("/sys/fs/cgroup/memory" if legacy else "/sys/fs/cgroup") / relative.lstrip("/")
                item["owned_cgroup"] = {
                    name: (root / name).read_text().strip()
                    for name in ("memory.current", "memory.peak", "memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.events")
                    if (root / name).exists()
                }
            _append(args.output, item)
            if item["host_mem_available_kib"] < 64 * 1024**2:
                stop_owned("host available memory fell below 64 GiB")
                return 1
            numa_reason = runtime_pressure(
                item["numa_meminfo_kib"], item.get("owned_cgroup", {})
            )
            if numa_reason:
                stop_owned(numa_reason)
                return 1
            stable = all(len(values) == 3 and max(values) - min(values) <= 64 for values in baselines.values())
            if waiting and stable and not args.ready_file.exists():
                args.ready_file.write_text(json.dumps({
                    "time": time.time(), "status": "NATIVE_TP8_GUARD_ARMED",
                    "physical_devices": list(range(8)),
                    "baseline_mib": {str(k): max(v) for k, v in baselines.items()},
                }, indent=2) + "\n")
            if time.monotonic() - started > args.seconds:
                stop_owned("bounded native TP8 guard window ended")
                return 1
        except BaseException as error:
            try:
                state = inspect()
                if not state["State"]["Running"] and state["State"]["Status"] in ("exited", "dead"):
                    _append(args.output, {"time": time.time(), "action": "OWNED_EXITED_DURING_SAMPLE", "sampling_error": repr(error)})
                    return 0
                stop_owned(f"native TP8 guard failure: {error!r}")
            finally:
                raise
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
