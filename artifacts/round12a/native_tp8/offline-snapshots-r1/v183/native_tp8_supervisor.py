#!/usr/bin/env python3
"""Supervise, stop, and remove one native TP8 model process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from m3_cleanup import wait_for_guard_exit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--driver-root", required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    if args.container != contract["container"] or len(args.container_id) != 64:
        raise ValueError("fixed native TP8 identity required")

    def command(values, timeout=30):
        return subprocess.check_output(values, text=True, timeout=timeout)

    def inspect():
        document = json.loads(command(["docker", "inspect", args.container]))[0]
        if document["Id"] != args.container_id:
            raise RuntimeError("native TP8 owned container identity changed")
        return document

    report = {
        "schema": 1, "status": "WAITING_FOR_SERVER", "version": contract["version"],
        "mode": "NATIVE_TP8", "block": contract["block"], "container_id": args.container_id,
        "started_at": time.time(), "generation_driver_starts": 0,
    }
    path = args.out / "native-tp8-supervisor.json"
    path.touch(exist_ok=False)

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    child = None
    try:
        deadline = time.time() + 21600
        while time.time() < deadline:
            document = inspect()
            if not document["State"]["Running"]:
                raise RuntimeError("native TP8 server exited before completion")
            if any(file.stat().st_size > 512 * 1024**2 for file in args.out.glob("*.log")):
                raise RuntimeError("native TP8 log ceiling exceeded")
            log = args.out / "server.log"
            if child is None and log.exists():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 65536))
                    tail = stream.read().decode(errors="replace")
                if "Uvicorn running on http://127.0.0.1:30091" in tail or "The server is fired up and ready to roll!" in tail:
                    raw = args.out / contract["raw_result"]
                    if raw.exists():
                        raise RuntimeError("native TP8 request ledger already exists; no retry")
                    driver = ["docker", "exec", args.container_id, "python"] + [
                        value.replace("{DRIVER_ROOT}", args.driver_root) for value in contract["driver_command"]
                    ]
                    child = subprocess.Popen(driver, stdout=(args.out / "native-tp8-driver.log").open("x"), stderr=subprocess.STDOUT)
                    report.update(status="RUNNING_DRIVER", server_ready_at=time.time(), generation_driver_starts=1, driver_command=driver)
                    save()
            if child is not None:
                code = child.poll()
                if code is not None:
                    if code != 0:
                        raise RuntimeError(f"native TP8 driver exited {code}")
                    break
            time.sleep(1)
        else:
            raise TimeoutError("native TP8 bounded model window expired")
        raw = json.loads((args.out / contract["raw_result"]).read_text())
        if raw.get("status") != contract["driver_pass_status"] or raw.get("generation_attempts") != 4:
            raise RuntimeError("native TP8 raw result failed")
        stop_started = time.time()
        subprocess.run(["docker", "stop", "-t", "10", args.container_id], check=True, timeout=30)
        report["guard_exit_ack"] = wait_for_guard_exit(args.out / "monitor.jsonl", not_before=stop_started)
        sys.path.insert(0, str(args.contract.parent))
        from native_tp8_admission import parse_first8_processes
        processes = parse_first8_processes(command(["npu-smi", "info"]))
        if any(processes[device] for device in range(8)):
            raise RuntimeError("physical0-7 retained a process after native TP8 stop")
        rows = [json.loads(line) for line in (args.out / "monitor.jsonl").read_text().splitlines() if line.strip()]
        samples = [row for row in rows if "devices" in row]
        peaks = {
            str(device): max(row["devices"][str(device)]["used_mib"] for row in samples)
            for device in range(8)
        }
        cgroup_peaks = []
        min_host_available_kib = min(row["host_mem_available_kib"] for row in samples)
        min_numa_memfree_kib = {
            str(node): min(
                row["numa_meminfo_kib"][str(node)]["MemFree"] for row in samples
            )
            for node in (2, 3, 4, 7)
        }
        for row in samples:
            cg = row.get("owned_cgroup", {})
            value = cg.get("memory.peak") or cg.get("memory.max_usage_in_bytes")
            if value is not None:
                cgroup_peaks.append(int(value))
        subprocess.run(["docker", "rm", args.container_id], check=True, timeout=30)
        final = {
            **raw, "status": "PASS_ROUND12A_NATIVE_TP8_MODEL_PROCESS",
            "resource": {
                "per_device_peak_hbm_mib": peaks,
                "host_cgroup_peak_bytes": max(cgroup_peaks) if cgroup_peaks else None,
                "min_host_mem_available_kib": min_host_available_kib,
                "min_numa_memfree_kib": min_numa_memfree_kib,
                "physical0_7_processes_after_stop": processes,
            },
            "cleanup": {"guard_exit_ack": report["guard_exit_ack"], "container_removed": True},
        }
        (args.out / "native-tp8-final.json").write_text(json.dumps(final, indent=2) + "\n")
        report.update(status=final["status"], ended_at=time.time(), final_result="native-tp8-final.json")
        save()
        print(report["status"], contract["version"], flush=True)
        return 0
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc(), ended_at=time.time())
        save()
        if child is not None and child.poll() is None:
            child.kill()
        try:
            document = inspect()
            if document["State"]["Running"]:
                stop_started = time.time()
                subprocess.run(["docker", "stop", "-t", "10", args.container_id], check=False, timeout=30)
                report["failure_guard_exit_ack"] = wait_for_guard_exit(args.out / "monitor.jsonl", not_before=stop_started)
            subprocess.run(["docker", "rm", args.container_id], check=False, timeout=30)
        except BaseException as cleanup:
            report["cleanup_error"] = repr(cleanup)
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
