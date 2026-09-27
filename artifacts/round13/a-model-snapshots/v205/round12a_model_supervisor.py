#!/usr/bin/env python3
"""Supervise and clean one immutable Round12a model process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
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
        raise ValueError("fixed Round12a model identity required")

    def command(values, timeout=30):
        return subprocess.check_output(values, text=True, timeout=timeout)

    def inspect():
        document = json.loads(command(["docker", "inspect", args.container]))[0]
        if document["Id"] != args.container_id:
            raise RuntimeError("owned container identity changed")
        return document

    report = {
        "schema": 1,
        "status": "WAITING_FOR_SERVER",
        "version": contract["version"],
        "mode": contract["mode"],
        "stage": contract["stage"],
        "container_id": args.container_id,
        "started_at": time.time(),
        "generation_driver_starts": 0,
    }
    path = args.out / "round12a-model-supervisor.json"
    path.touch(exist_ok=False)

    def save() -> None:
        path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    child = raw = None
    raw_name = contract["raw_result"]
    try:
        deadline = time.time() + 21600
        while time.time() < deadline:
            document = inspect()
            if not document["State"]["Running"]:
                raise RuntimeError("server exited before Round12a model completion")
            if any(file.stat().st_size > 512 * 1024**2 for file in args.out.glob("*.log")):
                raise RuntimeError("Round12a log ceiling exceeded")
            log = args.out / "server.log"
            if child is None and log.exists():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 65536))
                    tail = stream.read().decode(errors="replace")
                ready = (
                    "Uvicorn running on http://127.0.0.1:30091" in tail
                    or "The server is fired up and ready to roll!" in tail
                )
                if ready:
                    if (args.out / raw_name).exists():
                        raise RuntimeError("Round12a model ledger already exists; no retry")
                    driver = ["docker", "exec", args.container_id, "python"] + [
                        value.replace("{DRIVER_ROOT}", args.driver_root)
                        for value in contract["driver_command"]
                    ]
                    report.update(
                        status="RUNNING_DRIVER",
                        server_ready_at=time.time(),
                        generation_driver_starts=1,
                        driver_command=driver,
                    )
                    save()
                    with (args.out / "round12a-model-driver.log").open("x") as stream:
                        child = subprocess.Popen(
                            driver, stdout=stream, stderr=subprocess.STDOUT
                        )
            if child is not None and child.poll() is not None:
                report["driver_exit"] = child.returncode
                raw_path = args.out / raw_name
                if not raw_path.exists():
                    raise RuntimeError(
                        "Round12a driver exited before publishing raw result"
                    )
                raw = json.loads(raw_path.read_text())
                if (
                    raw.get("generation_attempts") != contract["generation_limit"]
                    or raw.get("abort_controls", 0) != contract["abort_limit"]
                    or raw.get("automatic_retries") != 0
                    or raw.get("profiler") != "OFF"
                ):
                    raise RuntimeError("Round12a model result budget mismatch")
                report["status"] = (
                    "DRIVER_PASS_WAITING_CLEANUP"
                    if child.returncode == 0
                    and raw.get("status") == contract["driver_pass_status"]
                    else "FAIL"
                )
                break
            time.sleep(5)
        else:
            raise TimeoutError("bounded Round12a model supervisor deadline")
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    finally:
        report["ended_at"] = time.time()
        save()
        try:
            document = inspect()
            stop_started = time.time()
            was_running = document["State"]["Running"]
            if was_running:
                subprocess.run(
                    ["docker", "stop", "-t", "10", args.container_id],
                    check=False,
                    timeout=30,
                )
            report["owned_exit_state"] = inspect()["State"]
            report["guard_exit_ack"] = wait_for_guard_exit(
                args.out / "monitor.jsonl",
                not_before=stop_started if was_running else report["started_at"],
            )
            subprocess.run(
                ["docker", "rm", args.container_id], check=True, timeout=30
            )
            report["owned_container_removed"] = True
            proc = command(["npu-smi", "info", "-t", "proc-mem", "-i", "6", "-c", "0"])
            (args.out / "post-cleanup-proc-mem.txt").write_text(proc)
            if "No process in device" not in proc:
                raise RuntimeError("physical12 is not empty after Round12a cleanup")
            report["physical12_after"] = "EMPTY"
        except BaseException as cleanup:
            report["cleanup_error"] = repr(cleanup)
            report["status"] = "FAIL_CLEANUP"
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.terminate()
        if report["status"] == "DRIVER_PASS_WAITING_CLEANUP" and raw is not None:
            final = dict(raw)
            final["status"] = "PASS_ROUND12A_MODEL_PROCESS"
            final["startup"] = {
                "supervisor_started_at": report["started_at"],
                "server_ready_at": report.get("server_ready_at"),
                "seconds": report.get("server_ready_at", report["started_at"])
                - report["started_at"],
            }
            final["cleanup"] = {
                "guard_exit_ack": True,
                "container_removed": True,
                "physical12_after": "EMPTY",
            }
            (args.out / "round12a-model-final.json").write_text(
                json.dumps(final, indent=2) + "\n"
            )
            report["status"] = "PASS_ROUND12A_MODEL_PROCESS"
        save()
    if report["status"] != "PASS_ROUND12A_MODEL_PROCESS":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
