#!/usr/bin/env python3
"""Launch and supervise one immutable native TP8 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

from m3_cleanup import wait_for_guard_exit


BASE = Path("/mnt/data1/kt-round9")
MUTABLE = {"round12-source-sha256.json", "execution-budget.json", "execution-budget.lock", "host-controller.json"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(root: Path) -> dict:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    contract = json.loads((root / "ROUND12A_NATIVE_TP8_CONTRACT.json").read_text())
    out = BASE / "evidence" / contract["evidence"]
    expected = json.loads((root / "round12-source-sha256.json").read_text())
    if tree(root) != expected:
        raise RuntimeError("native TP8 snapshot manifest mismatch")
    claim = root / "host-controller.json"
    with claim.open("x") as stream:
        json.dump({"version": contract["version"], "status": "LAUNCHING", "started": time.time()}, stream)
    report = {"schema": 1, "version": contract["version"], "block": contract["block"], "status": "LAUNCHING", "started": time.time()}

    def save():
        claim.write_text(json.dumps(report, indent=2) + "\n")

    try:
        launch = ["python3", str(root / "native_tp8_launcher.py"), "--root", str(root), "--admission", str(args.admission)]
        report["launch_command"] = launch
        subprocess.run(launch, check=True, timeout=300)
        start = json.loads((out / "model-start.json").read_text())
        report.update(status="SUPERVISING", container_id=start["container_id"])
        save()
        supervisor = [
            "python3", str(root / "native_tp8_supervisor.py"), "--contract",
            str(root / "ROUND12A_NATIVE_TP8_CONTRACT.json"), "--container", start["name"],
            "--container-id", start["container_id"], "--out", str(out),
            "--driver-root", "/round9/" + str(root.relative_to(BASE)),
        ]
        subprocess.run(supervisor, check=True, timeout=21780)
        report["status"] = json.loads((out / "native-tp8-supervisor.json").read_text())["status"]
        return 0
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        receipt = out / "container-created.json"
        if receipt.exists():
            try:
                expected_id = json.loads(receipt.read_text())["Id"]
                found = json.loads(subprocess.check_output(["docker", "inspect", contract["container"]], text=True, timeout=30))[0]
                if found["Id"] != expected_id:
                    raise RuntimeError("native TP8 failed-launch container identity changed")
                if found["State"]["Running"]:
                    stopped = time.time()
                    subprocess.run(["docker", "stop", "-t", "10", expected_id], check=False, timeout=30)
                    report["failed_launch_guard_ack"] = wait_for_guard_exit(out / "monitor.jsonl", not_before=stopped)
                subprocess.run(["docker", "rm", expected_id], check=False, timeout=30)
            except BaseException as cleanup:
                report["cleanup_error"] = repr(cleanup)
        raise
    finally:
        report["ended"] = time.time()
        save()


if __name__ == "__main__":
    raise SystemExit(main())
