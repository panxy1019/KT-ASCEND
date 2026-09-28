#!/usr/bin/env python3
"""Launch, supervise, and clean one immutable Round14a B profile model process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

from m3_cleanup import wait_for_guard_exit


TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
BASE = Path("/mnt/data1/kt-round9")
MUTABLE = {
    "round14a-source-sha256.json", "execution-budget.json",
    "execution-budget.lock", "host-controller.json",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(root):
    return {
        str(path.relative_to(root)): sha256(path)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    contract = json.loads((root / "ROUND14A_B_PROFILE_CONTRACT.json").read_text())
    out = BASE / "evidence" / contract["evidence"]
    if tree(root) != json.loads((root / "round14a-source-sha256.json").read_text()):
        raise RuntimeError("Round14a B profile snapshot manifest mismatch")
    claim = root / "host-controller.json"
    with claim.open("x") as stream:
        json.dump({"version": contract["version"], "stage": contract["stage"],
                   "started": time.time(), "status": "LAUNCHING"}, stream)
    report = {"schema": 1, "version": contract["version"],
              "stage": contract["stage"], "status": "LAUNCHING",
              "started": time.time()}

    def save():
        claim.write_text(json.dumps(report, indent=2) + "\n")

    def cleanup_failed_launch():
        receipt = out / "container-created.json"
        if not receipt.exists():
            return {"status": "NO_VERIFIED_OWNED_CONTAINER_RECEIPT"}
        expected = json.loads(receipt.read_text())["Id"]
        found = json.loads(subprocess.check_output(
            ["docker", "inspect", contract["container"]], text=True, timeout=30))[0]
        if found["Id"] != expected:
            raise RuntimeError("failed-launch owned container identity changed")
        stop_started = time.time()
        if found["State"]["Running"]:
            subprocess.run(["docker", "stop", "-t", "10", expected],
                           check=True, timeout=30)
        ack = wait_for_guard_exit(out / "monitor.jsonl", not_before=stop_started)
        subprocess.run(["docker", "rm", expected], check=True, timeout=30)
        return {"status": "REMOVED_AFTER_GUARD_ACK", "guard_exit_ack": ack,
                "container_id": expected}

    launch = [
        "python3", str(root / "source/tools/launch_hybrid.py"),
        "--version", str(contract["version"]), "--storage", "MXFP4",
        "--placement", "static", "--resource-mode", "shared",
        "--shared-training-container", TRAINING_ID,
        "--bank-receipt", str(root / "rebound/bank-receipt.json"),
        "--device-gate", str(BASE / "evidence/20260920-c128-expert-v19/result.json"),
        "--overlay-python", str(root / "overlay-python"),
        "--framework-profile", str(root / "framework-profile.json"),
        "--context-target", "4096",
        "--context-reference-profile", str(BASE / "evidence/20260922-plan10-m3-baseline-v64/context-reference-profile.json"),
        "--streaming-profile", str(root / "streaming-profile.json"),
        "--cancel-trace", "--cancel-closure", "--cost-mode", "OFF",
        "--cost-manifest", str(root / "cost-manifest.json"),
        "--cost-budget", str(root / "execution-budget.json"),
        "--cost-admission", str(args.admission),
    ]
    report["launch_command"] = launch
    try:
        subprocess.run(launch, check=True, timeout=300)
        start = json.loads((out / "model-start.json").read_text())
        report.update(status="SUPERVISING", container_id=start["container_id"])
        save()
        supervisor = [
            "python3", str(root / "round14a_b_profile_supervisor.py"),
            "--contract", str(root / "ROUND14A_B_PROFILE_CONTRACT.json"),
            "--container", start["name"], "--container-id", start["container_id"],
            "--out", str(out), "--driver-root",
            "/round9/" + str(root.relative_to(BASE)),
        ]
        subprocess.run(supervisor, check=True, timeout=21780)
        report["status"] = json.loads((out / "round14a-b-profile-supervisor.json").read_text())["status"]
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        try:
            report["failed_launch_cleanup"] = cleanup_failed_launch()
        except BaseException as cleanup:
            report["cleanup_error"] = repr(cleanup)
        raise
    finally:
        report["ended"] = time.time()
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
