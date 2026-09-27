#!/usr/bin/env python3
"""Run the frozen Native TP8 source tuple in its image without NPU devices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback


IMAGE = "601e76d17e9f"
MODEL_ROOT = Path("/home/admin/models/Qwen3.8-Flash-Next")
DRIVER_ROOT = Path("/usr/local/Ascend/driver")
OLD_QSA_SHA256 = "76e9f34066f0b42f42545427967c0b789d5c32031dc5c085e230efaf95e3c54c"
FORMAL_NATIVE_VERSIONS = (251, 252)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def docker_command(snapshot: Path, output: Path, *, image=IMAGE) -> list[str]:
    snapshot = Path(snapshot)
    output = Path(output)
    script = (
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh; "
        "unset ASCEND_RT_VISIBLE_DEVICES; "
        "exec python /snapshot/native_tp8_image_preflight.py "
        "--snapshot /snapshot --output /preflight/result.json"
    )
    command = [
        "docker", "run", "--rm", "--network", "none",
        "--cpus", "2", "--memory", "8g", "--memory-swap", "8g",
        "--pids-limit", "512",
        "-v", f"{snapshot}:/snapshot:ro",
        "-v", f"{output}:/preflight",
        "-v", f"{MODEL_ROOT}:/models:ro",
        "-v", f"{DRIVER_ROOT}:/usr/local/Ascend/driver:ro",
        "--entrypoint", "bash", image, "-lc", script,
    ]
    if "--device" in command:
        raise RuntimeError("same-image preflight must not expose an NPU device")
    return command


def validate_result(result: dict) -> bool:
    probe = result.get("no_device_import_probe")
    if (
        result.get("status") != "PASS_ROUND12A_NATIVE_TP8_SAME_IMAGE_PREFLIGHT"
        or result.get("physical_devices_exposed") != []
        or result.get("tp_size") != 8
        or result.get("qsa_sha256") != OLD_QSA_SHA256
        or not isinstance(probe, dict)
        or probe.get("restored") is not True
        or probe.get("mem_get_info_restored") is not True
        or probe.get("get_device_name_restored") is not True
        or probe.get("memory_capacity_for_serverargs_mib") != 65536
    ):
        raise RuntimeError("Round14 Native same-image identity gate failed")
    return True


def run(snapshot: Path, output: Path) -> dict:
    snapshot = Path(snapshot).resolve(strict=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    contract = json.loads((snapshot / "ROUND14_A_CONTRACT.json").read_text())
    if (contract.get("backend") != "NATIVE_TP8"
            or contract.get("version") not in FORMAL_NATIVE_VERSIONS):
        raise RuntimeError("Round14 Native repaired snapshot identity required")
    if not MODEL_ROOT.is_dir() or not DRIVER_ROOT.is_dir():
        raise RuntimeError("same-image read-only model/driver mount is unavailable")
    output.mkdir(parents=True)
    command = docker_command(snapshot, output)
    receipt = {
        "schema": 1,
        "status": "RUNNING",
        "version": contract["version"],
        "snapshot": str(snapshot),
        "snapshot_manifest_sha256": sha256(snapshot / "round14-source-sha256.json"),
        "physical_devices_exposed": [],
        "network": "none",
        "model_mount": f"{MODEL_ROOT}:/models:ro",
        "command": command,
        "started_at": time.time(),
    }
    receipt_path = output / "runner-receipt.json"

    def save():
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")

    save()
    try:
        with (output / "container.log").open("x") as log:
            completed = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT,
                timeout=900, check=False,
            )
        receipt["container_exit"] = completed.returncode
        if completed.returncode != 0:
            raise RuntimeError(f"same-image container exited {completed.returncode}")
        result = json.loads((output / "result.json").read_text())
        validate_result(result)
        receipt.update(
            status="PASS_ROUND14_NATIVE_SAME_IMAGE_PREFLIGHT",
            ended_at=time.time(),
            result_sha256=sha256(output / "result.json"),
            qsa_sha256=result["qsa_sha256"],
            tp_size=result["tp_size"],
        )
        save()
        checks = {
            name: sha256(output / name)
            for name in ("container.log", "result.json", "runner-receipt.json")
        }
        (output / "SHA256SUMS.json").write_text(
            json.dumps(checks, indent=2) + "\n"
        )
        return receipt
    except BaseException as error:
        receipt.update(
            status="FAIL_ROUND14_NATIVE_SAME_IMAGE_PREFLIGHT",
            ended_at=time.time(), error=repr(error), traceback=traceback.format_exc(),
        )
        save()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.snapshot, args.output)
    print(report["status"], report["version"], report["tp_size"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
