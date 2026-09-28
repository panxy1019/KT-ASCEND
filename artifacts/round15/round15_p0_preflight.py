#!/usr/bin/env python3
"""No-device source, identity and hook preflight for Round15 P0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROLES = {
    331: "A", 332: "B", 333: "A", 334: "B",
    335: "A", 336: "B",
    337: "A", 338: "B",
}
QWEN_SHA256 = "1986d8a880fe1503848eadf02a8ed572d543eac967fcb583b6f67d42250f3545"
MUTABLE = {
    "round15-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_qwen(path):
    if sha256(path) != QWEN_SHA256:
        raise RuntimeError("Round15 P0 requires the exact promoted Qwen source")


def verify_five_bucket_census(launcher_path, runner_path, module_path):
    launcher = Path(launcher_path).read_text()
    runner = Path(runner_path).read_text()
    module = Path(module_path)
    required = (
        "ROUND14A_FIVE_BUCKET_ENABLE=1",
        "ROUND14A_FIVE_BUCKET_PATH=/out/round14a-five-bucket-memory.json",
    )
    if (
        not module.is_file()
        or any(value not in launcher for value in required)
        or runner.count("round14a_five_bucket.capture(self)") != 1
        or "gdn_branch_model_ready(self)" not in runner
        or runner.index("round14a_five_bucket.capture(self)")
            < runner.index("gdn_branch_model_ready(self)")
    ):
        raise RuntimeError("Round15 P0 five-bucket producer integration is incomplete")


def tree(root):
    root = Path(root)
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    devices = sorted(str(path) for path in Path("/dev").glob("davinci[0-9]*"))
    if devices:
        raise RuntimeError(f"Round15 P0 offline preflight exposed NPU devices: {devices}")
    manifest = root / "round15-source-sha256.json"
    if tree(root) != json.loads(manifest.read_text()):
        raise RuntimeError("Round15 P0 immutable manifest mismatch")
    contract = json.loads((root / "ROUND15_P0_CONTRACT.json").read_text())
    if ROLES.get(contract.get("version")) != contract.get("block"):
        raise RuntimeError("Round15 P0 role mismatch")
    if (
        contract.get("generation_limit") != 4
        or contract.get("abort_limit") != 0
        or contract.get("profiler") != "BOUNDED_EXISTING_EVENT_OBSERVER"
        or contract.get("graph") is not False
        or contract.get("automatic_retries") != 0
    ):
        raise RuntimeError("Round15 P0 budget or mode mismatch")
    verify_qwen(root / "overlay-python/sglang/srt/models/qwen4_exp.py")
    launcher = (root / "source/tools/launch_hybrid.py").read_text()
    scheduler = (root / "overlay-python/sglang/srt/managers/scheduler.py").read_text()
    required = (
        "ROUND14_GDN_BACKEND_ENABLE=1",
        "ROUND13_ATTRIBUTION_ENABLE=1",
        "ROUND15_P0_PROFILE=1",
        "ROUND14A_FIVE_BUCKET_ENABLE=1",
        "ROUND14A_FIVE_BUCKET_PATH=/out/round14a-five-bucket-memory.json",
    )
    if any(value not in launcher for value in required) or "round13_before(self, batch)" not in scheduler:
        raise RuntimeError("Round15 P0 observer or promoted-GDN hooks are incomplete")
    runner = root / "overlay-python/sglang/srt/model_executor/model_runner.py"
    census_module = root / "overlay-python/sglang/srt/layers/round9/round14a_five_bucket.py"
    verify_five_bucket_census(root / "source/tools/launch_hybrid.py", runner, census_module)
    runtime_census = root / "source/runtime/round14a_five_bucket.py"
    if not runtime_census.is_file() or sha256(runtime_census) != sha256(census_module):
        raise RuntimeError("Round15 P0 five-bucket source identity mismatch")
    control = (root / "round15_p0_control.py").read_text()
    supervisor = (root / "round15_p0_supervisor.py").read_text()
    controller = (root / "round15_p0_controller.py").read_text()
    if (
        "REGISTERED_TRAINING_ONLY" not in control
        or "classify_physical12_after(proc, read_proc_cgroup)" not in supervisor
        or "validate_supervisor_removed_container(supervisor, expected)" not in controller
        or "BOUNDED_EXISTING_EVENT_OBSERVER" not in (root / "round15_p0_driver.py").read_text()
    ):
        raise RuntimeError("Round15 P0 A3 control-plane compatibility integration is incomplete")
    result = {
        "schema": 1,
        "status": "PASS_ROUND15_P0_OFFLINE_PREFLIGHT",
        "version": contract["version"],
        "block": contract["block"],
        "manifest_sha256": sha256(manifest),
        "qwen_sha256": QWEN_SHA256,
        "physical_devices_exposed": devices,
        "graph": False,
        "observer": "DEFERRED_EVENT_QUERY_NO_NEW_SYNCHRONIZE",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], contract["version"], contract["block"], flush=True)


if __name__ == "__main__":
    main()
