#!/usr/bin/env python3
"""CPU-only self-test for immutable Round15 P0 snapshots."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys


EXPECTED_QWEN = "1986d8a880fe1503848eadf02a8ed572d543eac967fcb583b6f67d42250f3545"
ROLES = {337: "A", 338: "B"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify(root):
    root = Path(root).resolve(strict=True)
    preflight = load(root / "round15_p0_preflight.py", f"preflight_{root.name}")
    manifest = json.loads((root / "round15-source-sha256.json").read_text())
    if preflight.tree(root) != manifest:
        raise RuntimeError(f"{root.name}: immutable manifest mismatch")
    contract = json.loads((root / "ROUND15_P0_CONTRACT.json").read_text())
    if ROLES.get(contract.get("version")) != contract.get("block"):
        raise RuntimeError(f"{root.name}: repair role mismatch")
    if contract.get("graph") is not False or contract.get("generation_limit") != 4:
        raise RuntimeError(f"{root.name}: graph or generation contract changed")
    qwen = root / "overlay-python/sglang/srt/models/qwen4_exp.py"
    if sha256(qwen) != EXPECTED_QWEN:
        raise RuntimeError(f"{root.name}: promoted Qwen source mismatch")
    launcher = root / "source/tools/launch_hybrid.py"
    runner = root / "overlay-python/sglang/srt/model_executor/model_runner.py"
    module = root / "overlay-python/sglang/srt/layers/round9/round14a_five_bucket.py"
    runtime_module = root / "source/runtime/round14a_five_bucket.py"
    preflight.verify_five_bucket_census(launcher, runner, module)
    if sha256(module) != sha256(runtime_module):
        raise RuntimeError(f"{root.name}: census source identities differ")
    census = load(module, f"round14a_five_bucket_{root.name}")
    if census.classify_model_name("layers.0.mlp.experts.3.up") != "Expert":
        raise RuntimeError(f"{root.name}: census classifier regression")
    if census.classify_model_name("layers.0.ple.weight") != "PLE":
        raise RuntimeError(f"{root.name}: PLE census classifier regression")
    driver = load(root / "round15_p0_driver.py", f"driver_{root.name}")
    control = load(root / "round15_p0_control.py", f"control_{root.name}")
    registered = control.classify_physical12_after(
        "Process id:2980236 Process name:python3.10 Process memory(MB):6123\n",
        lambda _pid: "0::/docker/" + control.TRAINING_ID + "\n",
    )
    if registered["status"] != "REGISTERED_TRAINING_ONLY":
        raise RuntimeError(f"{root.name}: registered-training cleanup control regression")
    if driver.normalize_result({"profiler": "BOUNDED_OBSERVER_ON"})["profiler"] != (
        "BOUNDED_EXISTING_EVENT_OBSERVER"
    ):
        raise RuntimeError(f"{root.name}: profiler metadata normalization regression")
    fixture = json.loads((root / "round11-t1-fixture.json").read_text())
    schedule = driver.build_schedule(
        fixture, version=contract["version"], block=contract["block"]
    )
    if [row["workload"] for row in schedule] != ["WARMUP", "W1", "W2", "DRAIN"]:
        raise RuntimeError(f"{root.name}: request schedule changed")
    return {
        "root": str(root),
        "version": contract["version"],
        "block": contract["block"],
        "manifest_sha256": sha256(root / "round15-source-sha256.json"),
        "qwen_sha256": sha256(qwen),
        "census_sha256": sha256(module),
        "model_runner_sha256": sha256(runner),
        "schedule": [row["workload"] for row in schedule],
        "control_plane": {
            "registered_training": registered["status"],
            "profiler": "BOUNDED_EXISTING_EVENT_OBSERVER",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": 1,
        "status": "PASS_ROUND15_P0_CPU_ONLY_SNAPSHOT_SELFTEST",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "torch": package_version("torch"),
            "torch_npu": package_version("torch-npu"),
        },
        "physical_devices_exposed": sorted(str(path) for path in Path("/dev").glob("davinci*")),
        "snapshots": [verify(root) for root in args.root],
    }
    if report["physical_devices_exposed"]:
        raise RuntimeError("CPU-only self-test unexpectedly exposed NPU devices")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], len(report["snapshots"]), flush=True)


if __name__ == "__main__":
    main()
