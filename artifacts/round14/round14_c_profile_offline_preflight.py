#!/usr/bin/env python3
"""No-device, same-image preflight for a frozen Round14 C profile snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


MUTABLE = {
    "round14-source-sha256.json", "execution-budget.json",
    "execution-budget.lock", "host-controller.json",
}
ROLES = {249: "A", 250: "B", 274: "A", 275: "B"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_devices():
    return sorted(str(path) for path in Path("/dev").glob("davinci[0-9]*"))


def preflight(root, *, physical_devices=None):
    root = Path(root).resolve(strict=True)
    devices = discover_devices() if physical_devices is None else list(physical_devices)
    if devices:
        raise RuntimeError("Round14 C offline preflight must not expose an NPU device")
    manifest_path = root / "round14-source-sha256.json"
    manifest = json.loads(manifest_path.read_text())
    if tree(root) != manifest:
        raise RuntimeError("Round14 C snapshot manifest mismatch")
    contract = json.loads((root / "ROUND14_C_PROFILE_CONTRACT.json").read_text())
    version = contract.get("version")
    block = contract.get("block")
    if (
        ROLES.get(version) != block
        or contract.get("stage") != "ROUND14_C_FRESH_S0_PROFILE"
        or contract.get("generation_limit") != 4
        or contract.get("abort_limit") != 0
        or contract.get("automatic_retries") != 0
        or contract.get("profiler") != "BOUNDED_OBSERVER_ON"
        or contract.get("profile_threshold") != 513
        or contract.get("default_threshold") != 129
    ):
        raise RuntimeError("Round14 C contract mismatch")
    launcher_path = root / "source/tools/launch_hybrid.py"
    launcher = launcher_path.read_text()
    required = (
        f"if a.version != {version}", contract["container"], contract["evidence"],
        "ROUND13_ATTRIBUTION_ENABLE=1",
        "ROUND13_ATTRIBUTION_SPANS=/out/round14-c-spans.jsonl",
        "ROUND13_ATTRIBUTION_BINDINGS=/out/round14-c-bindings.json",
    )
    if any(value not in launcher for value in required):
        raise RuntimeError("Round14 C launcher/source gate mismatch")
    compile(launcher, str(launcher_path), "exec")
    scheduler_path = root / "overlay-python/sglang/srt/managers/scheduler.py"
    scheduler = scheduler_path.read_text()
    if scheduler.count("round13_before(self, batch)") != 1 or scheduler.count(
        "round13_after(self)"
    ) != 1:
        raise RuntimeError("Round14 C scheduler observer hook mismatch")
    compile(scheduler, str(scheduler_path), "exec")
    observer = root / "overlay-python/sglang/srt/layers/round9/round13_attribution.py"
    compile(observer.read_text(), str(observer), "exec")

    overlay = root / "overlay-python"
    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        receipt = json.loads((overlay / "round9-candidate-overlay.json").read_text())
        profile = receipt.get("round14_c_profile", {})
        if (
            receipt.get("candidate_tree_sha256") != tree_digest(tree_files(overlay))
            or profile.get("status") != "INSTALLED_BOUNDED_ATTRIBUTION"
            or profile.get("device_query") != "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE"
        ):
            raise RuntimeError("Round14 C overlay receipt mismatch")
    finally:
        sys.path.remove(str(tools))

    driver = _load("round14_c_profile_driver_preflight", root / "round14_c_profile_driver.py")
    fixture_path = root / "round11-t1-fixture.json"
    schedule = driver.build_schedule(
        json.loads(fixture_path.read_text()), version=version, block=block
    )
    if len(schedule) != 4 or [row["workload"] for row in schedule] != [
        "WARMUP", "W2", "W3", "DRAIN",
    ]:
        raise RuntimeError("Round14 C request schedule mismatch")
    return {
        "schema": 1,
        "status": "PASS_ROUND14_C_PROFILE_OFFLINE_PREFLIGHT",
        "version": version,
        "block": block,
        "physical_devices_exposed": devices,
        "manifest_sha256": sha256(manifest_path),
        "overlay_tree_sha256": receipt["candidate_tree_sha256"],
        "fixture_sha256": sha256(fixture_path),
        "schedule_requests": len(schedule),
        "observer_device_query": profile["device_query"],
        "model_math_change": "NONE_OBSERVER_ONLY",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = preflight(args.root)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report["status"], report["version"], report["block"], flush=True)


if __name__ == "__main__":
    main()
