#!/usr/bin/env python3
"""No-device same-image preflight for one Round14a B profile snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROLES = {298: "A", 299: "B"}
MUTABLE = {"round14a-source-sha256.json", "execution-budget.json",
           "execution-budget.lock", "host-controller.json"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            and str(p.relative_to(root)) not in MUTABLE
            and not str(p.relative_to(root)).startswith("host-controller")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    devices = sorted(str(path) for path in Path("/dev").glob("davinci[0-9]*"))
    if devices:
        raise RuntimeError(f"Round14a B preflight exposed NPU devices: {devices}")
    manifest = root / "round14a-source-sha256.json"
    if tree(root) != json.loads(manifest.read_text()):
        raise RuntimeError("Round14a B profile manifest mismatch")
    contract = json.loads((root / "ROUND14A_B_PROFILE_CONTRACT.json").read_text())
    version = contract["version"]
    if ROLES.get(version) != contract.get("block"):
        raise RuntimeError("Round14a B profile role mismatch")
    if (contract.get("generation_limit") != 4 or contract.get("abort_limit") != 0
            or contract.get("profiler") != "BOUNDED_OBSERVER_ON"
            or contract.get("threshold") != 129
            or contract.get("automatic_retries") != 0):
        raise RuntimeError("Round14a B profile budget mismatch")
    framework = json.loads((root / "framework-profile.json").read_text())
    streaming = json.loads((root / "streaming-profile.json").read_text())
    if (framework["prefill"]["threshold_tokens"] != 129
            or streaming != {"schema": 1, "threshold_tokens": 129, "chunk_tokens": 32}):
        raise RuntimeError("Round14a B profile is not H129/chunk32")
    runner = (root / "overlay-python/sglang/srt/model_executor/model_runner.py").read_text()
    scheduler = (root / "overlay-python/sglang/srt/managers/scheduler.py").read_text()
    launcher = (root / "source/tools/launch_hybrid.py").read_text()
    if ("round14a_five_bucket.capture" not in runner
            or "round13_before(self, batch)" not in scheduler
            or "ROUND14A_FIVE_BUCKET_ENABLE=1" not in launcher
            or "ROUND13_ATTRIBUTION_ENABLE=1" not in launcher):
        raise RuntimeError("Round14a B profile runtime hooks are incomplete")
    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        overlay = root / "overlay-python"
        receipt = json.loads((overlay / "round9-candidate-overlay.json").read_text())
        if receipt["candidate_tree_sha256"] != tree_digest(tree_files(overlay)):
            raise RuntimeError("Round14a B overlay receipt mismatch")
    finally:
        sys.path.remove(str(tools))
    for name in ("round14a_b_profile_driver.py", "round14a_b_profile_controller.py",
                 "round14a_b_profile_supervisor.py", "round14a_b_profile_preflight.py"):
        compile((root / name).read_text(), name, "exec")
    result = {
        "schema": 1,
        "status": "PASS_ROUND14A_B_PROFILE_OFFLINE_PREFLIGHT",
        "version": version,
        "block": contract["block"],
        "physical_devices_exposed": devices,
        "manifest_sha256": sha256(manifest),
        "threshold": 129,
        "storage_deduplication": "DEVICE_STORAGE_POINTER_AND_STORAGE_BYTES",
        "observer": "DEFERRED_EVENT_QUERY_NO_NEW_SYNCHRONIZE",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], version, contract["block"], flush=True)


if __name__ == "__main__":
    main()
