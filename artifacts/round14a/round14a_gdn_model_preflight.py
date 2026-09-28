#!/usr/bin/env python3
"""No-device same-image preflight for a Round14a GDN model snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROLES = {
    292: ("smoke", "baseline", "S"), 293: ("smoke", "candidate", "S"),
    294: ("abba", "baseline", "A"), 295: ("abba", "candidate", "A"),
    296: ("abba", "candidate", "B"), 297: ("abba", "baseline", "B"),
}
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
        raise RuntimeError(f"model preflight exposed NPU devices: {devices}")
    manifest = root / "round14a-source-sha256.json"
    if tree(root) != json.loads(manifest.read_text()):
        raise RuntimeError("Round14a GDN model manifest mismatch")
    contract = json.loads((root / "ROUND14A_A_GDN_MODEL_CONTRACT.json").read_text())
    version = contract["version"]
    expected = ROLES.get(version)
    if expected != (contract["kind"], contract["implementation"], contract["block"]):
        raise RuntimeError("Round14a GDN model contract role mismatch")
    expected_attempts = 8 if expected[0] == "smoke" else 20
    expected_aborts = 1 if expected[0] == "smoke" else 0
    if (contract.get("generation_limit") != expected_attempts
            or contract.get("abort_limit") != expected_aborts
            or contract.get("profiler") != "OFF"
            or contract.get("threshold") != 129
            or contract.get("automatic_retries") != 0):
        raise RuntimeError("Round14a GDN model budget/profile mismatch")
    framework = json.loads((root / "framework-profile.json").read_text())
    streaming = json.loads((root / "streaming-profile.json").read_text())
    if (framework["prefill"]["threshold_tokens"] != 129
            or streaming != {"schema": 1, "threshold_tokens": 129, "chunk_tokens": 32}):
        raise RuntimeError("Round14a GDN model is not H129/chunk32")
    qwen = root / "overlay-python/sglang/srt/models/qwen4_exp.py"
    binding = root / "overlay-python/sglang/srt/layers/round9/round14a_gdn_branch_binding.py"
    runner = root / "overlay-python/sglang/srt/model_executor/model_runner.py"
    if (sha256(qwen) != contract["frozen_qwen_sha256"]
            or "gdn_branch_model_ready(self)" not in runner.read_text()
            or "ROUND14A_GDN_BRANCH_HIT" not in binding.read_text()):
        raise RuntimeError("Round14a GDN actual-branch evidence hook mismatch")
    launcher = (root / "source/tools/launch_hybrid.py").read_text()
    enabled = "1" if expected[1] == "candidate" else "0"
    if (f"ROUND14_GDN_BACKEND_ENABLE={enabled}" not in launcher
            or "ROUND14A_GDN_BRANCH_PATH=/out/round14a-gdn-branch.jsonl" not in launcher
            or "ROUND13_ATTRIBUTION_ENABLE" in launcher):
        raise RuntimeError("Round14a GDN launcher binding/profiler mismatch")
    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        overlay = root / "overlay-python"
        receipt = json.loads((overlay / "round9-candidate-overlay.json").read_text())
        if receipt["candidate_tree_sha256"] != tree_digest(tree_files(overlay)):
            raise RuntimeError("Round14a GDN overlay receipt mismatch")
    finally:
        sys.path.remove(str(tools))
    for name in ("round14a_gdn_model_driver.py", "round14a_gdn_model_controller.py",
                 "round14a_gdn_model_supervisor.py", "round14a_gdn_model_preflight.py"):
        compile((root / name).read_text(), name, "exec")
    result = {"schema": 1, "status": "PASS_ROUND14A_A_GDN_MODEL_OFFLINE_PREFLIGHT",
              "version": version, "kind": expected[0], "implementation": expected[1],
              "block": expected[2], "physical_devices_exposed": devices,
              "manifest_sha256": sha256(manifest), "qwen_sha256": sha256(qwen),
              "actual_branch_hook": True, "profiler": "OFF", "threshold": 129}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], version, expected[1], expected[2], flush=True)


if __name__ == "__main__":
    main()

