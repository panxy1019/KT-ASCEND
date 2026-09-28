#!/usr/bin/env python3
"""No-device same-image preflight for a Round14a concurrency snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROLES = {
    322: ("smoke", "C2", (2,), 2), 323: ("smoke", "C2", (2,), 2),
    324: ("smoke", "C2", (2,), 2), 325: ("smoke", "C2", (2,), 2),
    326: ("smoke", "C4", (4,), 4),
    327: ("mirror", "A", (1, 2, 4), 4),
    328: ("mirror", "B", (4, 2, 1), 4),
    329: ("mirror", "C", (4, 2, 1), 4),
    330: ("mirror", "D", (1, 2, 4), 4),
}
MUTABLE = {"round14a-source-sha256.json", "execution-budget.json",
           "execution-budget.lock", "host-controller.json"}


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(root):
    root = Path(root)
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
            and path.suffix != ".pyc" and str(path.relative_to(root)) not in MUTABLE
            and not str(path.relative_to(root)).startswith("host-controller")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    devices = sorted(str(path) for path in Path("/dev").glob("davinci[0-9]*"))
    if devices: raise RuntimeError(f"preflight exposed NPU devices: {devices}")
    manifest = root / "round14a-source-sha256.json"
    if tree(root) != json.loads(manifest.read_text()):
        raise RuntimeError("Round14a concurrency manifest mismatch")
    contract = json.loads((root / "ROUND14A_C_CONCURRENCY_CONTRACT.json").read_text())
    role = ROLES.get(contract["version"])
    if role is None or (contract["kind"], contract["block"],
                        tuple(contract["tier_order"]), contract["server_max_concurrency"]) != role:
        raise RuntimeError("Round14a concurrency role mismatch")
    expected_attempts = 9 if role[0] == "smoke" else 82
    expected_aborts = 1 if role[0] == "smoke" else 0
    if (contract["generation_limit"] != expected_attempts
            or contract["abort_limit"] != expected_aborts
            or contract["profiler"] != "OFF" or contract["threshold"] != 129):
        raise RuntimeError("Round14a concurrency budget/profile mismatch")
    profile = json.loads((root / "source/runtime/profiles/q64_bf16_c64_sync_physical15.json").read_text())
    if (profile["max_running_requests"] != role[3]
            or profile["max_mamba_cache_size"] != role[3]
            or not profile["disable_overlap_schedule"]):
        raise RuntimeError("Round14a concurrency server pool mismatch")
    launcher = (root / "source/tools/launch_hybrid.py").read_text()
    if (f"ROUND14A_CONCURRENCY_MAX={role[3]}" not in launcher
            or "ROUND14_GDN_BACKEND_ENABLE=0" not in launcher
            or "ROUND13_ATTRIBUTION_ENABLE" in launcher):
        raise RuntimeError("Round14a concurrency launcher identity mismatch")
    overlay = root / "overlay-python"
    qwen = overlay / "sglang/srt/models/qwen4_exp.py"
    runtime = overlay / "sglang/srt/layers/round9/round14a_concurrency_runtime.py"
    for marker in ("request_slices", "consume_for", "Round14a GDN requires"):
        if marker not in qwen.read_text():
            raise RuntimeError(f"Round14a concurrency Qwen marker missing: {marker}")
    if sha256(runtime) != contract["concurrency_runtime_sha256"]:
        raise RuntimeError("Round14a concurrency runtime hash mismatch")
    os.environ["ROUND14A_CONCURRENCY_MAX"] = str(role[3])
    sys.path.insert(0, str(overlay))
    try:
        from sglang.srt.layers.round9.round14a_concurrency_runtime import configured_max
        if configured_max() != role[3]: raise RuntimeError("concurrency runtime env mismatch")
    finally:
        sys.path.remove(str(overlay))
    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        receipt = json.loads((overlay / "round9-candidate-overlay.json").read_text())
        if receipt["candidate_tree_sha256"] != tree_digest(tree_files(overlay)):
            raise RuntimeError("Round14a concurrency overlay receipt mismatch")
    finally: sys.path.remove(str(tools))
    for name in ("round14a_concurrency_driver.py", "round14a_concurrency_controller.py",
                 "round14a_concurrency_supervisor.py", "round14a_concurrency_preflight.py"):
        compile((root / name).read_text(), name, "exec")
    value = {"schema": 1, "status": "PASS_ROUND14A_C_CONCURRENCY_OFFLINE_PREFLIGHT",
             "version": contract["version"], "kind": role[0], "block": role[1],
             "tier_order": list(role[2]), "server_max_concurrency": role[3],
             "physical_devices_exposed": devices, "manifest_sha256": sha256(manifest),
             "qwen_sha256": sha256(qwen), "profiler": "OFF", "threshold": 129}
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(value["status"], contract["version"], role[1], flush=True)


if __name__ == "__main__": main()
