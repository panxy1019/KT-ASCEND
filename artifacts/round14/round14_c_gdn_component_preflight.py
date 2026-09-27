#!/usr/bin/env python3
"""No-device same-image gate for a frozen Round14 C GDN component snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROLES = {
    276: "A", 277: "B", 278: "A", 279: "B", 280: "A", 281: "B",
    282: "A", 283: "B", 284: "A", 285: "B", 286: "A", 287: "B",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(root):
    root = Path(root)
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"source-sha256.json", "host-controller.json"}
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    devices = sorted(str(path) for path in Path("/dev").glob("davinci[0-9]*"))
    if devices:
        raise RuntimeError(f"component preflight exposed NPU devices: {devices}")
    manifest_path = root / "source-sha256.json"
    manifest = json.loads(manifest_path.read_text())
    if tree(root) != manifest:
        raise RuntimeError("component snapshot manifest mismatch")
    contract = json.loads(
        (root / "ROUND14_C_GDN_COMPONENT_CONTRACT.json").read_text()
    )
    version, block = contract.get("version"), contract.get("block")
    if (
        ROLES.get(version) != block
        or contract.get("stage") != "ROUND14_C_GDN_BACKEND_COMPONENT"
        or contract.get("candidate_default") != "OFF"
        or contract.get("automatic_retries") != 0
    ):
        raise RuntimeError("component contract mismatch")
    qwen_path = root / "overlay-python/sglang/srt/models/qwen4_exp.py"
    qwen = qwen_path.read_text()
    if (
        qwen.count("ROUND14_GDN_BACKEND_ENABLE") != 1
        or qwen.count("sigmoid_gated_rms_norm(") != 1
        or 'os.environ.get("ROUND14_GDN_BACKEND_ENABLE", "0") == "1"' not in qwen
        or "NPU_GDN_CORE_PLUS_FROZEN_QWEN4_SIGMOID_GATE" not in json.dumps(
            json.loads(
                (root / "overlay-python/round9-candidate-overlay.json").read_text()
            )
        )
    ):
        raise RuntimeError("GDN candidate source gate mismatch")
    compile(qwen, str(qwen_path), "exec")
    for name in (
        "round14_c_gdn_component_probe.py",
        "round14_c_gdn_component_controller.py",
        "monitor_owned_server.py",
        "numa_policy.py",
    ):
        path = root / name
        compile(path.read_text(), str(path), "exec")
    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files

        overlay = root / "overlay-python"
        receipt = json.loads((overlay / "round9-candidate-overlay.json").read_text())
        candidate = receipt.get("round14_c_gdn_candidate", {})
        if (
            receipt.get("candidate_tree_sha256") != tree_digest(tree_files(overlay))
            or candidate.get("status") != "INSTALLED_DEFAULT_OFF"
            or candidate.get("default") != "0"
            or candidate.get("qwen_sha256") != sha256(qwen_path)
        ):
            raise RuntimeError("component overlay receipt mismatch")
    finally:
        sys.path.remove(str(tools))
    result = {
        "schema": 1,
        "status": "PASS_ROUND14_C_GDN_COMPONENT_OFFLINE_PREFLIGHT",
        "version": version,
        "block": block,
        "physical_devices_exposed": devices,
        "manifest_sha256": sha256(manifest_path),
        "overlay_tree_sha256": receipt["candidate_tree_sha256"],
        "qwen_sha256": sha256(qwen_path),
        "candidate_default": "OFF",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], version, block, flush=True)


if __name__ == "__main__":
    main()
