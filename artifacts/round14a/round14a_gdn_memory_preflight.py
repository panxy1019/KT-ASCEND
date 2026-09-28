#!/usr/bin/env python3
"""No-device same-image gate for the frozen Round14a GDN diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


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
        raise RuntimeError(f"Round14a preflight exposed NPU devices: {devices}")
    manifest_path = root / "source-sha256.json"
    if tree(root) != json.loads(manifest_path.read_text()):
        raise RuntimeError("Round14a GDN diagnostic manifest mismatch")
    contract = json.loads((root / "ROUND14A_A_GDN_MEMORY_CONTRACT.json").read_text())
    if (
        (contract.get("version"), contract.get("block")) != (289, "DIAGNOSTIC")
        or contract.get("stage") != "ROUND14A_A_GDN_MEMORY_ROOT_CAUSE"
        or contract.get("candidate_default") != "OFF"
        or contract.get("empty_cache_allowed") is not False
        or contract.get("same_shape_cycles") != 20
    ):
        raise RuntimeError("Round14a GDN diagnostic contract mismatch")
    qwen_path = root / "overlay-python/sglang/srt/models/qwen4_exp.py"
    qwen = qwen_path.read_text()
    if (
        qwen.count("ROUND14_GDN_BACKEND_ENABLE") != 1
        or qwen.count("sigmoid_gated_rms_norm(") != 1
        or 'os.environ.get("ROUND14_GDN_BACKEND_ENABLE", "0") == "1"' not in qwen
    ):
        raise RuntimeError("Round14a GDN candidate source gate mismatch")
    for name in (
        "round14a_gdn_memory_probe.py",
        "gdn_memory_snapshot.py",
        "round14a_gdn_memory_controller.py",
        "monitor_owned_server.py",
        "numa_policy.py",
    ):
        path = root / name
        compile(path.read_text(), str(path), "exec")
    if "torch.npu.empty_cache()" in (root / "round14a_gdn_memory_probe.py").read_text():
        raise RuntimeError("Round14a diagnostic may not mask allocation growth")
    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files

        overlay = root / "overlay-python"
        receipt = json.loads((overlay / "round9-candidate-overlay.json").read_text())
        if receipt.get("candidate_tree_sha256") != tree_digest(tree_files(overlay)):
            raise RuntimeError("Round14a parent overlay receipt mismatch")
    finally:
        sys.path.remove(str(tools))
    result = {
        "schema": 1,
        "status": "PASS_ROUND14A_A_GDN_MEMORY_OFFLINE_PREFLIGHT",
        "version": 289,
        "physical_devices_exposed": devices,
        "manifest_sha256": sha256(manifest_path),
        "qwen_sha256": sha256(qwen_path),
        "candidate_default": "OFF",
        "empty_cache_calls": 0,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
