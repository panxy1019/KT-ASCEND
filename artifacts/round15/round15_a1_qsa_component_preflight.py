#!/usr/bin/env python3
"""CPU-only same-image preflight for a Round15 QSA component snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform


ROLES = {339: "A", 340: "B"}
PARENT_QSA_SHA256 = "0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree(root):
    root = Path(root)
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "source-sha256.json"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    devices = sorted(str(path) for path in Path("/dev").glob("davinci[0-9]*"))
    if devices:
        raise RuntimeError(f"Round15 A1 preflight exposed NPU devices: {devices}")
    manifest_path = root / "source-sha256.json"
    if tree(root) != json.loads(manifest_path.read_text()):
        raise RuntimeError("Round15 A1 snapshot manifest mismatch")
    contract = json.loads((root / "ROUND15_A1_QSA_COMPONENT_CONTRACT.json").read_text())
    version, block = contract.get("version"), contract.get("block")
    if (
        ROLES.get(version) != block
        or contract.get("stage") != "ROUND15_A1_QSA_INDEX_SELECTION_COMPONENT"
        or contract.get("candidate_default") != "OFF"
        or contract.get("candidate_family") != "QSA_INDEX_SELECTION"
        or contract.get("graph") is not False
        or contract.get("same_shape_cycles") != 20
        or contract.get("automatic_retries") != 0
    ):
        raise RuntimeError("Round15 A1 component contract mismatch")
    if sha256(root / "runtime_qsa.py") != PARENT_QSA_SHA256:
        raise RuntimeError("Round15 A1 frozen eager QSA source drift")
    helper = (root / "round15_a_qsa_candidate.py").read_text()
    if (
        "FULL_SET_CAUSAL" not in helper
        or "LIGHTNING_INDEXER" not in helper
        or "EAGER_REFERENCE" not in helper
    ):
        raise RuntimeError("Round15 A1 strategy gate missing")
    for name in (
        "round15_a_qsa_candidate.py",
        "round15_a1_qsa_component_probe.py",
        "round15_a1_qsa_component_controller.py",
        "round15_a1_qsa_component_preflight.py",
        "round15_p0_control.py",
        "monitor_owned_server.py",
        "numa_policy.py",
        "probe_identity.py",
        "runtime_qsa.py",
    ):
        path = root / name
        compile(path.read_text(), str(path), "exec")

    import torch
    import torch_npu

    schema = str(torch.ops.npu.npu_lightning_indexer.default._schema)
    required = (
        "Tensor query",
        "Tensor key",
        "Tensor weights",
        "actual_seq_lengths_query",
        "actual_seq_lengths_key",
        "layout_query",
        "layout_key",
        "sparse_count",
        "sparse_mode",
    )
    if not all(fragment in schema for fragment in required):
        raise RuntimeError(f"frozen npu_lightning_indexer schema mismatch: {schema}")
    result = {
        "schema": 1,
        "status": "PASS_ROUND15_A1_QSA_COMPONENT_OFFLINE_PREFLIGHT",
        "version": version,
        "block": block,
        "physical_devices_exposed": devices,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "npu_lightning_indexer_schema": schema,
        "manifest_sha256": sha256(manifest_path),
        "runtime_qsa_sha256": sha256(root / "runtime_qsa.py"),
        "candidate_default": "OFF",
        "graph": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], version, block, flush=True)


if __name__ == "__main__":
    main()
