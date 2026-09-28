#!/usr/bin/env python3
"""No-device same-image preflight for the Round15 B graph component."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform


UNQUANT_SHA256 = "56b46c170c8150afb437cb1bc0ea4453a456ca14f849170e7d0f27358cabcafb"
KTEP_SHA256 = "77dfbbab73431111e72b6ac2683e4f4ecfa617280436e9ce55a3d7f89177e46b"


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
        raise RuntimeError(f"Round15 B preflight exposed NPU devices: {devices}")
    if tree(root) != json.loads((root / "source-sha256.json").read_text()):
        raise RuntimeError("Round15 B component snapshot manifest mismatch")
    contract = json.loads((root / "ROUND15_B_GRAPH_COMPONENT_CONTRACT.json").read_text())
    if (
        contract.get("version") != 357
        or contract.get("stage") != "ROUND15_B_RESIDENT_MOE_SEGMENT_GRAPH_COMPONENT"
        or contract.get("candidate_default") != "OFF"
        or contract.get("graph_boundary") != "RESIDENT_NPU_MOE_ONLY"
        or contract.get("request_evidence_class")
        != "DIRECTIONAL_ONE_BASELINE_ONE_CANDIDATE_IF_COMPONENT_PASSES"
        or contract.get("automatic_retries") != 0
    ):
        raise RuntimeError("Round15 B component contract mismatch")
    if sha256(root / "runtime_unquant.py") != UNQUANT_SHA256:
        raise RuntimeError("Round15 B unquant provider source drift")
    if sha256(root / "runtime_kt_ep_wrapper.py") != KTEP_SHA256:
        raise RuntimeError("Round15 B KTEP source drift")
    for name in (
        "round15_b_segment_policy.py",
        "round15_b_resident_graph.py",
        "round15_b_graph_component_probe.py",
        "round15_b_graph_component_controller.py",
        "round15_b_graph_component_preflight.py",
        "round15_p0_control.py",
        "monitor_owned_server.py",
        "numa_policy.py",
        "probe_identity.py",
        "runtime_unquant.py",
        "runtime_kt_ep_wrapper.py",
    ):
        path = root / name
        compile(path.read_text(), str(path), "exec")

    import torch
    import torch_npu

    graph_signature = str(inspect.signature(torch.npu.graph))
    required_graph_parameters = ("npu_graph", "pool", "stream", "auto_dispatch_capture")
    if not all(name in graph_signature for name in required_graph_parameters):
        raise RuntimeError(f"Round15 B torch.npu.graph signature mismatch: {graph_signature}")
    missing_graph_methods = [
        name for name in ("replay", "reset", "pool") if not hasattr(torch.npu.NPUGraph, name)
    ]
    if missing_graph_methods:
        raise RuntimeError(f"Round15 B NPUGraph methods missing: {missing_graph_methods}")
    op_schemas = {}
    required_ops = (
        "npu_moe_init_routing_v2",
        "npu_grouped_matmul",
        "npu_swiglu",
        "npu_moe_finalize_routing",
    )
    for name in required_ops:
        packet = getattr(torch.ops.npu, name)
        if "default" not in packet.overloads():
            raise RuntimeError(f"Round15 B operator has no default overload: {name}")
        op_schemas[name] = str(packet.default._schema)
    result = {
        "schema": 1,
        "status": "PASS_ROUND15_B_GRAPH_COMPONENT_OFFLINE_PREFLIGHT",
        "version": 357,
        "physical_devices_exposed": devices,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "graph_signature": graph_signature,
        "npugraph_methods": ["replay", "reset", "pool"],
        "op_schemas": op_schemas,
        "manifest_sha256": sha256(root / "source-sha256.json"),
        "candidate_default": "OFF",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
