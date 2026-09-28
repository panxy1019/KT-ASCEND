#!/usr/bin/env python3
"""Build immutable v357 Round15 resident-MoE segment graph component snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


VERSION = 357
PARENT_MANIFEST_SHA256 = "ffeaefddd1d3bff79e815e6cd0147ffeca92f12e03b5cee5ab5dd233bb3cc0cc"
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


def patch_guard(source):
    marker = ")', name) is not None"
    index = source.rfind(marker)
    if index < 0:
        raise RuntimeError("Round15 B guard allowlist precondition mismatch")
    return source[:index] + "|kt-round15-b-graph-component-v357" + source[index:]


def contract():
    return {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": "ROUND15_B_RESIDENT_MOE_SEGMENT_GRAPH_COMPONENT",
        "version": VERSION,
        "container": "kt-round15-b-graph-component-v357",
        "evidence": "20260928-round15-b-graph-component-v357",
        "parent": "ROUND15_A_V341_BASELINE_S0_SOURCE_IDENTITY",
        "parent_snapshot": "round15/a-model-snapshots/v342",
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "candidate": "DECODE_BS1_RESIDENT_MOE_NPU_SEGMENT_GRAPH_TOPK_IDS_INT32",
        "candidate_default": "OFF",
        "graph_boundary": "RESIDENT_NPU_MOE_ONLY",
        "request_evidence_class": "DIRECTIONAL_ONE_BASELINE_ONE_CANDIDATE_IF_COMPONENT_PASSES",
        "capture_geometry": {
            "tokens": 1,
            "hidden": 2560,
            "topk": 10,
            "topk_ids_dtype": "torch.int32",
            "resident_experts": 32,
            "tp": 1,
            "dtype": "BF16_NZ29",
            "return_contribution": "FP32",
        },
        "same_shape_cycles": 20,
        "timing_samples": 15,
        "component_gain_gate": 0.10,
        "relative_l2_gate": 1e-5,
        "post_reset_growth_limit_bytes": 64 * 1024**2,
        "component_process_limit": 1,
        "automatic_retries": 0,
        "allocator_peak_limit_mib": 10240,
        "result": "result.json",
        "process_pass_status": "PASS_ROUND15_B_GRAPH_COMPONENT_PROCESS",
        "controller_pass_status": "PASS_ROUND15_B_GRAPH_COMPONENT_CONTROLLER",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--numa", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    parent = args.parent.resolve(strict=True)
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    if sha256(parent / "round15-a-source-sha256.json") != PARENT_MANIFEST_SHA256:
        raise RuntimeError("Round15 B parent manifest mismatch")
    unquant = parent / "overlay-python/sglang/srt/layers/quantization/unquant.py"
    ktep = parent / "overlay-python/sglang/srt/layers/moe/kt_ep_wrapper.py"
    if sha256(unquant) != UNQUANT_SHA256 or sha256(ktep) != KTEP_SHA256:
        raise RuntimeError("Round15 B parent provider source mismatch")
    output.mkdir(parents=True)
    root = args.scripts_root.resolve(strict=True)
    copies = {
        root / "round15_b_segment_policy.py": output / "round15_b_segment_policy.py",
        root / "round15_b_resident_graph.py": output / "round15_b_resident_graph.py",
        root / "round15_b_graph_component_probe.py": output / "round15_b_graph_component_probe.py",
        root / "round15_b_graph_component_controller.py": output / "round15_b_graph_component_controller.py",
        root / "round15_b_graph_component_preflight.py": output / "round15_b_graph_component_preflight.py",
        Path(args.control): output / "round15_p0_control.py",
        Path(args.numa): output / "numa_policy.py",
        Path(args.identity): output / "probe_identity.py",
        unquant: output / "runtime_unquant.py",
        ktep: output / "runtime_kt_ep_wrapper.py",
    }
    for source, destination in copies.items():
        shutil.copy2(source.resolve(strict=True), destination)
    monitor = patch_guard(Path(args.monitor).resolve(strict=True).read_text())
    compile(monitor, "monitor_owned_server.py", "exec")
    (output / "monitor_owned_server.py").write_text(monitor)
    (output / "ROUND15_B_GRAPH_COMPONENT_CONTRACT.json").write_text(
        json.dumps(contract(), indent=2) + "\n"
    )
    manifest = tree(output)
    (output / "source-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = {
        "schema": 1,
        "status": "PASS_ROUND15_B_GRAPH_COMPONENT_BUILD",
        "version": VERSION,
        "output": str(output),
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "unquant_sha256": UNQUANT_SHA256,
        "ktep_sha256": KTEP_SHA256,
        "manifest_sha256": sha256(output / "source-sha256.json"),
        "files": len(manifest),
    }
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], VERSION, flush=True)


if __name__ == "__main__":
    main()
