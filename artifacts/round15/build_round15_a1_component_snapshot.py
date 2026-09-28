#!/usr/bin/env python3
"""Build one immutable Round15 QSA index-selection component snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROLES = {339: "A", 340: "B"}
STAGE = "ROUND15_A1_QSA_INDEX_SELECTION_COMPONENT"
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


def contract(version):
    block = ROLES[version]
    lower = block.lower()
    return {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": STAGE,
        "version": version,
        "block": block,
        "container": f"kt-round15-a1-qsa-{lower}-v{version}",
        "evidence": f"20260928-round15-a1-qsa-{lower}-v{version}",
        "parent": "ROUND15_P0_V338_PROMOTED_S0",
        "candidate_family": "QSA_INDEX_SELECTION",
        "implementations": [
            "TORCH_NPU_LIGHTNING_INDEXER_DECODE_BSND",
            "FULL_SET_CAUSAL_SHORT_CONTEXT",
        ],
        "candidate_default": "OFF",
        "request_gate": "DECODE_W2",
        "qualified_effective_context_tokens": 256,
        "token_budget": 2048,
        "ratio": 4,
        "topk_blocks": 512,
        "graph": False,
        "same_shape_cycles": 20,
        "legal_growth_bytes": 65536,
        "full_parent_improvement_gate": 0.15,
        "downstream_relative_l2_gate": 0.001,
        "component_process_limit": 1,
        "automatic_retries": 0,
        "allocator_peak_limit_mib": 10240,
        "result": "result.json",
        "process_pass_status": "PASS_ROUND15_A1_QSA_COMPONENT_PROCESS",
        "controller_pass_status": "PASS_ROUND15_A1_QSA_COMPONENT_CONTROLLER",
    }


def patch_guard(source):
    marker = ")', name) is not None"
    index = source.rfind(marker)
    if index < 0:
        raise RuntimeError("Round15 A1 guard allowlist precondition mismatch")
    return (
        source[:index]
        + "|kt-round15-a1-qsa-[ab]-v(?:339|340)"
        + source[index:]
    )


def build(*, parent, output, version, scripts_root, control, monitor, numa, identity):
    parent = Path(parent).resolve(strict=True)
    output = Path(output)
    scripts_root = Path(scripts_root).resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    value = contract(version)
    runtime_qsa = parent / "overlay-python/sglang/srt/layers/round9/qsa.py"
    actual_parent_sha = sha256(runtime_qsa)
    # The constant is updated only after the frozen parent is inspected.  The
    # build receipt records both values so an accidental source drift is never
    # silently accepted.
    if actual_parent_sha != PARENT_QSA_SHA256:
        raise RuntimeError(
            f"Round15 A1 parent QSA SHA mismatch: {actual_parent_sha}"
        )
    output.mkdir(parents=True)
    copies = {
        scripts_root / "round15_a_qsa_candidate.py": output / "round15_a_qsa_candidate.py",
        scripts_root / "round15_a1_qsa_component_probe.py": output / "round15_a1_qsa_component_probe.py",
        scripts_root / "round15_a1_qsa_component_controller.py": output / "round15_a1_qsa_component_controller.py",
        scripts_root / "round15_a1_qsa_component_preflight.py": output / "round15_a1_qsa_component_preflight.py",
        Path(control): output / "round15_p0_control.py",
        Path(numa): output / "numa_policy.py",
        Path(identity): output / "probe_identity.py",
        runtime_qsa: output / "runtime_qsa.py",
    }
    for source, destination in copies.items():
        shutil.copy2(source.resolve(strict=True), destination)
    guard = patch_guard(Path(monitor).resolve(strict=True).read_text())
    compile(guard, "monitor_owned_server.py", "exec")
    (output / "monitor_owned_server.py").write_text(guard)
    (output / "ROUND15_A1_QSA_COMPONENT_CONTRACT.json").write_text(
        json.dumps(value, indent=2) + "\n"
    )
    manifest = tree(output)
    (output / "source-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "schema": 1,
        "status": "PASS_ROUND15_A1_QSA_COMPONENT_BUILD",
        "version": version,
        "block": value["block"],
        "output": str(output),
        "parent_qsa_sha256": actual_parent_sha,
        "manifest_sha256": sha256(output / "source-sha256.json"),
        "files": len(manifest),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=sorted(ROLES), required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--numa", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        parent=args.parent,
        output=args.output,
        version=args.version,
        scripts_root=args.scripts_root,
        control=args.control,
        monitor=args.monitor,
        numa=args.numa,
        identity=args.identity,
    )
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["version"], result["block"], flush=True)


if __name__ == "__main__":
    main()
