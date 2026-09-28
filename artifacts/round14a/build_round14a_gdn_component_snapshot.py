#!/usr/bin/env python3
"""Freeze immutable Round14a production-equivalent GDN component A/B snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROLES = {290: "A", 291: "B"}
MUTABLE = {"source-sha256.json", "host-controller.json"}
EXPECTED_PARENT_QWEN = "1986d8a880fe1503848eadf02a8ed572d543eac967fcb583b6f67d42250f3545"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in MUTABLE
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }


def contract(version: int) -> dict:
    block = ROLES.get(version)
    if block is None:
        raise ValueError("unsupported Round14a GDN component version")
    lower = block.lower()
    return {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": "ROUND14A_A_GDN_PRODUCTION_EQUIVALENT_COMPONENT",
        "version": version,
        "block": block,
        "container": f"kt-round14a-a-gdn-component-{lower}-v{version}",
        "evidence": f"20260927-round14a-a-gdn-component-{lower}-v{version}",
        "parent": "ROUND14A_A_GDN_MEMORY_DIAGNOSTIC_V289",
        "candidate": "NPU_GDN_CORE_PLUS_FROZEN_QWEN4_SIGMOID_GATE",
        "candidate_default": "OFF",
        "production_execution_mode": "torch.inference_mode",
        "empty_cache_allowed": False,
        "same_shape_cycles": 20,
        "legal_global_cache_limit_bytes": 65536,
        "component_process_limit": 1,
        "automatic_retries": 0,
        "relative_l2_gate": 0.03,
        "full_parent_improvement_gate": 0.15,
        "allocator_peak_limit_mib": 10240,
        "result": "result.json",
        "result_pass_status": "PASS_ROUND14A_A_GDN_COMPONENT_PROCESS",
        "controller_pass_status": "PASS_ROUND14A_A_GDN_COMPONENT_CONTROLLER",
    }


def patch_monitor(source: str) -> str:
    old = "r'|kt-round14a-a-gdn-memory-v289)', name) is not None"
    new = (
        "r'|kt-round14a-a-gdn-memory-v289'\n"
        "        r'|kt-round14a-a-gdn-component-[ab]-v(?:290|291))', name) is not None"
    )
    if source.count(old) != 1:
        raise RuntimeError("Round14a component monitor source precondition mismatch")
    patched = source.replace(old, new)
    compile(patched, "monitor_owned_server.py", "exec")
    return patched


def build(parent: Path, output: Path, version: int, probe: Path, controller: Path, preflight: Path):
    parent = parent.resolve(strict=True)
    probe = probe.resolve(strict=True)
    controller = controller.resolve(strict=True)
    preflight = preflight.resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    parent_contract = json.loads(
        (parent / "ROUND14A_A_GDN_MEMORY_CONTRACT.json").read_text()
    )
    if (parent_contract.get("version"), parent_contract.get("block")) != (289, "DIAGNOSTIC"):
        raise RuntimeError("Round14a component requires exact v289 diagnostic parent")
    qwen = parent / "overlay-python/sglang/srt/models/qwen4_exp.py"
    if sha256(qwen) != EXPECTED_PARENT_QWEN:
        raise RuntimeError("Round14a component parent qwen source drift")
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    for name in ("source-sha256.json", "host-controller.json", "execution-budget.lock"):
        (output / name).unlink(missing_ok=True)
    shutil.copy2(probe, output / probe.name)
    shutil.copy2(controller, output / controller.name)
    shutil.copy2(preflight, output / preflight.name)
    monitor = output / "monitor_owned_server.py"
    monitor.write_text(patch_monitor(monitor.read_text()))
    value = contract(version)
    (output / "ROUND14A_A_GDN_COMPONENT_CONTRACT.json").write_text(
        json.dumps(value, indent=2) + "\n"
    )
    for path in (output / probe.name, output / controller.name, output / preflight.name, monitor):
        compile(path.read_text(), str(path), "exec")
    manifest = tree(output)
    (output / "source-sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if tree(output) != manifest:
        raise RuntimeError("Round14a component snapshot manifest mismatch")
    return {
        "schema": 1,
        "status": "PASS_ROUND14A_A_GDN_COMPONENT_SNAPSHOT_BUILD",
        "version": version,
        "block": value["block"],
        "output": str(output.resolve()),
        "manifest_sha256": sha256(output / "source-sha256.json"),
        "qwen_sha256": sha256(output / "overlay-python/sglang/srt/models/qwen4_exp.py"),
        "probe_sha256": sha256(output / probe.name),
        "controller_sha256": sha256(output / controller.name),
        "preflight_sha256": sha256(output / preflight.name),
        "immutable_files": len(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=tuple(ROLES), required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.parent, args.output, args.version, args.probe, args.controller, args.preflight)
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["version"], report["block"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
