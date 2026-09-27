#!/usr/bin/env python3
"""Run exactly one already-frozen Round13 A model arm and retire its cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


BASE = Path("/mnt/data1/kt-round9")
MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
MODEL_ROOT = Path("/home/admin/models/Qwen3.8-Flash-Next") / MODEL_REVISION
ADMISSION_TOOL = BASE / "round13-a-component-stage-v3/a-component-transfer-v3/round13_admission.py"
CACHE_HOOK_TEMPLATE = BASE / "round13-cache-retirements/v206/kt_cache_retirement_hook.py"
CACHE_MANIFEST_TEMPLATE = BASE / "round13-cache-retirements/v206/manifest.json"
ARMS = {
    207: ("uniform", "A", "uniform-a"),
    208: ("frequency", "A", "frequency-a"),
    209: ("front_loading", "A", "front-a"),
    210: ("random_seed42", "A", "random-a"),
    211: ("random_seed42", "B", "random-b"),
    212: ("front_loading", "B", "front-b"),
    213: ("frequency", "B", "frequency-b"),
    214: ("uniform", "B", "uniform-b"),
    223: ("front_loading", "A", "front-a-r1"),
    224: ("random_seed42", "A", "random-a-r1"),
    225: ("front_loading", "A", "front-a"),
    226: ("random_seed42", "A", "random-a"),
}


def command(values, **kwargs):
    print("ROUND13_ARM_COMMAND", values[0], flush=True)
    return subprocess.run(values, check=True, **kwargs)


def hash_evidence(root: Path):
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}")
    (root / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n")


def require_empty_physical12():
    raw = subprocess.check_output(
        ["npu-smi", "info", "-t", "proc-mem", "-i", "6", "-c", "0"], text=True)
    if "No process in device" not in raw:
        raise RuntimeError("physical12 is not empty after owned cleanup")


def valid_cache_retirement(result):
    if (
        result.get("audit_complete") is not True
        or result.get("stat_identity_unchanged") is not True
        or result.get("active_references")
    ):
        return False
    return (
        result.get("status") == "PASS_CACHE_RETIRED"
        and result.get("cache_retirement_performed") is True
    ) or (
        result.get("status") == "NOOP_ZERO_RESIDENCY"
        and result.get("cache_retirement_performed") is False
    )


def strict_result(out: Path, version: int, strategy: str, block: str):
    final = json.loads((out / "round13-a-final.json").read_text())
    supervisor = json.loads((out / "round13-model-supervisor.json").read_text())
    physical = json.loads((out / "q32-physical.json").read_text())
    expected_abort = 1 if block == "A" else 0
    if not (
        final.get("status") == "PASS_ROUND13_A_MODEL_PROCESS"
        and final.get("version") == version
        and final.get("strategy") == strategy
        and final.get("block") == block
        and final.get("generation_attempts") == 24
        and final.get("abort_controls") == expected_abort
        and final.get("automatic_retries") == 0
        and set(final.get("measurement_windows", {})) == {"W0", "W1", "W2", "W3", "W4"}
        and supervisor.get("status") == "PASS_ROUND13_A_MODEL_PROCESS"
        and supervisor.get("guard_exit_ack")
        and supervisor.get("owned_container_removed") is True
        and supervisor.get("physical12_after") == "EMPTY"
        and physical.get("status") == "PASS_ROUND13_STATIC_PLACEMENT_PHYSICAL"
        and physical.get("strategy") == strategy
        and physical.get("total_experts") == 1536
        and physical.get("total_bytes") == 15099494400
    ):
        raise RuntimeError("Round13 A strict process result failed")
    return final


def archive_and_retire(out: Path, version: int, label: str, passed: bool):
    require_empty_physical12()
    hash_evidence(out)
    archives = BASE / "round13-archives"
    archives.mkdir(exist_ok=True)
    archive = archives / f"v{version}-a-{label}-{'pass' if passed else 'fail'}.tar.gz"
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(out, arcname=out.name)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n")

    cache = BASE / f"round13-cache-retirements/v{version}"
    cache.mkdir()
    shutil.copy2(CACHE_HOOK_TEMPLATE, cache / CACHE_HOOK_TEMPLATE.name)
    manifest = json.loads(CACHE_MANIFEST_TEMPLATE.read_text())
    manifest["created_at"] = time.time()
    manifest["experiment_block"] = (
        f"ROUND13_A_V{version}_{label.upper().replace('-', '_')}_"
        f"{'PASS' if passed else 'FAIL'}_ARCHIVED")
    names = manifest["project_activity"]["container_names"]
    container = f"kt-round13-a-{label}-v{version}"
    if container not in names:
        names.append(container)
    manifest["project_containers_running"] = 0
    manifest["project_processes_running"] = 0
    manifest["hash_verification"]["experiment_evidence_sha256"] = hashlib.sha256(
        (out / "SHA256SUMS.txt").read_bytes()).hexdigest()
    manifest["hash_verification"]["round13_evidence_path"] = str(out)
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    result_path = cache / "result.json"
    command([
        "python3", str(cache / CACHE_HOOK_TEMPLATE.name),
        "--manifest", str(manifest_path), "--output", str(result_path),
        "--expected-checkpoint-root", str(MODEL_ROOT),
        "--expected-checkpoint-identity", MODEL_REVISION,
    ])
    result = json.loads(result_path.read_text())
    if not valid_cache_retirement(result):
        raise RuntimeError("Round13 targeted cache retirement failed")
    with (cache / "SHA256SUMS.txt").open("w") as stream:
        for path in (manifest_path, result_path, cache / CACHE_HOOK_TEMPLATE.name):
            stream.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    return {"archive": str(archive), "cache": str(result_path)}


def run(version: int):
    strategy, block, label = ARMS[version]
    snapshot = BASE / f"round13-a-{label}-v{version}"
    contract = json.loads((snapshot / "ROUND13_MODEL_CONTRACT.json").read_text())
    if (contract.get("version"), contract.get("strategy"), contract.get("block")) != (
            version, strategy, block):
        raise RuntimeError("Round13 arm snapshot identity mismatch")
    admission = BASE / f"evidence/20260927-round13-a-v{version}-admission"
    admission.mkdir()
    shutil.copy2(ADMISSION_TOOL, admission / ADMISSION_TOOL.name)
    command(["python3", str(admission / ADMISSION_TOOL.name), "sample",
             "--output", str(admission / "discovery.json")])
    command(["python3", str(admission / ADMISSION_TOOL.name), "sample",
             "--output", str(admission / "validation.json")])
    command(["python3", str(admission / ADMISSION_TOOL.name), "validate",
             "--discovery", str(admission / "discovery.json"),
             "--validation", str(admission / "validation.json"),
             "--output", str(admission / "result.json")])
    hash_evidence(admission)
    out = BASE / "evidence" / contract["evidence"]
    passed = False
    error = None
    try:
        command(["python3", str(snapshot / "round13_model_controller.py"),
                 "--admission", str(admission / "result.json")])
        strict_result(out, version, strategy, block)
        passed = True
    except BaseException as caught:
        error = caught
    receipt = archive_and_retire(out, version, label, passed)
    if error is not None:
        raise error
    print("PASS_ROUND13_A_ARM_ARCHIVED_RETIRED", version, strategy, block,
          json.dumps(receipt, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=tuple(ARMS), required=True)
    run(parser.parse_args().version)


if __name__ == "__main__":
    main()
