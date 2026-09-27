#!/usr/bin/env python3
"""Run one frozen Round13 F final arm, archive it, then retire cache."""

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
ADMISSION_TOOL = BASE / "round13-b-component-stage-v1/b-component-transfer-v1/round13_admission.py"
CACHE_HOOK_TEMPLATE = BASE / "round13-cache-retirements/v208/kt_cache_retirement_hook.py"
CACHE_MANIFEST_TEMPLATE = BASE / "round13-cache-retirements/v208/manifest.json"
RUNS = {
    227: ("baseline", "A", "s0-a"),
    228: ("candidate", "A", "final-a"),
    229: ("candidate", "B", "final-b"),
    230: ("baseline", "B", "s0-b"),
}


def command(values, **kwargs):
    print("ROUND13_B_MODEL_COMMAND", values[0], flush=True)
    return subprocess.run(values, check=True, **kwargs)


def hash_evidence(root):
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
            )
    (root / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n")


def require_empty_physical12():
    raw = subprocess.check_output(
        ["npu-smi", "info", "-t", "proc-mem", "-i", "6", "-c", "0"],
        text=True,
    )
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


def strict_result(out, version, implementation, block):
    out = Path(out)
    final = json.loads((out / "round13-f-final.json").read_text())
    supervisor = json.loads((out / "round13-model-supervisor.json").read_text())
    process_status = "PASS_ROUND13_F_MODEL_PROCESS"
    expected_attempts = 24
    expected_aborts = 1 if block == "A" else 0
    windows_ok = set(final.get("measurement_windows", {})) == {"W0", "W1", "W2", "W3", "W4"}
    if not (
        final.get("status") == process_status
        and final.get("version") == version
        and final.get("implementation") == implementation
        and final.get("block") == block
        and final.get("generation_attempts") == expected_attempts
        and final.get("abort_controls") == expected_aborts
        and final.get("automatic_retries") == 0
        and windows_ok
        and supervisor.get("status") == process_status
        and supervisor.get("guard_exit_ack")
        and supervisor.get("owned_container_removed") is True
        and supervisor.get("physical12_after") == "EMPTY"
    ):
        raise RuntimeError("Round13 F strict process result failed")
    return final


def archive_and_retire(out, version, label, passed):
    require_empty_physical12()
    hash_evidence(out)
    archives = BASE / "round13-archives"
    archives.mkdir(exist_ok=True)
    archive = archives / (
        f"v{version}-f-{label}-{'pass' if passed else 'fail'}.tar.gz"
    )
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(out, arcname=out.name)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n"
    )

    cache = BASE / f"round13-cache-retirements/v{version}"
    cache.mkdir()
    shutil.copy2(CACHE_HOOK_TEMPLATE, cache / CACHE_HOOK_TEMPLATE.name)
    manifest = json.loads(CACHE_MANIFEST_TEMPLATE.read_text())
    manifest["created_at"] = time.time()
    manifest["experiment_block"] = (
        f"ROUND13_F_MODEL_V{version}_{label.upper().replace('-', '_')}_"
        f"{'PASS' if passed else 'FAIL'}_ARCHIVED"
    )
    container = f"kt-round13-f-{label}-v{version}"
    if container not in manifest["project_activity"]["container_names"]:
        manifest["project_activity"]["container_names"].append(container)
    manifest["project_containers_running"] = 0
    manifest["project_processes_running"] = 0
    manifest["hash_verification"]["experiment_evidence_sha256"] = hashlib.sha256(
        (out / "SHA256SUMS.txt").read_bytes()
    ).hexdigest()
    manifest["hash_verification"]["round13_evidence_path"] = str(out)
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    result_path = cache / "result.json"
    command(
        [
            "python3",
            str(cache / CACHE_HOOK_TEMPLATE.name),
            "--manifest",
            str(manifest_path),
            "--output",
            str(result_path),
            "--expected-checkpoint-root",
            str(MODEL_ROOT),
            "--expected-checkpoint-identity",
            MODEL_REVISION,
        ]
    )
    result = json.loads(result_path.read_text())
    if not valid_cache_retirement(result):
        raise RuntimeError("Round13 F targeted cache retirement failed")
    with (cache / "SHA256SUMS.txt").open("w") as stream:
        for path in (manifest_path, result_path, cache / CACHE_HOOK_TEMPLATE.name):
            stream.write(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            )
    return {"archive": str(archive), "cache": str(result_path)}


def run(version):
    implementation, block, label = RUNS[version]
    snapshot = BASE / f"round13-f-{label}-v{version}"
    contract = json.loads((snapshot / "ROUND13_MODEL_CONTRACT.json").read_text())
    if (
        contract.get("version"),
        contract.get("implementation"),
        contract.get("block"),
    ) != (version, implementation, block):
        raise RuntimeError("Round13 F snapshot identity mismatch")
    admission = BASE / f"evidence/20260927-round13-f-v{version}-admission"
    admission.mkdir()
    shutil.copy2(ADMISSION_TOOL, admission / ADMISSION_TOOL.name)
    command(
        [
            "python3",
            str(admission / ADMISSION_TOOL.name),
            "sample",
            "--output",
            str(admission / "discovery.json"),
        ]
    )
    command(
        [
            "python3",
            str(admission / ADMISSION_TOOL.name),
            "sample",
            "--output",
            str(admission / "validation.json"),
        ]
    )
    command(
        [
            "python3",
            str(admission / ADMISSION_TOOL.name),
            "validate",
            "--discovery",
            str(admission / "discovery.json"),
            "--validation",
            str(admission / "validation.json"),
            "--output",
            str(admission / "result.json"),
        ]
    )
    hash_evidence(admission)
    out = BASE / "evidence" / contract["evidence"]
    passed = False
    error = None
    try:
        command(
            [
                "python3",
                str(snapshot / "round13_model_controller.py"),
                "--admission",
                str(admission / "result.json"),
            ]
        )
        strict_result(out, version, implementation, block)
        passed = True
    except BaseException as caught:
        error = caught
    receipt = archive_and_retire(out, version, label, passed)
    if error is not None:
        raise error
    print(
        "PASS_ROUND13_F_MODEL_ARCHIVED_RETIRED",
        version,
        implementation,
        block,
        json.dumps(receipt, sort_keys=True),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=tuple(RUNS), required=True)
    run(parser.parse_args().version)


if __name__ == "__main__":
    main()
