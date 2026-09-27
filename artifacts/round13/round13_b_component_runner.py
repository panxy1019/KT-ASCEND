#!/usr/bin/env python3
"""Run one frozen Round13 B component block, archive it, then retire cache."""

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
STAGE = BASE / "round13-b-component-stage-v1/b-component-transfer-v1"
BUNDLE = BASE / "round13-b-qsa-component-v1"
ADMISSION_TOOL = STAGE / "round13_admission.py"
CONTROLLER = STAGE / "round13_b_component_controller.py"
MONITOR = STAGE / "round13_monitor_owned.py"
NUMA_POLICY = STAGE / "numa_policy.py"
CACHE_HOOK_TEMPLATE = BASE / "round13-cache-retirements/v208/kt_cache_retirement_hook.py"
CACHE_MANIFEST_TEMPLATE = BASE / "round13-cache-retirements/v208/manifest.json"
VERSIONS = {215: "A", 216: "B"}


def command(values, **kwargs):
    print("ROUND13_B_COMPONENT_COMMAND", values[0], flush=True)
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
        raise RuntimeError("physical12 is not empty after B component cleanup")


def strict_result(out, version, block):
    result = json.loads((out / "result.json").read_text())
    controller = json.loads((out / "controller.json").read_text())
    if not (
        result.get("status") == "PASS_ROUND13_B_QSA_DECODE_COMPONENT"
        and result.get("block") == block
        and result.get("row_count") == 9
        and result.get("automatic_retries") == 0
        and result.get("after_allocated") == 0
        and result.get("after_reserved") == 0
        and controller.get("status") == "PASS_ROUND13_B_COMPONENT_CONTROLLER"
        and controller.get("version") == version
        and controller.get("block") == block
        and controller.get("guard_exit_ack")
        and controller.get("owned_container_removed") is True
        and controller.get("physical12_after") == "EMPTY"
    ):
        raise RuntimeError("Round13 B strict component result failed")
    return result


def archive_and_retire(out, version, block, passed):
    require_empty_physical12()
    hash_evidence(out)
    archives = BASE / "round13-archives"
    archives.mkdir(exist_ok=True)
    archive = archives / (
        f"v{version}-b-component-{block.lower()}-"
        f"{'pass' if passed else 'fail'}.tar.gz"
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
        f"ROUND13_B_COMPONENT_V{version}_{block}_"
        f"{'PASS' if passed else 'FAIL'}_ARCHIVED"
    )
    name = f"kt-round13-b-profile-{block.lower()}-v{version}"
    if name not in manifest["project_activity"]["container_names"]:
        manifest["project_activity"]["container_names"].append(name)
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
    if not (
        result.get("status") == "PASS_CACHE_RETIRED"
        and result.get("cache_retirement_performed") is True
        and result.get("audit_complete") is True
        and not result.get("active_references")
    ):
        raise RuntimeError("Round13 B targeted cache retirement failed")
    with (cache / "SHA256SUMS.txt").open("w") as stream:
        for path in (manifest_path, result_path, cache / CACHE_HOOK_TEMPLATE.name):
            stream.write(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            )
    return {"archive": str(archive), "cache": str(result_path)}


def run(version):
    block = VERSIONS[version]
    admission = BASE / f"evidence/20260927-round13-b-v{version}-admission"
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
    out = BASE / f"evidence/20260927-round13-b-component-{block.lower()}-v{version}"
    passed = False
    error = None
    try:
        command(
            [
                "python3",
                str(CONTROLLER),
                "--version",
                str(version),
                "--admission",
                str(admission / "result.json"),
                "--candidate",
                str(BUNDLE),
                "--monitor",
                str(MONITOR),
                "--numa-policy",
                str(NUMA_POLICY),
                "--output",
                str(out),
            ]
        )
        strict_result(out, version, block)
        passed = True
    except BaseException as caught:
        error = caught
    receipt = archive_and_retire(out, version, block, passed)
    if error is not None:
        raise error
    print(
        "PASS_ROUND13_B_COMPONENT_ARCHIVED_RETIRED",
        version,
        block,
        json.dumps(receipt, sort_keys=True),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=tuple(VERSIONS), required=True)
    run(parser.parse_args().version)


if __name__ == "__main__":
    main()
