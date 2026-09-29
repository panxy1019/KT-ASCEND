#!/usr/bin/env python3
"""Derive one default-off Round17 attribution snapshot from frozen v509."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new)


def patch_launcher(
    source: str,
    *,
    version: int,
    block: str,
    snapshot_name: str,
    evidence_name: str,
    diagnostic_observer_sha256: str,
    baseline_observer_sha256: str,
) -> str:
    if block not in {"A", "B"}:
        raise ValueError("Round17 profile block must be A or B")
    for value in (diagnostic_observer_sha256, baseline_observer_sha256):
        if len(value) != 64:
            raise ValueError("observer SHA-256 is malformed")
    source = replace_once(
        source,
        "if a.version != 509 or a.cost_mode != 'OFF':",
        f"if a.version != {version} or a.cost_mode != 'OFF':",
        "version gate",
    )
    source = replace_once(
        source,
        "name='kt-root-cause-full-host-a-v509';out=root/'evidence'/'20260929-kt-root-cause-full-host-a-v509'",
        f"name='{snapshot_name}';out=root/'evidence'/'{evidence_name}'",
        "run identity",
    )
    source = replace_once(
        source,
        "KT_ROOT_TRACE_BINDING_PATH=/round9/kt-root-cause-full-host-a-v509/TRACE_NATIVE_BINDING.json",
        f"KT_ROOT_TRACE_BINDING_PATH=/round9/{snapshot_name}/TRACE_NATIVE_BINDING.json",
        "trace binding path",
    )
    source = replace_once(
        source,
        "ROUND16_A1_BLOCK_ID=round16-a1-fresh-profile-a-v001",
        f"ROUND16_A1_BLOCK_ID=round16-a1-fresh-profile-{block.lower()}-v001",
        "runtime block identity",
    )
    source = replace_once(
        source,
        "if candidate_files.get(_observer_relative)!='ec9e4c88a00251ae635dca9cee3eca4dbec168ddb6df2cfb4f01babe833ea49f':",
        f"if candidate_files.get(_observer_relative)!='{diagnostic_observer_sha256}':",
        "diagnostic observer binding",
    )
    return replace_once(
        source,
        "candidate_files[_observer_relative]='a4bd6c90bc05ec09e51d14194fb2eaf38e00dcb5578d48b91bfdfc1f4b392cc0'",
        f"candidate_files[_observer_relative]='{baseline_observer_sha256}'",
        "baseline observer normalization",
    )


def patch_block(value: dict, *, block: str, version: int) -> dict:
    if block not in {"A", "B"}:
        raise ValueError("Round17 profile block must be A or B")
    result = dict(value)
    lower = block.lower()
    result["block"] = block
    result["block_id"] = f"round16-a1-fresh-profile-{lower}-v001"
    result["snapshot_id"] = f"round17-profile-{lower}-v{version}-snapshot"
    result["round17_version"] = version
    result["round17_stage"] = "P1_COMPLETE_PARENT_AND_GLUE_PROFILE"
    result["enabled"] = False
    result["execution"] = "DEVICE_REQUIRED_NOT_RUN"
    return result


def patch_monitor(source: str, *, snapshot_name: str) -> str:
    return replace_once(
        source,
        "|kt-root-cause-full-host-a-v509)', name) is not None",
        f"|kt-root-cause-full-host-a-v509|{snapshot_name})', name) is not None",
        "owned container allow-list",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    args = parser.parse_args()
    expected = f"round17-profile-{args.block.lower()}-v{args.version}"
    if args.output.name != expected:
        raise ValueError(f"output name must be {expected}")
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("Round17 snapshot and receipt must be fresh")
    shutil.copytree(args.donor, args.output, symlinks=True)
    destination = (
        args.output
        / "overlay-python/sglang/srt/layers/round9/round13_attribution.py"
    )
    baseline_sha = "a4bd6c90bc05ec09e51d14194fb2eaf38e00dcb5578d48b91bfdfc1f4b392cc0"
    if sha256(destination) != "ec9e4c88a00251ae635dca9cee3eca4dbec168ddb6df2cfb4f01babe833ea49f":
        raise RuntimeError("donor is not frozen v509")
    shutil.copyfile(args.observer, destination)
    diagnostic_sha = sha256(destination)
    launcher = args.output / "source/tools/launch_hybrid.py"
    monitor = args.output / "source/tools/monitor_owned_server.py"
    evidence_name = f"20260929-{expected}"
    launcher.write_text(
        patch_launcher(
            launcher.read_text(),
            version=args.version,
            block=args.block,
            snapshot_name=expected,
            evidence_name=evidence_name,
            diagnostic_observer_sha256=diagnostic_sha,
            baseline_observer_sha256=baseline_sha,
        )
    )
    monitor.write_text(patch_monitor(monitor.read_text(), snapshot_name=expected))
    block_path = args.output / "round16-a1-block.json"
    block_path.write_text(
        json.dumps(
            patch_block(
                json.loads(block_path.read_text()),
                block=args.block,
                version=args.version,
            ),
            indent=2,
        )
        + "\n"
    )
    budget_path = args.output / "execution-budget.json"
    budget = json.loads(budget_path.read_text())
    budget["model_starts_used"] = 0
    budget["starts"] = []
    budget_path.write_text(json.dumps(budget, indent=2) + "\n")
    for relative in ("execution-budget.lock", "host-controller.json"):
        path = args.output / relative
        if path.exists():
            path.unlink()
    for path in (destination, launcher, monitor):
        compile(path.read_text(), str(path), "exec")
    receipt = {
        "schema": 1,
        "status": "PASS_ROUND17_PROFILE_SNAPSHOT",
        "created_at": time.time(),
        "donor": str(args.donor),
        "output": str(args.output),
        "version": args.version,
        "block": args.block,
        "observer_sha256": diagnostic_sha,
        "baseline_observer_sha256": baseline_sha,
        "launcher_sha256": sha256(launcher),
        "monitor_sha256": sha256(monitor),
        "block_contract_sha256": sha256(block_path),
        "math_changes": 0,
        "synchronization_changes": 0,
        "default_enabled": False,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt["status"])


if __name__ == "__main__":
    main()
