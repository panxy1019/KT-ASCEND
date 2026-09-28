#!/usr/bin/env python3
"""Freeze one Round14a B fresh-S0 five-bucket attribution snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROLES = {298: "A", 299: "B"}
STAGE = "ROUND14A_B_FRESH_S0_FIVE_BUCKET_PROFILE"
MUTABLE = {"round14a-source-sha256.json", "execution-budget.json",
           "execution-budget.lock", "host-controller.json"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise RuntimeError(f"Round14a B {label} precondition mismatch")
    return source.replace(old, new)


def tree(root):
    root = Path(root)
    return {str(path.relative_to(root)): sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
            and str(path.relative_to(root)) not in MUTABLE
            and not str(path.relative_to(root)).startswith("host-controller")}


def contract(version):
    block = ROLES[version]
    lower = block.lower()
    return {
        "schema": 1, "status": "FROZEN_NOT_RUN", "stage": STAGE,
        "version": version, "block": block,
        "container": f"kt-round14a-b-profile-{lower}-v{version}",
        "evidence": f"20260928-round14a-b-profile-{lower}-v{version}",
        "parent": "ROUND14_FINAL_S0_UNIFORM_Q32_PROMOTED_QSA_H129",
        "threshold": 129, "profiler": "BOUNDED_OBSERVER_ON",
        "generation_limit": 4, "abort_limit": 0, "automatic_retries": 0,
        "raw_result": "round14a-b-profile-raw.json",
        "final_result": "round14a-b-profile-final.json",
        "driver_pass_status": "PASS_ROUND14A_B_PROFILE_DRIVER_RAW",
        "process_pass_status": "PASS_ROUND14A_B_PROFILE_PROCESS",
        "driver_command": [
            "{DRIVER_ROOT}/round14a_b_profile_driver.py",
            "--version", str(version), "--block", block,
            "--out", "/out", "--fixture",
            "{DRIVER_ROOT}/round11-t1-fixture.json",
        ],
    }


def _patch_launcher(source, value):
    source = replace_once(source,
        "if a.version != 274 or a.cost_mode != 'OFF':",
        f"if a.version != {value['version']} or a.cost_mode != 'OFF':", "version")
    source = replace_once(source,
        "name='kt-round14-c-profile-a-v274';out=root/'evidence'/'20260927-round14-c-profile-a-v274'",
        f"name='{value['container']}';out=root/'evidence'/'{value['evidence']}'", "identity")
    source = replace_once(source,
        "if budget.get('stage')!='ROUND14_C_FRESH_S0_PROFILE':raise RuntimeError('wrong Round14 C profile budget')",
        f"if budget.get('stage')!='{STAGE}':raise RuntimeError('wrong Round14a B profile budget')",
        "budget")
    source = replace_once(source,
        " 'ROUND13_ATTRIBUTION_SPANS=/out/round14-c-spans.jsonl',\n"
        " 'ROUND13_ATTRIBUTION_BINDINGS=/out/round14-c-bindings.json'):",
        " 'ROUND13_ATTRIBUTION_SPANS=/out/round14a-b-profile-spans.jsonl',\n"
        " 'ROUND13_ATTRIBUTION_BINDINGS=/out/round14a-b-profile-bindings.json',\n"
        " 'ROUND14A_FIVE_BUCKET_ENABLE=1',\n"
        " 'ROUND14A_FIVE_BUCKET_PATH=/out/round14a-five-bucket-memory.json'):",
        "observer environment")
    compile(source, "launch_hybrid.py", "exec")
    return source


def _refresh_overlay(output, memory_source):
    overlay = output / "overlay-python"
    target = overlay / "sglang/srt/layers/round9/round14a_five_bucket.py"
    shutil.copy2(memory_source, target)
    runner = overlay / "sglang/srt/model_executor/model_runner.py"
    anchor = (
        "        from sglang.srt.layers.round9.streaming_binding import streaming_model_ready\n"
        "        streaming_model_ready(self)\n"
    )
    addition = anchor + (
        "        from sglang.srt.layers.round9 import round14a_five_bucket\n"
        "        round14a_five_bucket.capture(self)\n"
    )
    runner.write_text(replace_once(runner.read_text(), anchor, addition, "model-ready hook"))
    compile(runner.read_text(), str(runner), "exec")
    tools = output / "source/tools"
    import sys
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        receipt_path = overlay / "round9-candidate-overlay.json"
        receipt = json.loads(receipt_path.read_text())
        receipt.setdefault("modules", {})[target.name] = sha256(target)
        receipt["candidate_files"] = tree_files(overlay)
        receipt["candidate_tree_sha256"] = tree_digest(receipt["candidate_files"])
        receipt["round14a_five_bucket"] = {
            "status": "INSTALLED_STORAGE_DEDUP_CENSUS",
            "module_sha256": sha256(target),
            "model_runner_sha256": sha256(runner),
            "observer": "EXISTING_DEFERRED_EVENT_QUERY_NO_NEW_SYNCHRONIZE",
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        sys.path.remove(str(tools))


def build(parent, output, version, driver, controller, supervisor, preflight,
          memory_source):
    parent = Path(parent).resolve(strict=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    value = contract(version)
    old_contract = json.loads((parent / "ROUND14_C_PROFILE_CONTRACT.json").read_text())
    if (old_contract.get("version"), old_contract.get("block")) != (274, "A"):
        raise RuntimeError("Round14a B requires exact fresh S0 v274 source parent")
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    for name in ("round14-source-sha256.json", "round14a-source-sha256.json",
                 "host-controller.json", "execution-budget.lock",
                 "ROUND14_C_PROFILE_CONTRACT.json", "round14_c_profile_controller.py",
                 "round14_c_profile_supervisor.py", "round14_c_profile_driver.py"):
        (output / name).unlink(missing_ok=True)
    for source in (driver, controller, supervisor, preflight):
        shutil.copy2(Path(source).resolve(strict=True), output / Path(source).name)
    shutil.copy2(Path(memory_source).resolve(strict=True), output / "source/runtime/round14a_five_bucket.py")

    launcher = output / "source/tools/launch_hybrid.py"
    launcher.write_text(_patch_launcher(launcher.read_text(), value))
    monitor = output / "source/tools/monitor_owned_server.py"
    monitor.write_text(replace_once(
        monitor.read_text(),
        "r'|kt-round14-a-k-[ab]-v(?:235|238)|kt-round14-c-profile-[ab]-v(?:249|250|274|275))', name) is not None",
        "r'|kt-round14-a-k-[ab]-v(?:235|238)|kt-round14-c-profile-[ab]-v(?:249|250|274|275)'\n"
        "        r'|kt-round14a-b-profile-[ab]-v(?:298|299))', name) is not None",
        "monitor identity"))
    compile(monitor.read_text(), str(monitor), "exec")
    _refresh_overlay(output, Path(memory_source).resolve(strict=True))

    framework_path = output / "framework-profile.json"
    framework = json.loads(framework_path.read_text())
    framework["prefill"]["threshold_tokens"] = 129
    framework_path.write_text(json.dumps(framework, indent=2) + "\n")
    streaming_path = output / "streaming-profile.json"
    streaming = json.loads(streaming_path.read_text())
    streaming["threshold_tokens"] = 129
    streaming_path.write_text(json.dumps(streaming, indent=2) + "\n")
    (output / "ROUND14A_B_PROFILE_CONTRACT.json").write_text(
        json.dumps(value, indent=2) + "\n")
    budget = {"schema": 1, "stage": STAGE, "version": version,
              "block": value["block"], "model_starts_limit": 1,
              "model_starts_used": 0, "starts": [], "generation_limit": 4,
              "abort_limit": 0, "automatic_retries_limit": 0}
    (output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    for path in (launcher, monitor, output / Path(driver).name,
                 output / Path(controller).name, output / Path(supervisor).name,
                 output / Path(preflight).name,
                 output / "overlay-python/sglang/srt/layers/round9/round14a_five_bucket.py"):
        compile(path.read_text(), str(path), "exec")
    manifest = tree(output)
    manifest_path = output / "round14a-source-sha256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if tree(output) != manifest:
        raise RuntimeError("Round14a B snapshot manifest mismatch")
    return {"schema": 1, "status": "PASS_ROUND14A_B_PROFILE_SNAPSHOT_BUILD",
            "version": version, "block": value["block"],
            "output": str(output.resolve()), "manifest_sha256": sha256(manifest_path),
            "immutable_files": len(manifest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=tuple(ROLES), required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--memory-source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.parent, args.output, args.version, args.driver,
                   args.controller, args.supervisor, args.preflight,
                   args.memory_source)
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["version"], report["block"], flush=True)


if __name__ == "__main__":
    main()
