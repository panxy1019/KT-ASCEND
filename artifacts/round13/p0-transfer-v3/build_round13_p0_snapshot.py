#!/usr/bin/env python3
"""Build one immutable Round13 P0 S0 profile snapshot from Round12a v177."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys


ROLES = {184: "A", 185: "B", 186: "A", 187: "A"}
PARENT_IDENTITY = "ROUND12A_V174_V177_KT_PROMOTED_QSA_S0"
MUTABLE = {
    "round13-source-sha256.json", "execution-budget.json",
    "execution-budget.lock", "host-controller.json",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise ValueError(f"{label} precondition mismatch")
    return source.replace(old, new)


def patch_scheduler(source):
    old = (
        "                from sglang.srt.layers.round9.framework_hooks import framework_before\n"
        "                framework_before(self, batch)\n"
        "                result = self.run_batch(batch)\n"
        "                self.process_batch_result(batch, result)\n"
    )
    new = (
        "                from sglang.srt.layers.round9.framework_hooks import framework_before\n"
        "                framework_before(self, batch)\n"
        "                from sglang.srt.layers.round9.round13_attribution import (\n"
        "                    before as round13_before, after as round13_after)\n"
        "                round13_before(self, batch)\n"
        "                result = self.run_batch(batch)\n"
        "                round13_after(self)\n"
        "                self.process_batch_result(batch, result)\n"
    )
    return replace_once(source, old, new, "Round13 scheduler hook")


def patch_launcher(source, version):
    block = ROLES[version]
    source = replace_once(
        source, "if a.version != 177 or a.cost_mode != 'OFF':",
        f"if a.version != {version} or a.cost_mode != 'OFF':",
        "Round13 launcher version")
    source = replace_once(
        source,
        "name='kt-round12a-kt-b-v177';out=root/'evidence'/'20260927-round12a-kt-b-v177'",
        f"name='kt-round13-p0-s0-{block.lower()}-v{version}';out=root/'evidence'/'20260927-round13-p0-s0-{block.lower()}-v{version}'",
        "Round13 launcher identity")
    source = replace_once(
        source,
        "if budget.get('stage')!='ROUND12A_ABBA':raise RuntimeError('wrong Round12a ABBA budget')",
        "if budget.get('stage')!='ROUND13_P0_PROFILE':raise RuntimeError('wrong Round13 P0 budget')",
        "Round13 launcher budget")
    anchor = "args+=['-e','ROUND9_HYBRID_PROBE=1']\n"
    observer = (
        anchor +
        "for env in ('ROUND13_ATTRIBUTION_ENABLE=1',\n"
        " 'ROUND13_ATTRIBUTION_SPANS=/out/round13-spans.jsonl',\n"
        " 'ROUND13_ATTRIBUTION_BINDINGS=/out/round13-bindings.json',\n"
        " 'ROUND13_DEV_ROUTE_COUNTS=/out/round13-dev-route-counts.json'):\n"
        " args+=['-e',env]\n"
    )
    source = replace_once(source, anchor, observer, "Round13 observer environment")
    compile(source, "launch_hybrid.py", "exec")
    return source


def refresh_overlay(root, observer):
    overlay = root / "overlay-python"
    target = overlay / "sglang/srt/layers/round9/round13_attribution.py"
    shutil.copy2(observer, target)
    scheduler = overlay / "sglang/srt/managers/scheduler.py"
    scheduler.write_text(patch_scheduler(scheduler.read_text()))
    compile(scheduler.read_text(), str(scheduler), "exec")
    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        receipt_path = overlay / "round9-candidate-overlay.json"
        receipt = json.loads(receipt_path.read_text())
        receipt.setdefault("modules", {})[observer.name] = sha256(observer)
        receipt["candidate_files"] = tree_files(overlay)
        receipt["candidate_tree_sha256"] = tree_digest(receipt["candidate_files"])
        receipt["round13_p0"] = {
            "status": "INSTALLED_BOUNDED_ATTRIBUTION_AND_DEV_COUNTER",
            "observer_sha256": sha256(observer),
            "scheduler_sha256": sha256(scheduler),
            "device_query": "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE",
            "route_copy": "DEV_ONLY",
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        sys.path.remove(str(tools))


def source_manifest(root):
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def build(args):
    version = args.version
    block = ROLES.get(version)
    if block is None:
        raise ValueError("Round13 P0 version is outside its frozen/repair identities")
    if args.output.exists():
        raise FileExistsError(args.output)
    shutil.copytree(args.parent, args.output, ignore=shutil.ignore_patterns("__pycache__"))
    for path in (args.output, *args.output.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    for name in (
        "host-controller.json", "execution-budget.lock", "round12-source-sha256.json",
        "ROUND12A_MODEL_CONTRACT.json", "ROUND12A_ABBA_QUEUE.json",
    ):
        path = args.output / name
        if path.exists():
            path.unlink()

    inputs = args.inputs
    copies = {
        "round13_attribution.py": "source/runtime/round13_attribution.py",
        "round13_p0_driver.py": "round13_p0_driver.py",
        "round13_model_controller.py": "round13_model_controller.py",
        "round13_model_supervisor.py": "round13_model_supervisor.py",
        "round11_t4_driver.py": "round11_t4_driver.py",
        "round13_monitor_owned.py": "source/tools/monitor_owned_server.py",
        "round11-t1-fixture.json": "round11-t1-fixture.json",
        "round13-dev-fixture.json": "round13-dev-fixture.json",
    }
    for source, target in copies.items():
        shutil.copy2(inputs / source, args.output / target)
    observer = inputs / "round13_attribution.py"
    refresh_overlay(args.output, observer)

    launcher = args.output / "source/tools/launch_hybrid.py"
    launcher.write_text(patch_launcher(launcher.read_text(), version))
    framework_path = args.output / "framework-profile.json"
    framework = json.loads(framework_path.read_text())
    if (framework.get("resident_capacity") != 32 or
            framework.get("prefill", {}).get("update_residents") is not False or
            framework.get("prefill", {}).get("threshold_tokens") != 129):
        raise ValueError("P0 parent is not Round12a S0 H129")
    framework["prefill"]["threshold_tokens"] = 513
    framework_path.write_text(json.dumps(framework, indent=2) + "\n")
    streaming_path = args.output / "streaming-profile.json"
    streaming = json.loads(streaming_path.read_text())
    if streaming.get("threshold_tokens") != 129 or streaming.get("chunk_tokens") != 32:
        raise ValueError("P0 parent streaming identity mismatch")
    streaming["threshold_tokens"] = 513
    streaming_path.write_text(json.dumps(streaming, indent=2) + "\n")

    name = f"kt-round13-p0-s0-{block.lower()}-v{version}"
    evidence = f"20260927-round13-p0-s0-{block.lower()}-v{version}"
    budget = {
        "schema": 1, "stage": "ROUND13_P0_PROFILE", "version": version,
        "block": block, "model_starts_limit": 1, "model_starts_used": 0,
        "starts": [], "generation_limit": 7, "abort_limit": 0,
        "automatic_retries_limit": 0,
    }
    (args.output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    contract = {
        "schema": 1, "status": "FROZEN_NOT_RUN", "stage": "ROUND13_P0",
        "version": version, "block": block, "container": name,
        "evidence": evidence, "parent_identity": PARENT_IDENTITY,
        "default_threshold": 129, "profile_threshold": 513,
        "profiler": "BOUNDED_OBSERVER_ON", "generation_limit": 7,
        "abort_limit": 0, "automatic_retries": 0,
        "raw_result": "round13-p0-raw.json",
        "final_result": "round13-p0-final.json",
        "driver_pass_status": "PASS_ROUND13_P0_DRIVER_RAW",
        "process_pass_status": "PASS_ROUND13_P0_PROCESS",
        "driver_command": [
            "{DRIVER_ROOT}/round13_p0_driver.py", "--block", block,
            "--out", "/out", "--fixture", "{DRIVER_ROOT}/round11-t1-fixture.json",
            "--dev-fixture", "{DRIVER_ROOT}/round13-dev-fixture.json",
            "--parent-identity", PARENT_IDENTITY,
        ],
    }
    (args.output / "ROUND13_MODEL_CONTRACT.json").write_text(
        json.dumps(contract, indent=2) + "\n")
    manifest = source_manifest(args.output)
    (args.output / "round13-source-sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for source in (launcher, args.output / "round13_model_controller.py",
                   args.output / "round13_model_supervisor.py",
                   args.output / "round13_p0_driver.py"):
        compile(source.read_text(), str(source), "exec")
    print("PASS_ROUND13_P0_SNAPSHOT", block, version, len(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=tuple(ROLES), required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
