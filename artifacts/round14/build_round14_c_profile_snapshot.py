#!/usr/bin/env python3
"""Build one immutable Round14 C fresh-S0 attribution snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROLES = {249: "A", 250: "B", 274: "A", 275: "B"}
STAGE = "ROUND14_C_FRESH_S0_PROFILE"
MUTABLE = {
    "round14-source-sha256.json", "execution-budget.json",
    "execution-budget.lock", "host-controller.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Round14 C {label} precondition mismatch")
    return source.replace(old, new)


def tree(root: Path) -> dict[str, str]:
    root = Path(root)
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def contract(version: int) -> dict:
    block = ROLES.get(version)
    if block is None:
        raise ValueError("unsupported Round14 C profile version")
    lower = block.lower()
    return {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": STAGE,
        "version": version,
        "block": block,
        "container": f"kt-round14-c-profile-{lower}-v{version}",
        "evidence": f"20260927-round14-c-profile-{lower}-v{version}",
        "parent": "ROUND13_FINAL_S0_UNIFORM_Q32_PROMOTED_QSA",
        "default_threshold": 129,
        "profile_threshold": 513,
        "profiler": "BOUNDED_OBSERVER_ON",
        "generation_limit": 4,
        "abort_limit": 0,
        "automatic_retries": 0,
        "raw_result": "round14-c-profile-raw.json",
        "final_result": "round14-c-profile-final.json",
        "driver_pass_status": "PASS_ROUND14_C_PROFILE_DRIVER_RAW",
        "process_pass_status": "PASS_ROUND14_C_PROFILE_PROCESS",
        "driver_command": [
            "{DRIVER_ROOT}/round14_c_profile_driver.py",
            "--version", str(version), "--block", block,
            "--out", "/out", "--fixture",
            "{DRIVER_ROOT}/round11-t1-fixture.json",
        ],
    }


def patch_scheduler(source: str) -> str:
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
    return replace_once(source, old, new, "scheduler hook")


def _patch_launcher(source: str, value: dict) -> str:
    source = replace_once(
        source, "if a.version != 235 or a.cost_mode != 'OFF':",
        f"if a.version != {value['version']} or a.cost_mode != 'OFF':",
        "launcher version",
    )
    source = replace_once(
        source,
        "name='kt-round14-a-k-a-v235';out=root/'evidence'/'20260927-round14-a-k-a-v235'",
        f"name='{value['container']}';out=root/'evidence'/'{value['evidence']}'",
        "launcher identity",
    )
    source = replace_once(
        source,
        "if budget.get('stage')!='ROUND14_A_KT_NATIVE_TP8_SYSTEM_COMPARISON':raise RuntimeError('wrong Round14 A budget')",
        f"if budget.get('stage')!='{STAGE}':raise RuntimeError('wrong Round14 C profile budget')",
        "launcher budget",
    )
    anchor = "args+=['-e','ROUND9_HYBRID_PROBE=1']\n"
    observer = (
        anchor
        + "for env in ('ROUND13_ATTRIBUTION_ENABLE=1',\n"
        + " 'ROUND13_ATTRIBUTION_SPANS=/out/round14-c-spans.jsonl',\n"
        + " 'ROUND13_ATTRIBUTION_BINDINGS=/out/round14-c-bindings.json'):\n"
        + " args+=['-e',env]\n"
    )
    source = replace_once(source, anchor, observer, "observer environment")
    compile(source, "launch_hybrid.py", "exec")
    return source


def _refresh_overlay(output: Path, observer: Path) -> None:
    overlay = output / "overlay-python"
    target = overlay / "sglang/srt/layers/round9/round13_attribution.py"
    shutil.copy2(observer, target)
    scheduler = overlay / "sglang/srt/managers/scheduler.py"
    scheduler.write_text(patch_scheduler(scheduler.read_text()))
    compile(scheduler.read_text(), str(scheduler), "exec")
    tools = output / "source/tools"
    import sys
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        receipt_path = overlay / "round9-candidate-overlay.json"
        receipt = json.loads(receipt_path.read_text())
        receipt.setdefault("modules", {})[observer.name] = sha256(observer)
        receipt["candidate_files"] = tree_files(overlay)
        receipt["candidate_tree_sha256"] = tree_digest(receipt["candidate_files"])
        receipt["round14_c_profile"] = {
            "status": "INSTALLED_BOUNDED_ATTRIBUTION",
            "observer_sha256": sha256(observer),
            "scheduler_sha256": sha256(scheduler),
            "device_query": "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE",
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        sys.path.remove(str(tools))


def build(parent: Path, output: Path, version: int,
          driver: Path, observer: Path) -> dict:
    parent = Path(parent).resolve(strict=True)
    output = Path(output)
    driver = Path(driver).resolve(strict=True)
    observer = Path(observer).resolve(strict=True)
    value = contract(version)
    if output.exists():
        raise FileExistsError(output)
    parent_contract = json.loads((parent / "ROUND14_A_CONTRACT.json").read_text())
    if parent_contract.get("version") != 235 or parent_contract.get("backend") != "KT":
        raise RuntimeError("Round14 C requires the fresh S0 v235 compute parent")
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    for name in ("host-controller.json", "execution-budget.lock", "round14-source-sha256.json"):
        (output / name).unlink(missing_ok=True)

    launcher = output / "source/tools/launch_hybrid.py"
    launcher.write_text(_patch_launcher(launcher.read_text(), value))
    monitor = output / "source/tools/monitor_owned_server.py"
    monitor.write_text(replace_once(
        monitor.read_text(),
        "r'|kt-round14-a-k-[ab]-v(?:235|238))', name) is not None",
        "r'|kt-round14-a-k-[ab]-v(?:235|238)|kt-round14-c-profile-[ab]-v(?:249|250|274|275))', name) is not None",
        "monitor identity",
    ))
    _refresh_overlay(output, observer)
    shutil.copy2(observer, output / "source/runtime/round13_attribution.py")
    shutil.copy2(driver, output / "round14_c_profile_driver.py")

    controller = (output / "round14_a_controller.py").read_text()
    for old, new in (
        ("Round14 A KT", "Round14 C profile"),
        ("ROUND14_A_CONTRACT.json", "ROUND14_C_PROFILE_CONTRACT.json"),
        ("round14_a_supervisor.py", "round14_c_profile_supervisor.py"),
        ("round14-a-supervisor.json", "round14-c-profile-supervisor.json"),
    ):
        controller = controller.replace(old, new)
    (output / "round14_c_profile_controller.py").write_text(controller)
    supervisor = (output / "round14_a_supervisor.py").read_text()
    for old, new in (
        ("Round14 A KT", "Round14 C profile"),
        ("Round14 A", "Round14 C profile"),
        ("round14-a-supervisor.json", "round14-c-profile-supervisor.json"),
        ("round14-a-driver.log", "round14-c-profile-driver.log"),
    ):
        supervisor = supervisor.replace(old, new)
    (output / "round14_c_profile_supervisor.py").write_text(supervisor)
    (output / "round14_a_controller.py").unlink()
    (output / "round14_a_supervisor.py").unlink()
    (output / "ROUND14_A_CONTRACT.json").unlink()

    (output / "ROUND14_C_PROFILE_CONTRACT.json").write_text(
        json.dumps(value, indent=2) + "\n"
    )
    budget = {
        "schema": 1, "stage": STAGE, "version": version,
        "block": value["block"], "model_starts_limit": 1,
        "model_starts_used": 0, "starts": [], "generation_limit": 4,
        "abort_limit": 0, "automatic_retries_limit": 0,
    }
    (output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    framework = json.loads((output / "framework-profile.json").read_text())
    streaming = json.loads((output / "streaming-profile.json").read_text())
    if (framework.get("prefill", {}).get("threshold_tokens") != 513
            or streaming.get("threshold_tokens") != 513
            or streaming.get("chunk_tokens") != 32):
        raise RuntimeError("Round14 C parent profile is not frozen H513/chunk32")
    for path in (
        launcher, monitor, output / "round14_c_profile_controller.py",
        output / "round14_c_profile_supervisor.py", output / "round14_c_profile_driver.py",
    ):
        compile(path.read_text(), str(path), "exec")
    manifest = tree(output)
    (output / "round14-source-sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if tree(output) != manifest:
        raise RuntimeError("Round14 C snapshot manifest is inconsistent")
    return {
        "schema": 1,
        "status": "PASS_ROUND14_C_PROFILE_SNAPSHOT_BUILD",
        "version": version,
        "block": value["block"],
        "output": str(output.resolve()),
        "manifest_sha256": sha256(output / "round14-source-sha256.json"),
        "immutable_files": len(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=tuple(ROLES), required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = build(args.parent, args.output, args.version, args.driver, args.observer)
    if args.report:
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["version"], report["block"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
