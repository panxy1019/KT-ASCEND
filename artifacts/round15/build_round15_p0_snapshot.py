#!/usr/bin/env python3
"""Freeze one Round15 fresh promoted-S0 profile snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


ROLES = {
    331: "A", 332: "B", 333: "A", 334: "B",
    335: "A", 336: "B",
    337: "A", 338: "B",
}
STAGE = "ROUND15_P0_FRESH_PROMOTED_S0_PROFILE"
MUTABLE = {
    "round15-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise RuntimeError(f"Round15 P0 {label} precondition mismatch")
    return source.replace(old, new)


def tree(root):
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


def contract(version):
    block = ROLES[version]
    lower = block.lower()
    return {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": STAGE,
        "version": version,
        "block": block,
        "container": f"kt-round15-p0-{lower}-v{version}",
        "evidence": f"20260928-round15-p0-{lower}-v{version}",
        "parent": "ROUND14A_V295_PROMOTED_GDN_S0",
        "threshold": 129,
        "profiler": "BOUNDED_EXISTING_EVENT_OBSERVER",
        "graph": False,
        "generation_limit": 4,
        "abort_limit": 0,
        "automatic_retries": 0,
        "raw_result": "round15-p0-raw.json",
        "final_result": "round15-p0-final.json",
        "driver_pass_status": "PASS_ROUND15_P0_PROFILE_DRIVER_RAW",
        "process_pass_status": "PASS_ROUND15_P0_PROFILE_PROCESS",
        "driver_command": [
            "{DRIVER_ROOT}/round15_p0_driver.py",
            "--version", str(version),
            "--block", block,
            "--out", "/out",
            "--fixture", "{DRIVER_ROOT}/round11-t1-fixture.json",
        ],
    }


def patch_launcher(source, value):
    source = replace_once(
        source,
        "if a.version != 295 or a.cost_mode != 'OFF':",
        f"if a.version != {value['version']} or a.cost_mode != 'OFF':",
        "version gate",
    )
    source = replace_once(
        source,
        "name='kt-round14a-a-gdn-abba-candidate-a-v295';out=root/'evidence'/'20260927-round14a-a-gdn-abba-candidate-a-v295'",
        f"name='{value['container']}';out=root/'evidence'/'{value['evidence']}'",
        "launch identity",
    )
    source = replace_once(
        source,
        "args+=['-e','ROUND14_GDN_BACKEND_ENABLE=1']",
        "args+=['-e','ROUND14_GDN_BACKEND_ENABLE=1']\n"
        "for env in ('ROUND13_ATTRIBUTION_ENABLE=1',\n"
        " 'ROUND13_ATTRIBUTION_SPANS=/out/round14a-b-profile-spans.jsonl',\n"
        " 'ROUND13_ATTRIBUTION_BINDINGS=/out/round14a-b-profile-bindings.json',\n"
        " 'ROUND14A_FIVE_BUCKET_ENABLE=1',\n"
        " 'ROUND14A_FIVE_BUCKET_PATH=/out/round14a-five-bucket-memory.json',\n"
        " 'ROUND15_P0_PROFILE=1'):\n"
        " args+=['-e',env]",
        "observer environment",
    )
    source = replace_once(
        source,
        "if budget.get('stage')!='ROUND14A_A_GDN_FULL_MODEL':raise RuntimeError('wrong Round14a GDN model budget')",
        "if budget.get('stage')!='ROUND15_P0_FRESH_PROMOTED_S0_PROFILE':raise RuntimeError('wrong Round15 P0 model budget')",
        "budget stage gate",
    )
    compile(source, "launch_hybrid.py", "exec")
    return source


def patch_model_runner_for_five_bucket(source):
    anchor = (
        "        from sglang.srt.layers.round9.round14a_gdn_branch_binding import gdn_branch_model_ready\n"
        "        gdn_branch_model_ready(self)\n"
    )
    addition = anchor + (
        "        from sglang.srt.layers.round9 import round14a_five_bucket\n"
        "        round14a_five_bucket.capture(self)\n"
    )
    patched = replace_once(source, anchor, addition, "five-bucket model-ready hook")
    compile(patched, "model_runner.py", "exec")
    return patched


def install_five_bucket_census(output, memory_source):
    output = Path(output)
    memory_source = Path(memory_source).resolve(strict=True)
    runtime_target = output / "source/runtime/round14a_five_bucket.py"
    overlay_target = (
        output / "overlay-python/sglang/srt/layers/round9/round14a_five_bucket.py"
    )
    shutil.copy2(memory_source, runtime_target)
    shutil.copy2(memory_source, overlay_target)

    runner = output / "overlay-python/sglang/srt/model_executor/model_runner.py"
    runner.write_text(patch_model_runner_for_five_bucket(runner.read_text()))

    helper_path = output / "source/tools/install_candidate_overlay.py"
    spec = importlib.util.spec_from_file_location("_round15_overlay_receipt", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    overlay = output / "overlay-python"
    receipt_path = overlay / "round9-candidate-overlay.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.setdefault("modules", {})[overlay_target.name] = sha256(overlay_target)
    receipt["candidate_files"] = helper.tree_files(overlay)
    receipt["candidate_tree_sha256"] = helper.tree_digest(receipt["candidate_files"])
    receipt["round15_five_bucket"] = {
        "status": "INSTALLED_EXISTING_STORAGE_DEDUP_CENSUS",
        "module_sha256": sha256(overlay_target),
        "runtime_source_sha256": sha256(runtime_target),
        "model_runner_sha256": sha256(runner),
        "capture_point": "AFTER_PROMOTED_GDN_MODEL_READY_BEFORE_REQUESTS",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return {
        "module_sha256": sha256(overlay_target),
        "runtime_source_sha256": sha256(runtime_target),
        "model_runner_sha256": sha256(runner),
        "receipt_sha256": sha256(receipt_path),
    }


def patch_supervisor_control(source):
    source = replace_once(
        source,
        "import traceback\n\nfrom m3_cleanup import wait_for_guard_exit\n",
        "import traceback\n\n"
        "from round15_p0_control import classify_physical12_after, read_proc_cgroup\n"
        "from m3_cleanup import wait_for_guard_exit\n",
        "supervisor control import",
    )
    source = replace_once(
        source,
        "            if \"No process in device\" not in proc:\n"
        "                raise RuntimeError(\"physical12 is not empty after Round15 P0 profile cleanup\")\n"
        "            report[\"physical12_after\"] = \"EMPTY\"\n",
        "            physical = classify_physical12_after(proc, read_proc_cgroup)\n"
        "            report[\"physical12_after\"] = physical[\"status\"]\n"
        "            report[\"physical12_processes\"] = physical[\"processes\"]\n",
        "supervisor registered-training cleanup",
    )
    source = replace_once(
        source,
        "                \"physical12_after\": \"EMPTY\",\n",
        "                \"physical12_after\": report[\"physical12_after\"],\n"
        "                \"physical12_processes\": report[\"physical12_processes\"],\n",
        "supervisor final cleanup identity",
    )
    compile(source, "round15_p0_supervisor.py", "exec")
    return source


def patch_controller_control(source):
    source = replace_once(
        source,
        "from m3_cleanup import wait_for_guard_exit\n",
        "from m3_cleanup import wait_for_guard_exit\n"
        "from round15_p0_control import validate_supervisor_removed_container\n",
        "controller control import",
    )
    old = (
        "        found = json.loads(subprocess.check_output(\n"
        "            [\"docker\", \"inspect\", contract[\"container\"]], text=True, timeout=30))[0]\n"
    )
    new = (
        "        try:\n"
        "            found = json.loads(subprocess.check_output(\n"
        "                [\"docker\", \"inspect\", contract[\"container\"]],\n"
        "                text=True, timeout=30, stderr=subprocess.DEVNULL))[0]\n"
        "        except subprocess.CalledProcessError:\n"
        "            supervisor_path = out / \"round15-p0-supervisor.json\"\n"
        "            if not supervisor_path.is_file():\n"
        "                raise\n"
        "            supervisor = json.loads(supervisor_path.read_text())\n"
        "            proof = validate_supervisor_removed_container(supervisor, expected)\n"
        "            proof[\"guard_exit_ack\"] = wait_for_guard_exit(\n"
        "                out / \"monitor.jsonl\",\n"
        "                not_before=float(supervisor.get(\"started_at\", 0)))\n"
        "            return proof\n"
    )
    source = replace_once(source, old, new, "controller already-removed cleanup")
    compile(source, "round15_p0_controller.py", "exec")
    return source


def _generated_runtime_source(path, replacements):
    source = Path(path).read_text()
    for old, new in replacements:
        source = source.replace(old, new)
    compile(source, str(path), "exec")
    return source


def build(parent, output, version, driver, preflight, base_driver, base_controller,
          base_supervisor, memory_source):
    parent = Path(parent).resolve(strict=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    value = contract(version)
    qwen = parent / "overlay-python/sglang/srt/models/qwen4_exp.py"
    if sha256(qwen) != "1986d8a880fe1503848eadf02a8ed572d543eac967fcb583b6f67d42250f3545":
        raise RuntimeError("Round15 P0 parent is not the promoted Round14a Qwen source")
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    for name in (
        "round14a-source-sha256.json",
        "host-controller.json",
        "execution-budget.lock",
        "ROUND14A_A_GDN_MODEL_CONTRACT.json",
        "round14a_gdn_model_controller.py",
        "round14a_gdn_model_supervisor.py",
        "round14a_gdn_model_driver.py",
    ):
        (output / name).unlink(missing_ok=True)
    shutil.copy2(driver, output / "round15_p0_driver.py")
    shutil.copy2(preflight, output / "round15_p0_preflight.py")
    shutil.copy2(base_driver, output / "round14a_b_profile_driver.py")

    common = [
        ("Round14a B profile", "Round15 P0 profile"),
        ("round14a-source-sha256.json", "round15-source-sha256.json"),
        ("ROUND14A_B_PROFILE_CONTRACT.json", "ROUND15_P0_CONTRACT.json"),
        ("round14a_b_profile_driver.py", "round15_p0_driver.py"),
        ("round14a-b-profile-raw.json", "round15-p0-raw.json"),
        ("round14a-b-profile-final.json", "round15-p0-final.json"),
        ("round14a-b-profile-supervisor.json", "round15-p0-supervisor.json"),
        ("round14a-b-profile-driver.log", "round15-p0-driver.log"),
        ("PASS_ROUND14A_B_PROFILE_PROCESS", "PASS_ROUND15_P0_PROFILE_PROCESS"),
    ]
    controller_source = _generated_runtime_source(base_controller, common + [
        ("round14a_b_profile_supervisor.py", "round15_p0_supervisor.py"),
    ])
    controller_source = patch_controller_control(controller_source)
    supervisor_source = patch_supervisor_control(
        _generated_runtime_source(base_supervisor, common)
    )
    (output / "round15_p0_controller.py").write_text(controller_source)
    (output / "round15_p0_supervisor.py").write_text(supervisor_source)
    shutil.copy2(Path(__file__).with_name("round15_p0_control.py"),
                 output / "round15_p0_control.py")

    launcher = output / "source/tools/launch_hybrid.py"
    launcher.write_text(patch_launcher(launcher.read_text(), value))
    monitor = output / "source/tools/monitor_owned_server.py"
    monitor.write_text(replace_once(
        monitor.read_text(),
        "r'|kt-round14a-a-gdn-(?:smoke|abba)-(?:baseline|candidate)-[sab]-v(?:292|293|294|295|296|297))', name) is not None",
        "r'|kt-round14a-a-gdn-(?:smoke|abba)-(?:baseline|candidate)-[sab]-v(?:292|293|294|295|296|297)'\n"
        "        r'|kt-round15-p0-[ab]-v(?:331|332|333|334|335|336|337|338))', name) is not None",
        "monitor identity",
    ))
    compile(monitor.read_text(), str(monitor), "exec")
    census = install_five_bucket_census(output, memory_source)

    (output / "ROUND15_P0_CONTRACT.json").write_text(json.dumps(value, indent=2) + "\n")
    budget = {
        "schema": 1,
        "stage": STAGE,
        "version": version,
        "block": value["block"],
        "model_starts_limit": 1,
        "model_starts_used": 0,
        "starts": [],
        "generation_limit": 4,
        "abort_limit": 0,
        "automatic_retries_limit": 0,
    }
    (output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    manifest = tree(output)
    manifest_path = output / "round15-source-sha256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if tree(output) != manifest:
        raise RuntimeError("Round15 P0 snapshot manifest mismatch")
    return {
        "schema": 1,
        "status": "PASS_ROUND15_P0_SNAPSHOT_BUILD",
        "version": version,
        "block": value["block"],
        "output": str(output.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "qwen_sha256": sha256(qwen),
        "five_bucket_census": census,
        "immutable_files": len(manifest),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=tuple(ROLES), required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--base-driver", type=Path, required=True)
    parser.add_argument("--base-controller", type=Path, required=True)
    parser.add_argument("--base-supervisor", type=Path, required=True)
    parser.add_argument("--memory-source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        args.parent, args.output, args.version, args.driver, args.preflight,
        args.base_driver, args.base_controller, args.base_supervisor,
        args.memory_source,
    )
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["version"], report["block"], flush=True)


if __name__ == "__main__":
    main()
