#!/usr/bin/env python3
"""Freeze one bounded Round14a true-concurrency model snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pprint
import shutil
import sys

from round14a_concurrency_patches import (
    patch_all_runtime, patch_kv_cache_mixin, patch_qwen, replace_once,
)


ROLES = {
    322: ("smoke", "C2", (2,), 2), 323: ("smoke", "C2", (2,), 2),
    324: ("smoke", "C2", (2,), 2), 325: ("smoke", "C2", (2,), 2),
    326: ("smoke", "C4", (4,), 4),
    327: ("mirror", "A", (1, 2, 4), 4),
    328: ("mirror", "B", (4, 2, 1), 4),
    329: ("mirror", "C", (4, 2, 1), 4),
    330: ("mirror", "D", (1, 2, 4), 4),
}
STAGE = "ROUND14A_C_TRUE_CONCURRENCY"
MUTABLE = {"round14a-source-sha256.json", "execution-budget.json",
           "execution-budget.lock", "host-controller.json"}


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(root):
    root = Path(root)
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
            and path.suffix != ".pyc" and str(path.relative_to(root)) not in MUTABLE
            and not str(path.relative_to(root)).startswith("host-controller")}


def contract(version):
    kind, block, tiers, maximum = ROLES[version]
    suffix = f"{kind}-{block.lower()}"
    attempts = 9 if kind == "smoke" else 82
    aborts = 1 if kind == "smoke" else 0
    return {
        "schema": 1, "status": "FROZEN_NOT_RUN", "stage": STAGE,
        "version": version, "kind": kind, "block": block,
        "tier_order": list(tiers), "server_max_concurrency": maximum,
        "container": f"kt-round14a-c-concurrency-{suffix}-v{version}",
        "evidence": f"20260928-round14a-c-concurrency-{suffix}-v{version}",
        "parent": "ROUND14_S0_UNIFORM_Q32_GDN_BASELINE_QSA_PROMOTED_PLE_ASYNC_H129",
        "profiler": "OFF", "threshold": 129, "dynamic": "OFF",
        "generation_limit": attempts, "abort_limit": aborts,
        "automatic_retries": 0,
        "raw_result": "round14a-concurrency-raw.json",
        "final_result": "round14a-concurrency-final.json",
        "driver_pass_status": "PASS_ROUND14A_C_CONCURRENCY_DRIVER_RAW",
        "process_pass_status": "PASS_ROUND14A_C_CONCURRENCY_PROCESS",
        "driver_command": [
            "{DRIVER_ROOT}/round14a_concurrency_driver.py", "--version", str(version),
            "--out", "/out", "--fixture", "{DRIVER_ROOT}/round10-t1-fixture.json",
        ],
    }


def _patch_approved_binding(source, document):
    start = source.index("APPROVED_QSA_BINDING = {")
    end = source.index("\n\n\ndef _bind_qsa_source", start)
    rendered = "APPROVED_QSA_BINDING = " + pprint.pformat(
        document, width=88, sort_dicts=False)
    patched = source[:start] + rendered + source[end:]
    compile(patched, "context_reference_gate.py", "exec")
    return patched


def _patch_launcher(source, value):
    source = replace_once(source,
        "if a.version != 294 or a.cost_mode != 'OFF':",
        f"if a.version != {value['version']} or a.cost_mode != 'OFF':", "version")
    source = replace_once(source,
        "name='kt-round14a-a-gdn-abba-baseline-a-v294';out=root/'evidence'/'20260927-round14a-a-gdn-abba-baseline-a-v294'",
        f"name='{value['container']}';out=root/'evidence'/'{value['evidence']}'", "identity")
    source = replace_once(source,
        "if budget.get('stage')!='ROUND14A_A_GDN_FULL_MODEL':raise RuntimeError('wrong Round14a GDN model budget')",
        f"if budget.get('stage')!='{STAGE}':raise RuntimeError('wrong Round14a concurrency budget')",
        "budget stage")
    source = replace_once(source,
        "qsa_binding_path=source.parent/'ROUND14A_QSA_GDN_SOURCE_BINDING.json'",
        "qsa_binding_path=source.parent/'ROUND14A_CONCURRENCY_QSA_SOURCE_BINDING.json'",
        "QSA source binding")
    env_anchor = "args+=['-e','ROUND9_HYBRID_PROBE=1']\n"
    source = replace_once(source, env_anchor, env_anchor +
        f"args+=['-e','ROUND14A_CONCURRENCY_MAX={value['server_max_concurrency']}']\n",
        "concurrency environment")
    compile(source, "launch_hybrid.py", "exec")
    return source


def _refresh_overlay(output, value):
    overlay = output / "overlay-python"
    tools = output / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        receipt_path = overlay / "round9-candidate-overlay.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["candidate_model_sha256"] = sha256(
            overlay / "sglang/srt/models/qwen4_exp.py")
        receipt["candidate_files"] = tree_files(overlay)
        receipt["candidate_tree_sha256"] = tree_digest(receipt["candidate_files"])
        receipt["round14a_concurrency"] = {
            "status": "INSTALLED_BOUNDED_PER_REQUEST_STATE_AND_ENGINE_BATCH",
            "version": value["version"],
            "max_concurrency": value["server_max_concurrency"],
            "runtime_sha256": sha256(overlay /
                "sglang/srt/layers/round9/round14a_concurrency_runtime.py"),
            "qwen_sha256": receipt["candidate_model_sha256"],
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally: sys.path.remove(str(tools))


def build(parent, output, version, driver, controller, supervisor, preflight,
          runtime_source):
    parent = Path(parent).resolve(strict=True); output = Path(output)
    if output.exists(): raise FileExistsError(output)
    value = contract(version)
    parent_contract = json.loads((parent / "ROUND14A_A_GDN_MODEL_CONTRACT.json").read_text())
    if (parent_contract.get("version"), parent_contract.get("implementation")) != (294, "baseline"):
        raise RuntimeError("concurrency snapshot requires exact Round14 S0 v294 parent")
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in output.rglob("*"):
        if path.is_file(): path.chmod(path.stat().st_mode | 0o200)
    for name in ("source-sha256.json", "round14-source-sha256.json",
                 "round14a-source-sha256.json", "host-controller.json",
                 "execution-budget.lock", "ROUND14A_A_GDN_MODEL_CONTRACT.json"):
        (output / name).unlink(missing_ok=True)
    for path in (driver, controller, supervisor, preflight):
        shutil.copy2(Path(path).resolve(strict=True), output / Path(path).name)

    overlay_runtime = output / "overlay-python/sglang/srt/layers/round9"
    source_runtime = output / "source/runtime"
    shutil.copy2(runtime_source, overlay_runtime / runtime_source.name)
    shutil.copy2(runtime_source, source_runtime / runtime_source.name)
    qwen = output / "overlay-python/sglang/srt/models/qwen4_exp.py"
    qwen.write_text(patch_qwen(qwen.read_text()))
    kv_mixin = output / "overlay-python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py"
    kv_mixin.write_text(patch_kv_cache_mixin(kv_mixin.read_text()))
    for root in (overlay_runtime, source_runtime):
        for name in ("context_reference_binding.py", "ple_binding.py",
                     "prefill_runtime.py", "prefill_binding.py", "context_binding.py",
                     "framework_hooks.py", "streaming_binding.py", "hybrid_probe.py"):
            path = root / name
            path.write_text(patch_all_runtime(str(path), path.read_text()))

    old_binding_path = parent / "ROUND14A_QSA_GDN_SOURCE_BINDING.json"
    binding = json.loads(old_binding_path.read_text())
    binding.update(
        status="PASS_ROUND14A_CONCURRENCY_QSA_SOURCE_BINDING",
        candidate="QUERY_AWARE_VECTORIZED_SPARSE_GQA_PER_REQUEST_BATCHED",
        candidate_qwen4_exp_sha256=sha256(qwen),
        candidate_context_reference_runtime_sha256=sha256(
            overlay_runtime / "context_reference_runtime.py"),
        candidate_context_reference_binding_sha256=sha256(
            overlay_runtime / "context_reference_binding.py"),
        round14a_gdn_qsa_source_binding_sha256=sha256(old_binding_path),
        round14a_concurrency_runtime_sha256=sha256(
            overlay_runtime / "round14a_concurrency_runtime.py"),
        scope="QSA_MATH_UNCHANGED_PER_REQUEST_STATE_SPLIT_AND_BOUNDED_BATCH",
    )
    binding_path = output / "ROUND14A_CONCURRENCY_QSA_SOURCE_BINDING.json"
    binding_path.write_text(json.dumps(binding, indent=2) + "\n")
    for path in (source_runtime / "context_reference_gate.py",
                 overlay_runtime / "context_reference_gate.py"):
        path.write_text(_patch_approved_binding(path.read_text(), binding))

    launcher = output / "source/tools/launch_hybrid.py"
    launcher.write_text(_patch_launcher(launcher.read_text(), value))
    monitor = output / "source/tools/monitor_owned_server.py"
    old = ("r'|kt-round14a-a-gdn-(?:smoke|abba)-(?:baseline|candidate)-[sab]-v(?:292|293|294|295|296|297))', name) is not None")
    new = ("r'|kt-round14a-a-gdn-(?:smoke|abba)-(?:baseline|candidate)-[sab]-v(?:292|293|294|295|296|297)'\n"
           "        r'|kt-round14a-c-concurrency-(?:smoke-c[24]|mirror-[abcd])-v(?:322|323|324|325|326|327|328|329|330))', name) is not None")
    monitor.write_text(replace_once(monitor.read_text(), old, new, "monitor identity"))
    compile(monitor.read_text(), str(monitor), "exec")

    profile_path = output / "source/runtime/profiles/q64_bf16_c64_sync_physical15.json"
    profile = json.loads(profile_path.read_text())
    profile["max_running_requests"] = value["server_max_concurrency"]
    profile["max_mamba_cache_size"] = value["server_max_concurrency"]
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    value["concurrency_runtime_sha256"] = sha256(
        overlay_runtime / "round14a_concurrency_runtime.py")
    (output / "ROUND14A_C_CONCURRENCY_CONTRACT.json").write_text(
        json.dumps(value, indent=2) + "\n")
    budget = {"schema": 1, "stage": STAGE, "version": version,
              "kind": value["kind"], "block": value["block"],
              "model_starts_limit": 1, "model_starts_used": 0, "starts": [],
              "generation_limit": value["generation_limit"],
              "abort_limit": value["abort_limit"], "automatic_retries_limit": 0}
    (output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    _refresh_overlay(output, value)
    for path in (launcher, monitor, output / driver.name, output / controller.name,
                 output / supervisor.name, output / preflight.name, qwen):
        compile(path.read_text(), str(path), "exec")
    manifest = tree(output)
    manifest_path = output / "round14a-source-sha256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if tree(output) != manifest:
        raise RuntimeError("Round14a concurrency snapshot manifest mismatch")
    return {"schema": 1, "status": "PASS_ROUND14A_C_CONCURRENCY_SNAPSHOT_BUILD",
            "version": version, "kind": value["kind"], "block": value["block"],
            "tiers": value["tier_order"], "max_concurrency": value["server_max_concurrency"],
            "output": str(output.resolve()), "manifest_sha256": sha256(manifest_path),
            "qwen_sha256": sha256(qwen), "binding_sha256": sha256(binding_path),
            "runtime_sha256": value["concurrency_runtime_sha256"],
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
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.parent, args.output, args.version, args.driver,
                  args.controller, args.supervisor, args.preflight,
                  args.runtime_source)
    args.report.write_text(json.dumps(value, indent=2) + "\n")
    print(value["status"], value["version"], flush=True)


if __name__ == "__main__": main()
