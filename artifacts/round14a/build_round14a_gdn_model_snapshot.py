#!/usr/bin/env python3
"""Freeze one Round14a GDN full-model smoke or ABBA snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pprint
import shutil


ROLES = {
    292: ("smoke", "baseline", "S"), 293: ("smoke", "candidate", "S"),
    294: ("abba", "baseline", "A"), 295: ("abba", "candidate", "A"),
    296: ("abba", "candidate", "B"), 297: ("abba", "baseline", "B"),
}
STAGE = "ROUND14A_A_GDN_FULL_MODEL"
FROZEN_QWEN_SHA256 = "1986d8a880fe1503848eadf02a8ed572d543eac967fcb583b6f67d42250f3545"
MUTABLE = {"round14a-source-sha256.json", "execution-budget.json",
           "execution-budget.lock", "host-controller.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Round14a GDN model {label} precondition mismatch")
    return source.replace(old, new)


def tree(root: Path) -> dict[str, str]:
    root = Path(root)
    return {str(path.relative_to(root)): sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
            and str(path.relative_to(root)) not in MUTABLE
            and not str(path.relative_to(root)).startswith("host-controller")}


def contract(version: int) -> dict:
    kind, implementation, block = ROLES[version]
    lower = f"{kind}-{implementation}-{block.lower()}"
    attempts = 8 if kind == "smoke" else 20
    aborts = 1 if kind == "smoke" else 0
    return {
        "schema": 1, "status": "FROZEN_NOT_RUN", "stage": STAGE,
        "version": version, "kind": kind, "implementation": implementation,
        "block": block, "container": f"kt-round14a-a-gdn-{lower}-v{version}",
        "evidence": f"20260927-round14a-a-gdn-{lower}-v{version}",
        "parent": "ROUND14A_A_GDN_COMPONENT_V290_COMPUTE_IDENTITY",
        "candidate": "NPU_GDN_CORE_PLUS_FROZEN_QWEN4_SIGMOID_GATE",
        "candidate_default": "OFF", "threshold": 129, "profiler": "OFF",
        "generation_limit": attempts, "abort_limit": aborts,
        "automatic_retries": 0,
        "raw_result": "round14a-gdn-model-raw.json",
        "final_result": "round14a-gdn-model-final.json",
        "driver_pass_status": "PASS_ROUND14A_A_GDN_MODEL_DRIVER_RAW",
        "process_pass_status": "PASS_ROUND14A_A_GDN_MODEL_PROCESS",
        "frozen_qwen_sha256": FROZEN_QWEN_SHA256,
        "driver_command": [
            "{DRIVER_ROOT}/round14a_gdn_model_driver.py", "--version", str(version),
            "--out", "/out", "--fixture10", "{DRIVER_ROOT}/round10-t1-fixture.json",
            "--fixture11", "{DRIVER_ROOT}/round11-t1-fixture.json",
        ],
    }


def _patch_approved_binding(source: str, document: dict) -> str:
    start = source.index("APPROVED_QSA_BINDING = {")
    end = source.index("\n\n\ndef _bind_qsa_source", start)
    rendered = "APPROVED_QSA_BINDING = " + pprint.pformat(document, width=88, sort_dicts=False)
    patched = source[:start] + rendered + source[end:]
    compile(patched, "context_reference_gate.py", "exec")
    return patched


def _qsa_binding(parent: Path) -> dict:
    old_path = parent / "ROUND12_QSA_SOURCE_BINDING.json"
    old = json.loads(old_path.read_text())
    return {
        **old,
        "status": "PASS_ROUND14A_QSA_GDN_SOURCE_BINDING",
        "candidate": "QUERY_AWARE_VECTORIZED_SPARSE_GQA_WITH_GDN_SELECTOR",
        "candidate_qwen4_exp_sha256": FROZEN_QWEN_SHA256,
        "round12_qsa_source_binding_sha256": sha256(old_path),
        "round14a_gdn_component_result_sha256": sha256(
            parent.parents[1] / "A_GDN_COMPONENT_RESULT.json"),
        "scope": "QSA_BYTES_UNCHANGED_GDN_BRANCH_ADDED_DEFAULT_OFF",
    }


def _patch_launcher(source: str, value: dict) -> str:
    source = replace_once(source,
        "if a.version != 274 or a.cost_mode != 'OFF':",
        f"if a.version != {value['version']} or a.cost_mode != 'OFF':", "version")
    source = replace_once(source,
        "name='kt-round14-c-profile-a-v274';out=root/'evidence'/'20260927-round14-c-profile-a-v274'",
        f"name='{value['container']}';out=root/'evidence'/'{value['evidence']}'", "identity")
    source = replace_once(source,
        "if budget.get('stage')!='ROUND14_C_FRESH_S0_PROFILE':raise RuntimeError('wrong Round14 C profile budget')",
        f"if budget.get('stage')!='{STAGE}':raise RuntimeError('wrong Round14a GDN model budget')",
        "budget")
    source = replace_once(source,
        "qsa_binding_path=source.parent/'ROUND12_QSA_SOURCE_BINDING.json'",
        "qsa_binding_path=source.parent/'ROUND14A_QSA_GDN_SOURCE_BINDING.json'",
        "QSA/GDN source binding")
    observer = (
        "for env in ('ROUND13_ATTRIBUTION_ENABLE=1',\n"
        " 'ROUND13_ATTRIBUTION_SPANS=/out/round14-c-spans.jsonl',\n"
        " 'ROUND13_ATTRIBUTION_BINDINGS=/out/round14-c-bindings.json'):\n"
        " args+=['-e',env]\n"
    )
    source = replace_once(source, observer, "", "profiler removal")
    anchor = "args+=['-e','ROUND9_HYBRID_PROBE=1']\n"
    enabled = "1" if value["implementation"] == "candidate" else "0"
    addition = anchor + (
        f"args+=['-e','ROUND14_GDN_BACKEND_ENABLE={enabled}']\n"
        "args+=['-e','ROUND14A_GDN_BRANCH_PATH=/out/round14a-gdn-branch.jsonl']\n"
    )
    source = replace_once(source, anchor, addition, "GDN environment")
    compile(source, "launch_hybrid.py", "exec")
    return source


def _refresh_overlay(output: Path, binding_source: Path) -> None:
    overlay = output / "overlay-python"
    target = overlay / "sglang/srt/layers/round9/round14a_gdn_branch_binding.py"
    shutil.copy2(binding_source, target)
    runner = overlay / "sglang/srt/model_executor/model_runner.py"
    anchor = (
        "        from sglang.srt.layers.round9.streaming_binding import streaming_model_ready\n"
        "        streaming_model_ready(self)\n"
    )
    addition = anchor + (
        "        from sglang.srt.layers.round9.round14a_gdn_branch_binding import gdn_branch_model_ready\n"
        "        gdn_branch_model_ready(self)\n"
    )
    runner.write_text(replace_once(runner.read_text(), anchor, addition, "model hook"))
    compile(runner.read_text(), str(runner), "exec")
    tools = output / "source/tools"
    import sys
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        receipt_path = overlay / "round9-candidate-overlay.json"
        receipt = json.loads(receipt_path.read_text())
        receipt.setdefault("modules", {})[target.name] = sha256(target)
        receipt["candidate_model_sha256"] = sha256(
            overlay / "sglang/srt/models/qwen4_exp.py")
        receipt["candidate_files"] = tree_files(overlay)
        receipt["candidate_tree_sha256"] = tree_digest(receipt["candidate_files"])
        receipt["round14a_gdn_model_binding"] = {
            "status": "INSTALLED_ACTUAL_FORWARD_COMPLETION_HOOK",
            "binding_sha256": sha256(target), "model_runner_sha256": sha256(runner),
            "qwen_unchanged_sha256": receipt["candidate_model_sha256"],
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        sys.path.remove(str(tools))


def build(parent: Path, output: Path, version: int, driver: Path,
          controller: Path, supervisor: Path, preflight: Path,
          binding_source: Path) -> dict:
    parent = parent.resolve(strict=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    value = contract(version)
    parent_contract = json.loads((parent / "ROUND14A_A_GDN_COMPONENT_CONTRACT.json").read_text())
    if (parent_contract.get("version"), parent_contract.get("block")) != (290, "A"):
        raise RuntimeError("Round14a GDN model requires exact v290 compute parent")
    qwen_parent = parent / "overlay-python/sglang/srt/models/qwen4_exp.py"
    if sha256(qwen_parent) != FROZEN_QWEN_SHA256:
        raise RuntimeError("Round14a GDN qwen source drift")
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    for name in ("source-sha256.json", "round14-source-sha256.json",
                 "round14a-source-sha256.json", "host-controller.json",
                 "execution-budget.lock"):
        (output / name).unlink(missing_ok=True)
    for source_path in (driver, controller, supervisor, preflight):
        shutil.copy2(source_path.resolve(strict=True), output / source_path.name)
    launcher = output / "source/tools/launch_hybrid.py"
    launcher.write_text(_patch_launcher(launcher.read_text(), value))
    monitor = output / "source/tools/monitor_owned_server.py"
    old = "r'|kt-round14-a-k-[ab]-v(?:235|238)|kt-round14-c-profile-[ab]-v(?:249|250|274|275))', name) is not None"
    new = ("r'|kt-round14-a-k-[ab]-v(?:235|238)|kt-round14-c-profile-[ab]-v(?:249|250|274|275)'\n"
           "        r'|kt-round14a-a-gdn-(?:smoke|abba)-(?:baseline|candidate)-[sab]-v(?:292|293|294|295|296|297))', name) is not None")
    monitor.write_text(replace_once(monitor.read_text(), old, new, "monitor identity"))
    compile(monitor.read_text(), str(monitor), "exec")
    binding = _qsa_binding(parent)
    (output / "ROUND14A_QSA_GDN_SOURCE_BINDING.json").write_text(
        json.dumps(binding, indent=2) + "\n")
    for path in (output / "source/runtime/context_reference_gate.py",
                 output / "overlay-python/sglang/srt/layers/round9/context_reference_gate.py"):
        path.write_text(_patch_approved_binding(path.read_text(), binding))
    # Refresh the overlay receipt only after every overlay mutation, including
    # the Round14a QSA/GDN source gate above.
    _refresh_overlay(output, binding_source.resolve(strict=True))

    framework_path = output / "framework-profile.json"
    framework = json.loads(framework_path.read_text())
    framework["prefill"]["threshold_tokens"] = 129
    framework_path.write_text(json.dumps(framework, indent=2) + "\n")
    streaming_path = output / "streaming-profile.json"
    streaming = json.loads(streaming_path.read_text())
    streaming["threshold_tokens"] = 129
    streaming_path.write_text(json.dumps(streaming, indent=2) + "\n")
    (output / "ROUND14A_A_GDN_MODEL_CONTRACT.json").write_text(
        json.dumps(value, indent=2) + "\n")
    budget = {"schema": 1, "stage": STAGE, "version": version,
              "kind": value["kind"], "implementation": value["implementation"],
              "block": value["block"], "model_starts_limit": 1,
              "model_starts_used": 0, "starts": [],
              "generation_limit": value["generation_limit"],
              "abort_limit": value["abort_limit"], "automatic_retries_limit": 0}
    (output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    for path in (launcher, monitor, output / driver.name, output / controller.name,
                 output / supervisor.name, output / preflight.name,
                 output / "overlay-python/sglang/srt/layers/round9/round14a_gdn_branch_binding.py"):
        compile(path.read_text(), str(path), "exec")
    manifest = tree(output)
    manifest_path = output / "round14a-source-sha256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if tree(output) != manifest:
        raise RuntimeError("Round14a GDN model snapshot manifest mismatch")
    return {"schema": 1, "status": "PASS_ROUND14A_A_GDN_MODEL_SNAPSHOT_BUILD",
            "version": version, "kind": value["kind"],
            "implementation": value["implementation"], "block": value["block"],
            "output": str(output.resolve()), "manifest_sha256": sha256(manifest_path),
            "qwen_sha256": sha256(output / "overlay-python/sglang/srt/models/qwen4_exp.py"),
            "qsa_gdn_binding_sha256": sha256(output / "ROUND14A_QSA_GDN_SOURCE_BINDING.json"),
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
    parser.add_argument("--binding-source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.parent, args.output, args.version, args.driver, args.controller,
                   args.supervisor, args.preflight, args.binding_source)
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["version"], report["implementation"], flush=True)


if __name__ == "__main__":
    main()
