#!/usr/bin/env python3
"""Build one immutable Round14a B baseline/FPLUS1 request arm."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import sys


ROLES = {
    306: ("baseline", "A", "baseline-a"),
    307: ("fplus1", "A", "fplus1-a"),
    308: ("fplus1", "B", "fplus1-b"),
    309: ("baseline", "B", "baseline-b"),
    310: ("baseline", "A", "baseline-a"),
    311: ("fplus1", "A", "fplus1-a"),
    312: ("fplus1", "B", "fplus1-b"),
    313: ("baseline", "B", "baseline-b"),
    314: ("baseline", "A", "baseline-a"),
    315: ("fplus1", "A", "fplus1-a"),
    316: ("fplus1", "B", "fplus1-b"),
    317: ("baseline", "B", "baseline-b"),
    318: ("baseline", "A", "baseline-a"),
    319: ("fplus1", "A", "fplus1-a"),
    320: ("fplus1", "B", "fplus1-b"),
    321: ("baseline", "B", "baseline-b"),
}
MUTABLE = {"round14a-source-sha256.json", "execution-budget.json",
           "execution-budget.lock", "host-controller.json"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise ValueError(f"{label} precondition mismatch: {source.count(old)}")
    return source.replace(old, new)


def load_round13_builder(repo):
    path = repo / "artifacts/round13/build_round13_a_snapshot.py"
    spec = importlib.util.spec_from_file_location("round13_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def patch_candidate_sources(root, round13):
    framework = root / "overlay-python/sglang/srt/layers/round9/framework_hooks.py"
    value = round13.patch_framework(framework.read_text())
    value = value.replace("round13_placement_runtime", "round14a_resident_runtime")
    framework.write_text(value)

    qwen = root / "overlay-python/sglang/srt/models/qwen4_exp.py"
    value = round13.patch_qwen(qwen.read_text())
    value = value.replace("round13_placement_runtime", "round14a_resident_runtime")
    value = replace_once(
        value, "validate_layer_config(cfg, layer_id, round13_profile, args)",
        "validate_layer_config(cfg, layer_id, round13_profile)",
        "Round14a per-layer residency validator signature")
    value = value.replace("ROUND13_PLACEMENT_COMPONENT_GATE_SHA256",
                          "ROUND14A_RESIDENT_COMPONENT_GATE_SHA256")
    value = value.replace("ROUND13_PLACEMENT_COMPONENT_GATE",
                          "ROUND14A_RESIDENT_COMPONENT_GATE")
    qwen.write_text(value)

    prefill = root / "overlay-python/sglang/srt/layers/round9/prefill_binding.py"
    value = round13.patch_prefill_binding(prefill.read_text())
    value = value.replace(
        "Round13 nonuniform placement is qualified only for H513 Hybrid <=512",
        "Round14a FPLUS1 is qualified only for H129 Hybrid inputs <=128")
    value = value.replace("round13_placement_runtime", "round14a_resident_runtime")
    value = value.replace("Round13 nonuniform", "Round14a FPLUS1")
    prefill.write_text(value)

    probe = root / "overlay-python/sglang/srt/layers/round9/hybrid_probe.py"
    value = round13.patch_hybrid_probe(probe.read_text())
    value = value.replace("round13_placement_runtime", "round14a_resident_runtime")
    probe.write_text(value)
    return framework, qwen, prefill, probe


def bind_qsa_gdn_consumer(root, candidate_qwen_sha256):
    binding_path = root / "ROUND14A_QSA_GDN_SOURCE_BINDING.json"
    binding = json.loads(binding_path.read_text())
    old = binding.get("candidate_qwen4_exp_sha256")
    if old != "1986d8a880fe1503848eadf02a8ed572d543eac967fcb583b6f67d42250f3545":
        raise ValueError("Round14a parent QSA/GDN consumer identity mismatch")
    gate_digests = {}
    for gate_path in (
        root / "source/runtime/context_reference_gate.py",
        root / "overlay-python/sglang/srt/layers/round9/context_reference_gate.py",
    ):
        source = replace_once(gate_path.read_text(), old, candidate_qwen_sha256,
                              f"exact QSA/GDN placement consumer {gate_path}")
        compile(source, str(gate_path), "exec")
        gate_path.write_text(source)
        gate_digests[str(gate_path.relative_to(root))] = sha(gate_path)
    binding["candidate_qwen4_exp_sha256"] = candidate_qwen_sha256
    binding_path.write_text(json.dumps(binding, indent=2) + "\n")
    delta = {
        "schema": 1, "status": "PASS_ROUND14A_B_RESIDENT_SOURCE_BINDING",
        "parent_qwen4_exp_sha256": old,
        "resident_qwen4_exp_sha256": candidate_qwen_sha256,
        "context_reference_gate_sha256": gate_digests,
        "qsa_sha256": binding["candidate_qsa_sha256"],
        "gdn_component_result_sha256": binding["round14a_gdn_component_result_sha256"],
        "scope": binding["scope"],
    }
    (root / "ROUND14A_B_RESIDENT_SOURCE_BINDING.json").write_text(
        json.dumps(delta, indent=2) + "\n")
    return delta


def patch_launcher(source, version, strategy, label):
    source = replace_once(source,
        "if a.version != 294 or a.cost_mode != 'OFF':",
        f"if a.version != {version} or a.cost_mode != 'OFF':", "launcher version")
    source = replace_once(source,
        "name='kt-round14a-a-gdn-abba-baseline-a-v294';out=root/'evidence'/'20260927-round14a-a-gdn-abba-baseline-a-v294'",
        f"name='kt-round14a-b-resident-{label}-v{version}';out=root/'evidence'/'20260928-round14a-b-resident-{label}-v{version}'",
        "launcher identity")
    source = replace_once(source,
        "if budget.get('stage')!='ROUND14A_A_GDN_FULL_MODEL':raise RuntimeError('wrong Round14a GDN model budget')",
        "if budget.get('stage')!='ROUND14A_B_INCREMENTAL_RESIDENCY_REQUEST':raise RuntimeError('wrong Round14a resident model budget')",
        "launcher budget")
    if strategy == "fplus1":
        anchor = "args+=['-e','ROUND9_HYBRID_PROBE=1']\n"
        env = (anchor +
            f"args+=['-e','ROUND14A_RESIDENT_PROFILE=/round9/round14a-b-resident-{label}-v{version}/ROUND14A_RESIDENT_PROFILES.json']\n"
            f"args+=['-e','ROUND14A_RESIDENT_PROFILE_SHA256='+digest(root/'round14a-b-resident-{label}-v{version}/ROUND14A_RESIDENT_PROFILES.json')]\n"
            "args+=['-e','ROUND14A_RESIDENT_STRATEGY=fplus1']\n"
            f"args+=['-e','ROUND14A_RESIDENT_COMPONENT_GATE=/round9/round14a-b-resident-{label}-v{version}/B_RESIDENT_COMPONENT_GATE.json']\n"
            f"args+=['-e','ROUND14A_RESIDENT_COMPONENT_GATE_SHA256='+digest(root/'round14a-b-resident-{label}-v{version}/B_RESIDENT_COMPONENT_GATE.json')]\n")
        source = replace_once(source, anchor, env, "launcher residency environment")
    compile(source, "launch_hybrid.py", "exec")
    return source


def patch_control_files(root):
    controller = root / "round14a_resident_model_controller.py"
    supervisor = root / "round14a_resident_model_supervisor.py"
    c = controller.read_text()
    c = c.replace("Round14a GDN", "Round14a resident")
    c = c.replace("ROUND14A_A_GDN_MODEL_CONTRACT.json", "ROUND14A_B_RESIDENT_MODEL_CONTRACT.json")
    c = c.replace("round14a_gdn_model_supervisor.py", "round14a_resident_model_supervisor.py")
    c = c.replace("round14a-gdn-model-supervisor.json", "round14a-resident-model-supervisor.json")
    controller.write_text(c)
    s = supervisor.read_text()
    s = s.replace("Round14a GDN", "Round14a resident")
    s = s.replace("round14a-gdn-model-supervisor.json", "round14a-resident-model-supervisor.json")
    s = s.replace("round14a-gdn-model-driver.log", "round14a-resident-model-driver.log")
    supervisor.write_text(s)


def source_manifest(root):
    return {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            and str(path.relative_to(root)) not in MUTABLE
            and not str(path.relative_to(root)).startswith("host-controller")}


def build(args):
    strategy, block, label = ROLES[args.version]
    if args.output.exists():
        raise FileExistsError(args.output)
    shutil.copytree(args.parent, args.output, ignore=shutil.ignore_patterns("__pycache__"))
    for path in (args.output, *args.output.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    for name in ("host-controller.json", "execution-budget.lock", "round14a-source-sha256.json"):
        path = args.output / name
        if path.exists():
            path.unlink()
    copies = {
        "round14a_resident_model_driver.py": "round14a_resident_model_driver.py",
        "round14a_resident_model_preflight.py": "round14a_resident_model_preflight.py",
        "round14a_resident_runtime.py": "overlay-python/sglang/srt/layers/round9/round14a_resident_runtime.py",
        "ROUND14A_RESIDENT_PROFILES.json": "ROUND14A_RESIDENT_PROFILES.json",
        "B_RESIDENT_COMPONENT_GATE.json": "B_RESIDENT_COMPONENT_GATE.json",
    }
    for source, target in copies.items():
        shutil.copy2(args.inputs / source, args.output / target)
    shutil.copy2(args.output / "round14a_gdn_model_controller.py",
                 args.output / "round14a_resident_model_controller.py")
    shutil.copy2(args.output / "round14a_gdn_model_supervisor.py",
                 args.output / "round14a_resident_model_supervisor.py")
    patch_control_files(args.output)

    round13 = load_round13_builder(args.repo)
    source_delta = None
    changed = {}
    if strategy == "fplus1":
        framework, qwen, prefill, probe = patch_candidate_sources(args.output, round13)
        source_delta = bind_qsa_gdn_consumer(args.output, sha(qwen))
        changed = {"framework_hooks_sha256": sha(framework), "qwen4_exp_sha256": sha(qwen),
                   "prefill_binding_sha256": sha(prefill), "hybrid_probe_sha256": sha(probe)}

        overlay = args.output / "overlay-python"
        sys.path.insert(0, str(args.output / "source/tools"))
        try:
            from install_candidate_overlay import tree_digest, tree_files
            receipt_path = overlay / "round9-candidate-overlay.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["candidate_files"] = tree_files(overlay)
            receipt["candidate_tree_sha256"] = tree_digest(receipt["candidate_files"])
            receipt["round14a_b_resident"] = {
                "status": "INSTALLED_FROZEN_FPLUS1_INCREMENTAL_RESIDENCY",
                "strategy": strategy, **changed,
                "source_binding_sha256": sha(args.output / "ROUND14A_B_RESIDENT_SOURCE_BINDING.json"),
            }
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        finally:
            sys.path.remove(str(args.output / "source/tools"))

    # Both arms receive the immutable runtime module so their snapshot trees
    # stay structurally comparable.  Rebind the overlay receipt even when the
    # baseline leaves that module dormant.
    overlay = args.output / "overlay-python"
    sys.path.insert(0, str(args.output / "source/tools"))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        receipt_path = overlay / "round9-candidate-overlay.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["candidate_files"] = tree_files(overlay)
        receipt["candidate_tree_sha256"] = tree_digest(receipt["candidate_files"])
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        sys.path.remove(str(args.output / "source/tools"))

    launcher = args.output / "source/tools/launch_hybrid.py"
    launcher.write_text(patch_launcher(launcher.read_text(), args.version, strategy, label))
    monitor = args.output / "source/tools/monitor_owned_server.py"
    monitor_source = monitor.read_text()
    monitor_source = replace_once(
        monitor_source,
        r"|kt-round14a-a-gdn-(?:smoke|abba)-(?:baseline|candidate)-[sab]-v(?:292|293|294|295|296|297))",
        r"|kt-round14a-a-gdn-(?:smoke|abba)-(?:baseline|candidate)-[sab]-v(?:292|293|294|295|296|297)"
        r"|kt-round14a-b-resident-(?:baseline|fplus1)-[ab]-v(?:306|307|308|309|310|311|312|313|314|315|316|317|318|319|320|321))",
        "Round14a B owned-container allowlist")
    compile(monitor_source, str(monitor), "exec")
    monitor.write_text(monitor_source)
    budget = {
        "schema": 1, "stage": "ROUND14A_B_INCREMENTAL_RESIDENCY_REQUEST",
        "version": args.version, "strategy": strategy, "block": block,
        "model_starts_limit": 1, "model_starts_used": 0, "starts": [],
        "generation_limit": 20, "abort_limit": 0, "automatic_retries_limit": 0,
    }
    (args.output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    contract = {
        "schema": 1, "status": "FROZEN_NOT_RUN", "stage": budget["stage"],
        "version": args.version, "strategy": strategy, "block": block,
        "container": f"kt-round14a-b-resident-{label}-v{args.version}",
        "evidence": f"20260928-round14a-b-resident-{label}-v{args.version}",
        "generation_limit": 20, "abort_limit": 0, "profiler": "OFF",
        "driver_command": [
            "{DRIVER_ROOT}/round14a_resident_model_driver.py",
            "--version", str(args.version), "--out", "/out",
            "--fixture", "{DRIVER_ROOT}/round10-t1-fixture.json",
            "--profiles", "{DRIVER_ROOT}/ROUND14A_RESIDENT_PROFILES.json",
        ],
        "raw_result": "round14a-resident-model-raw.json",
        "final_result": "round14a-resident-model-result.json",
        "driver_pass_status": "PASS_ROUND14A_B_RESIDENT_MODEL_DRIVER_RAW",
        "process_pass_status": "PASS_ROUND14A_B_RESIDENT_MODEL_PROCESS",
        "source_change": changed,
        "source_binding": source_delta,
        "parent_snapshot": str(args.parent),
        "parent_snapshot_sha256": sha(args.parent / "round14a-source-sha256.json"),
    }
    (args.output / "ROUND14A_B_RESIDENT_MODEL_CONTRACT.json").write_text(
        json.dumps(contract, indent=2) + "\n")
    (args.output / "round14a-source-sha256.json").write_text(
        json.dumps(source_manifest(args.output), indent=2) + "\n")
    build_receipt = {
        "schema": 1, "status": "PASS_ROUND14A_B_RESIDENT_MODEL_BUILD",
        "version": args.version, "strategy": strategy, "block": block,
        "snapshot": str(args.output), "snapshot_manifest_sha256": sha(args.output / "round14a-source-sha256.json"),
        "contract_sha256": sha(args.output / "ROUND14A_B_RESIDENT_MODEL_CONTRACT.json"),
    }
    args.build_receipt.write_text(json.dumps(build_receipt, indent=2) + "\n")
    for path in args.output.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)
    print(json.dumps(build_receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=tuple(ROLES), required=True)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
