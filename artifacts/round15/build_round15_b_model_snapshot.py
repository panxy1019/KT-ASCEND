#!/usr/bin/env python3
"""Build the two immutable Round15 B directional request snapshots."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


ROLES = {354: ("screen", "baseline", "S"), 360: ("screen", "candidate", "S")}
STAGE = "ROUND15_B_GRAPH_FULL_MODEL"
PARENT_KTEP_SHA256 = "77dfbbab73431111e72b6ac2683e4f4ecfa617280436e9ce55a3d7f89177e46b"
PARENT_QWEN_SHA256 = "82dcde3e6209911912f1f82b21d9b96cca1fead019ae65b435264d100cd67ab4"
PARENT_MANIFEST_SHA256 = "e328e2594e219a17a6bfebc8404439a84dc8aad600805b944b0ae7caed25add3"
MUTABLE = {
    "round15-a-source-sha256.json", "round15-b-source-sha256.json",
    "execution-budget.json", "execution-budget.lock", "host-controller.json",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise RuntimeError(f"Round15 B {label} precondition mismatch")
    return source.replace(old, new)


def tree(root):
    root = Path(root)
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
        and path.suffix != ".pyc" and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def contract(version, binding):
    kind, implementation, block = ROLES[version]
    return {
        "schema": 1, "status": "FROZEN_NOT_RUN", "stage": STAGE,
        "version": version, "kind": kind, "implementation": implementation,
        "block": block,
        "container": f"kt-round15-b-graph-{kind}-{implementation}-s-v{version}",
        "evidence": f"20260928-round15-b-graph-{kind}-{implementation}-s-v{version}",
        "parent": "ROUND15_A_V341_S0_DEFAULTS",
        "candidate": "DECODE_BS1_RESIDENT_MOE_NPU_SEGMENT_GRAPH_TOPK_IDS_INT32",
        "candidate_default": "OFF",
        "request_gate": "DIRECTIONAL_W1_W2_TWO_PROCESS_SCREEN",
        "request_evidence_class": "DIRECTIONAL_NOT_FOUR_PROCESS_ABBA",
        "startup_cache_hook": {
            "scope": "MODEL_CHECKPOINT_ONLY_131_FILES",
            "trigger": "AFTER_LOAD_AND_GLOBAL_FD_MMAP_REFERENCE_COUNT_ZERO",
            "operation": "POSIX_FADV_DONTNEED_WITH_FINCORE",
            "bank_files": "EXCLUDED_WHILE_MAPPED",
            "global_drop_caches": False,
        },
        "threshold": 129, "profiler": "OFF", "graph": implementation == "candidate",
        "generation_limit": 14, "abort_limit": 1, "automatic_retries": 0,
        "raw_result": "round15-b-model-raw.json",
        "final_result": "round15-b-model-final.json",
        "driver_pass_status": "PASS_ROUND15_B_GRAPH_MODEL_DRIVER_RAW",
        "process_pass_status": "PASS_ROUND15_B_GRAPH_MODEL_PROCESS",
        "source_binding_digest": hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "driver_command": [
            "{DRIVER_ROOT}/round15_b_model_driver.py", "--version", str(version),
            "--out", "/out", "--fixture10", "{DRIVER_ROOT}/round10-t1-fixture.json",
        ],
    }


def patch_ktep(source):
    old = (
        "        else:\n"
        "            gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)\n"
        "            output = gpu_combine_input.hidden_states\n"
    )
    new = (
        "        else:\n"
        "            from sglang.srt.layers.moe.round15_b_resident_graph import apply_resident_moe_graph\n"
        "            _round15_graph_used, gpu_combine_input = apply_resident_moe_graph(\n"
        "                self, layer, masked_dispatch_output\n"
        "            )\n"
        "            if not _round15_graph_used:\n"
        "                gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)\n"
        "            output = gpu_combine_input.hidden_states\n"
    )
    result = replace_once(source, old, new, "KTEP graph call site")
    compile(result, "kt_ep_wrapper.py", "exec")
    return result


def patch_qwen(source):
    old = (
        "        hidden, residual = self.mlp_hyper_connection.mix(hidden)\n"
        "        hidden = self.mlp(hidden, forward_batch, False)\n"
        "        hidden = self.mlp_hyper_connection.combine(hidden, residual)\n"
    )
    new = (
        "        hidden, residual = self.mlp_hyper_connection.mix(hidden)\n"
        "        _round15_method = getattr(getattr(self.mlp, \"experts\", None), \"quant_method\", None)\n"
        "        _round15_missing = object()\n"
        "        _round15_previous = getattr(_round15_method, \"_round15_forward_mode\", _round15_missing)\n"
        "        if _round15_method is not None:\n"
        "            _round15_method._round15_forward_mode = forward_batch.forward_mode\n"
        "        try:\n"
        "            hidden = self.mlp(hidden, forward_batch, False)\n"
        "        finally:\n"
        "            if _round15_method is not None:\n"
        "                if _round15_previous is _round15_missing:\n"
        "                    delattr(_round15_method, \"_round15_forward_mode\")\n"
        "                else:\n"
        "                    _round15_method._round15_forward_mode = _round15_previous\n"
        "        hidden = self.mlp_hyper_connection.combine(hidden, residual)\n"
    )
    result = replace_once(source, old, new, "explicit forward-mode metadata")
    compile(result, "qwen4_exp.py", "exec")
    return result


def patch_qsa_gate(source, proof):
    start = source.index("APPROVED_QSA_BINDING = ")
    end = source.index("\n\n\ndef _bind_qsa_source", start)
    result = source[:start] + "APPROVED_QSA_BINDING = " + repr(proof) + source[end:]
    compile(result, "context_reference_gate.py", "exec")
    return result


def patch_monitor(source):
    marker = ")', name) is not None"
    index = source.rfind(marker)
    if index < 0:
        raise RuntimeError("Round15 B monitor allowlist precondition mismatch")
    pattern = "|kt-round15-b-graph-screen-(?:baseline|candidate)-s-v(?:354|360)"
    return source[:index] + pattern + source[index:]


def patch_launcher(source, value):
    source = replace_once(source, "if a.version != 341 or a.cost_mode != 'OFF':",
                          f"if a.version != {value['version']} or a.cost_mode != 'OFF':",
                          "launcher version")
    source = replace_once(
        source,
        "name='kt-round15-a-qsa-screen-baseline-s-v341';out=root/'evidence'/'20260928-round15-a-qsa-screen-baseline-s-v341'",
        f"name='{value['container']}';out=root/'evidence'/'{value['evidence']}'",
        "launcher identity",
    )
    source = replace_once(
        source,
        "if budget.get('stage')!='ROUND15_A_QSA_FULL_MODEL':raise RuntimeError('wrong Round15 A model budget')",
        "if budget.get('stage')!='ROUND15_B_GRAPH_FULL_MODEL':raise RuntimeError('wrong Round15 B model budget')",
        "budget stage",
    )
    env = "args+=['-e','ROUND14_GDN_BACKEND_ENABLE=1']"
    replacement = env + "\nargs+=['-e','ROUND15_B_GRAPH_EVENTS=/out/round15-b-graph-events.jsonl']"
    if value["implementation"] == "candidate":
        replacement += "\nargs+=['-e','ROUND15_B_RESIDENT_GRAPH_ENABLE=1']"
    source = replace_once(source, env, replacement, "graph environment")
    anchor = "sglang=a.overlay_python.resolve(strict=True)\n"
    check = (
        anchor
        + "round15_b_binding=json.loads((source.parent/'ROUND15_B_GRAPH_SOURCE_BINDING.json').read_text())\n"
        + "for rel,key in ((\"sglang/srt/layers/moe/kt_ep_wrapper.py\",\"ktep_sha256\"),"
          "(\"sglang/srt/models/qwen4_exp.py\",\"qwen_sha256\"),"
          "(\"sglang/srt/layers/moe/round15_b_resident_graph.py\",\"graph_helper_sha256\"),"
          "(\"sglang/srt/layers/moe/round15_b_segment_policy.py\",\"policy_sha256\")):\n"
        + " if digest(sglang/rel)!=round15_b_binding[key]:raise RuntimeError('Round15 B graph source binding mismatch')\n"
    )
    source = replace_once(source, anchor, check, "runtime source gate")
    old_backend_gate = (
        "  for rel,sha in backend_source.items():\n"
        "   if rel.startswith(('sglang/srt/layers/moe/','sglang/srt/layers/quantization/')):\n"
        "    if digest(sglang/rel)!=sha:raise RuntimeError('layerwise MoE backend differs from device gate')\n"
    )
    new_backend_gate = (
        "  for rel,sha in backend_source.items():\n"
        "   if rel.startswith(('sglang/srt/layers/moe/','sglang/srt/layers/quantization/')):\n"
        "    actual=digest(sglang/rel)\n"
        "    if (rel=='sglang/srt/layers/moe/kt_ep_wrapper.py' and\n"
        "        actual==round15_b_binding['ktep_sha256']):\n"
        "     ROUND15_B_APPROVED_KTEP_SOURCE=True\n"
        "     continue\n"
        "    if actual!=sha:raise RuntimeError('layerwise MoE backend differs from device gate')\n"
    )
    source = replace_once(source, old_backend_gate, new_backend_gate, "exact KTEP source-gate tuple")
    compile(source, "launch_hybrid.py", "exec")
    return source


def update_overlay_receipt(root, binding):
    helper_path = root / "source/tools/install_candidate_overlay.py"
    spec = importlib.util.spec_from_file_location("_round15_b_overlay", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    overlay = root / "overlay-python"
    receipt_path = overlay / "round9-candidate-overlay.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["round15_b_graph"] = {
        "status": "INSTALLED_DEFAULT_OFF_DECODE_BS1_RESIDENT_MOE_GRAPH", **binding,
    }
    receipt["candidate_files"] = helper.tree_files(overlay)
    receipt["candidate_tree_sha256"] = helper.tree_digest(receipt["candidate_files"])
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


def build(parent, output, version, scripts_root, component_result):
    parent = Path(parent).resolve(strict=True)
    output = Path(output)
    scripts_root = Path(scripts_root).resolve(strict=True)
    component_result = Path(component_result).resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    ktep_rel = Path("overlay-python/sglang/srt/layers/moe/kt_ep_wrapper.py")
    qwen_rel = Path("overlay-python/sglang/srt/models/qwen4_exp.py")
    if sha256(parent / "round15-a-source-sha256.json") != PARENT_MANIFEST_SHA256:
        raise RuntimeError("Round15 B parent manifest changed")
    if sha256(parent / ktep_rel) != PARENT_KTEP_SHA256 or sha256(parent / qwen_rel) != PARENT_QWEN_SHA256:
        raise RuntimeError("Round15 B parent runtime source changed")
    component = json.loads(component_result.read_text())
    if component.get("status") != "PASS_ROUND15_B_GRAPH_COMPONENT_PRODUCTION_INT32_GATE_DIRECTIONAL_MODEL_OPEN":
        raise RuntimeError("Round15 B component gate is not open")

    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    for stale in ("host-controller.json", "execution-budget.lock"):
        path = output / stale
        if path.exists():
            path.unlink()

    graph_rel = Path("overlay-python/sglang/srt/layers/moe/round15_b_resident_graph.py")
    policy_rel = Path("overlay-python/sglang/srt/layers/moe/round15_b_segment_policy.py")
    shutil.copy2(scripts_root / "round15_b_resident_graph.py", output / graph_rel)
    shutil.copy2(scripts_root / "round15_b_segment_policy.py", output / policy_rel)
    implementation = ROLES[version][1]
    if implementation == "candidate":
        (output / ktep_rel).write_text(patch_ktep((output / ktep_rel).read_text()))
        (output / qwen_rel).write_text(patch_qwen((output / qwen_rel).read_text()))
        qsa_proof_path = output / "ROUND14A_QSA_GDN_SOURCE_BINDING.json"
        qsa_proof = json.loads(qsa_proof_path.read_text())
        qsa_proof["status"] = "PASS_ROUND15_B_GRAPH_QSA_SOURCE_BINDING"
        qsa_proof["candidate_qwen4_exp_sha256"] = sha256(output / qwen_rel)
        qsa_proof["scope"] = "ROUND15_A_QSA_DEFAULT_OFF_WITH_DECODE_BS1_RESIDENT_MOE_GRAPH"
        qsa_proof_path.write_text(json.dumps(qsa_proof, indent=2) + "\n")
        for gate in (
            output / "source/runtime/context_reference_gate.py",
            output / "overlay-python/sglang/srt/layers/round9/context_reference_gate.py",
        ):
            gate.write_text(patch_qsa_gate(gate.read_text(), qsa_proof))

    binding = {
        "schema": 1, "status": "PASS_ROUND15_B_GRAPH_SOURCE_BINDING",
        "implementation": implementation,
        "candidate": "DECODE_BS1_RESIDENT_MOE_NPU_SEGMENT_GRAPH_TOPK_IDS_INT32",
        "candidate_default": "OFF", "ktep_sha256": sha256(output / ktep_rel),
        "qwen_sha256": sha256(output / qwen_rel),
        "graph_helper_sha256": sha256(output / graph_rel),
        "policy_sha256": sha256(output / policy_rel),
        "component_result_sha256": sha256(component_result),
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "scope": "SERIAL_TP1_Q32_BF16_NZ29_DECODE_BATCH1_TOPK_IDS_INT32_ONLY",
    }
    (output / "ROUND15_B_GRAPH_SOURCE_BINDING.json").write_text(json.dumps(binding, indent=2) + "\n")
    value = contract(version, binding)
    launcher = output / "source/tools/launch_hybrid.py"
    launcher.write_text(patch_launcher(launcher.read_text(), value))
    monitor = output / "source/tools/monitor_owned_server.py"
    monitor.write_text(patch_monitor(monitor.read_text()))
    compile(monitor.read_text(), str(monitor), "exec")
    for name in (
        "round15_b_model_driver.py", "round15_b_model_supervisor.py",
        "round15_b_model_controller.py", "round15_b_model_preflight.py",
        "round15_p0_control.py",
    ):
        shutil.copy2(scripts_root / name, output / name)
    (output / "ROUND15_B_MODEL_CONTRACT.json").write_text(json.dumps(value, indent=2) + "\n")
    budget = {
        "schema": 1, "stage": STAGE, "version": version, "kind": value["kind"],
        "implementation": implementation, "block": value["block"],
        "model_starts_limit": 1, "model_starts_used": 0, "starts": [],
        "generation_limit": 14, "abort_limit": 1, "automatic_retries_limit": 0,
    }
    (output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    update_overlay_receipt(output, binding)
    (output / "round15-b-source-sha256.json").write_text(json.dumps(tree(output), indent=2) + "\n")
    return {
        "schema": 1, "status": "PASS_ROUND15_B_MODEL_BUILD", "version": version,
        "implementation": implementation, "output": str(output),
        "binding_sha256": sha256(output / "ROUND15_B_GRAPH_SOURCE_BINDING.json"),
        "manifest_sha256": sha256(output / "round15-b-source-sha256.json"),
        "component_result_sha256": sha256(component_result),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=sorted(ROLES), required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--component-result", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.parent, args.output, args.version, args.scripts_root, args.component_result)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["version"], result["implementation"], flush=True)


if __name__ == "__main__":
    main()
