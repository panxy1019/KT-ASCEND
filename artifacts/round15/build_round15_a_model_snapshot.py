#!/usr/bin/env python3
"""Build one immutable Round15 QSA quality/performance screen snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


ROLES = {
    341: ("screen", "baseline", "S"),
    342: ("screen", "candidate", "S"),
}
STAGE = "ROUND15_A_QSA_FULL_MODEL"
PARENT_QSA_SHA256 = "0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35"
PARENT_QWEN_SHA256 = "1986d8a880fe1503848eadf02a8ed572d543eac967fcb583b6f67d42250f3545"
MUTABLE = {
    "round14a-source-sha256.json",
    "round15-a-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise RuntimeError(f"Round15 A {label} precondition mismatch")
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


def contract(version, *, frozen_qwen_sha256=None):
    kind, implementation, block = ROLES[version]
    lower = block.lower()
    generations = 14
    aborts = 1
    return {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": STAGE,
        "version": version,
        "kind": kind,
        "implementation": implementation,
        "block": block,
        "container": f"kt-round15-a-qsa-{kind}-{implementation}-{lower}-v{version}",
        "evidence": f"20260928-round15-a-qsa-{kind}-{implementation}-{lower}-v{version}",
        "parent": "ROUND15_P0_V338_PROMOTED_S0",
        "candidate": "FULL_SET_CAUSAL_SHORT_CONTEXT",
        "candidate_default": "OFF",
        "request_gate": "DIRECTIONAL_W1_W2_TWO_PROCESS_SCREEN",
        "request_evidence_class": "DIRECTIONAL_NOT_FOUR_PROCESS_ABBA",
        "threshold": 129,
        "profiler": "OFF",
        "graph": False,
        "generation_limit": generations,
        "abort_limit": aborts,
        "automatic_retries": 0,
        "raw_result": "round15-a-model-raw.json",
        "final_result": "round15-a-model-final.json",
        "driver_pass_status": "PASS_ROUND15_A_QSA_MODEL_DRIVER_RAW",
        "process_pass_status": "PASS_ROUND15_A_QSA_MODEL_PROCESS",
        "frozen_qwen_sha256": frozen_qwen_sha256,
        "driver_command": [
            "{DRIVER_ROOT}/round15_a_model_driver.py",
            "--version", str(version),
            "--out", "/out",
            "--fixture10", "{DRIVER_ROOT}/round10-t1-fixture.json",
        ],
    }


def patch_qsa(source):
    source = replace_once(
        source,
        "import math\nimport torch\n",
        "import json\nimport math\nimport os\nfrom pathlib import Path\nimport time\nimport torch\n\n"
        "_ROUND15_QSA_FULL_SET_SEEN = set()\n\n"
        "def _round15_record_full_set(layer_id, logical_position, completed_groups):\n"
        "    path = os.environ.get(\"ROUND15_QSA_SELECTION_EVENTS\")\n"
        "    if path is None or layer_id in _ROUND15_QSA_FULL_SET_SEEN:\n"
        "        return\n"
        "    _ROUND15_QSA_FULL_SET_SEEN.add(layer_id)\n"
        "    with Path(path).open(\"a\") as stream:\n"
        "        stream.write(json.dumps({\"status\": \"ROUND15_QSA_FULL_SET_HIT\",\n"
        "            \"time\": time.time(), \"layer\": int(layer_id),\n"
        "            \"logical_position\": int(logical_position),\n"
        "            \"completed_groups\": int(completed_groups),\n"
        "            \"implementation\": \"FULL_SET_CAUSAL_SHORT_CONTEXT\"}) + \"\\n\")\n",
        "QSA imports/receipt recorder",
    )
    source = replace_once(
        source,
        "def select_tokens(index_query,compressed_keys,logical_positions,*,ratio=4,topk_blocks=512):",
        "def select_tokens(index_query,compressed_keys,logical_positions,*,ratio=4,topk_blocks=512,layer_id=None):",
        "QSA selector signature",
    )
    anchor = (
        "    result=torch.full((index_query.shape[0],width),-1,dtype=torch.long,device=index_query.device)\n"
        "    k=min(topk_blocks,compressed_keys.shape[0])\n"
    )
    replacement = (
        "    result=torch.full((index_query.shape[0],width),-1,dtype=torch.long,device=index_query.device)\n"
        "    if (os.environ.get(\"ROUND15_QSA_FULL_SET_ENABLE\", \"0\") == \"1\"\n"
        "            and layer_id is not None and index_query.shape[0] == 1\n"
        "            and compressed_keys.shape[0] <= topk_blocks):\n"
        "        ids=torch.arange(width,dtype=torch.long,device=index_query.device)[None,:]\n"
        "        result=torch.where(ids<=logical_positions.long()[:,None],ids,result)\n"
        "        _round15_record_full_set(layer_id,logical_positions[0],compressed_keys.shape[0])\n"
        "        return result\n"
        "    k=min(topk_blocks,compressed_keys.shape[0])\n"
    )
    source = replace_once(source, anchor, replacement, "QSA full-set branch")
    compile(source, "qsa.py", "exec")
    return source


def patch_qwen(source):
    old = (
        "            topk_blocks=self.token_topk // self.ratio,\n"
        "        )\n"
    )
    new = (
        "            topk_blocks=self.token_topk // self.ratio,\n"
        "            layer_id=self.layer_id,\n"
        "        )\n"
    )
    source = replace_once(source, old, new, "QSA layer identity")
    compile(source, "qwen4_exp.py", "exec")
    return source


def patch_guard(source):
    marker = ")', name) is not None"
    index = source.rfind(marker)
    if index < 0:
        raise RuntimeError("Round15 A model guard allowlist precondition mismatch")
    pattern = (
        "|kt-round15-a-qsa-screen-(?:baseline|candidate)-s-"
        "v(?:341|342)"
    )
    return source[:index] + pattern + source[index:]


def patch_launcher(source, value):
    source = replace_once(
        source,
        "if a.version != 295 or a.cost_mode != 'OFF':",
        f"if a.version != {value['version']} or a.cost_mode != 'OFF':",
        "launcher version",
    )
    source = replace_once(
        source,
        "name='kt-round14a-a-gdn-abba-candidate-a-v295';out=root/'evidence'/'20260927-round14a-a-gdn-abba-candidate-a-v295'",
        f"name='{value['container']}';out=root/'evidence'/'{value['evidence']}'",
        "launcher identity",
    )
    env = "args+=['-e','ROUND14_GDN_BACKEND_ENABLE=1']"
    replacement = env + "\nargs+=['-e','ROUND15_QSA_SELECTION_EVENTS=/out/round15-qsa-selection.jsonl']"
    if value["implementation"] == "candidate":
        replacement += "\nargs+=['-e','ROUND15_QSA_FULL_SET_ENABLE=1']"
    source = replace_once(source, env, replacement, "candidate environment")
    source = replace_once(
        source,
        "if budget.get('stage')!='ROUND14A_A_GDN_FULL_MODEL':raise RuntimeError('wrong Round14a GDN model budget')",
        "if budget.get('stage')!='ROUND15_A_QSA_FULL_MODEL':raise RuntimeError('wrong Round15 A model budget')",
        "budget stage",
    )
    compile(source, "launch_hybrid.py", "exec")
    return source


def source_binding(parent_proof, qsa_sha, qwen_sha, component_result):
    return {
        "schema": 1,
        "status": "PASS_ROUND15_A_QSA_SOURCE_BINDING",
        "candidate": "FULL_SET_CAUSAL_SHORT_CONTEXT",
        "qualified_qsa_sha256": parent_proof["qualified_qsa_sha256"],
        "candidate_qsa_sha256": qsa_sha,
        "candidate_qwen4_exp_sha256": qwen_sha,
        "component_result_sha256": sha256(component_result),
        "parent_round14a_binding_sha256": hashlib.sha256(
            json.dumps(parent_proof, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "candidate_context_reference_runtime_sha256": parent_proof[
            "candidate_context_reference_runtime_sha256"
        ],
        "candidate_context_reference_binding_sha256": parent_proof[
            "candidate_context_reference_binding_sha256"
        ],
        "scope": "DECODE_BATCH1_SHORT_CONTEXT_FULL_CAUSAL_SET_DEFAULT_OFF_GRAPH_OFF",
    }


def patch_gate(source, proof):
    start = source.index("APPROVED_QSA_BINDING = ")
    end = source.index("\n\n\ndef _bind_qsa_source", start)
    replacement = "APPROVED_QSA_BINDING = " + repr(proof)
    patched = source[:start] + replacement + source[end:]
    compile(patched, "context_reference_gate.py", "exec")
    return patched


def update_overlay_receipt(root):
    helper_path = root / "source/tools/install_candidate_overlay.py"
    spec = importlib.util.spec_from_file_location("_round15_a_overlay", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    overlay = root / "overlay-python"
    receipt_path = overlay / "round9-candidate-overlay.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["candidate_files"] = helper.tree_files(overlay)
    receipt["candidate_tree_sha256"] = helper.tree_digest(receipt["candidate_files"])
    receipt["round15_a_qsa"] = {
        "status": "INSTALLED_DEFAULT_OFF_FULL_SET_CAUSAL_SHORT_CONTEXT",
        "qsa_sha256": sha256(
            overlay / "sglang/srt/layers/round9/qsa.py"
        ),
        "qwen_sha256": sha256(overlay / "sglang/srt/models/qwen4_exp.py"),
        "gate_sha256": sha256(
            overlay / "sglang/srt/layers/round9/context_reference_gate.py"
        ),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


def build(*, parent, output, version, scripts_root, component_result):
    parent = Path(parent).resolve(strict=True)
    output = Path(output)
    scripts_root = Path(scripts_root).resolve(strict=True)
    component_result = Path(component_result).resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    if sha256(parent / "overlay-python/sglang/srt/layers/round9/qsa.py") != PARENT_QSA_SHA256:
        raise RuntimeError("Round15 A model parent QSA source changed")
    if sha256(parent / "overlay-python/sglang/srt/models/qwen4_exp.py") != PARENT_QWEN_SHA256:
        raise RuntimeError("Round15 A model parent Qwen source changed")
    component = json.loads(component_result.read_text())
    if (
        component.get("status") != "PASS_ROUND15_A1_QSA_TWO_BLOCK_COMPONENT_GATE"
        or component.get("request_smoke_gate_open") is not True
    ):
        raise RuntimeError("Round15 A model build lacks the passed two-block component gate")
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    for stale in (
        "round14a-source-sha256.json",
        "ROUND14A_A_GDN_MODEL_CONTRACT.json",
        "host-controller.json",
        "execution-budget.lock",
    ):
        path = output / stale
        if path.exists():
            path.unlink()

    qsa_paths = (
        output / "source/runtime/qsa.py",
        output / "overlay-python/sglang/srt/layers/round9/qsa.py",
    )
    for path in qsa_paths:
        path.write_text(patch_qsa(path.read_text()))
    qwen = output / "overlay-python/sglang/srt/models/qwen4_exp.py"
    qwen.write_text(patch_qwen(qwen.read_text()))
    qsa_sha, qwen_sha = sha256(qsa_paths[1]), sha256(qwen)

    parent_proof_path = output / "ROUND14A_QSA_GDN_SOURCE_BINDING.json"
    parent_proof = json.loads(parent_proof_path.read_text())
    proof = source_binding(parent_proof, qsa_sha, qwen_sha, component_result)
    parent_proof_path.write_text(json.dumps(proof, indent=2) + "\n")
    for gate in (
        output / "source/runtime/context_reference_gate.py",
        output / "overlay-python/sglang/srt/layers/round9/context_reference_gate.py",
    ):
        gate.write_text(patch_gate(gate.read_text(), proof))

    value = contract(version, frozen_qwen_sha256=qwen_sha)
    launcher = output / "source/tools/launch_hybrid.py"
    launcher.write_text(patch_launcher(launcher.read_text(), value))
    monitor = output / "source/tools/monitor_owned_server.py"
    monitor.write_text(patch_guard(monitor.read_text()))
    compile(monitor.read_text(), str(monitor), "exec")
    for name in (
        "round15_a_model_driver.py",
        "round15_a_model_supervisor.py",
        "round15_a_model_controller.py",
        "round15_a_model_preflight.py",
        "round15_p0_control.py",
    ):
        shutil.copy2(scripts_root / name, output / name)
    (output / "ROUND15_A_MODEL_CONTRACT.json").write_text(
        json.dumps(value, indent=2) + "\n"
    )
    budget = {
        "schema": 1,
        "stage": STAGE,
        "version": version,
        "kind": value["kind"],
        "implementation": value["implementation"],
        "block": value["block"],
        "model_starts_limit": 1,
        "model_starts_used": 0,
        "starts": [],
        "generation_limit": value["generation_limit"],
        "abort_limit": value["abort_limit"],
        "automatic_retries_limit": 0,
    }
    (output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    update_overlay_receipt(output)
    manifest = tree(output)
    (output / "round15-a-source-sha256.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return {
        "schema": 1,
        "status": "PASS_ROUND15_A_MODEL_BUILD",
        "version": version,
        "kind": value["kind"],
        "implementation": value["implementation"],
        "block": value["block"],
        "output": str(output),
        "qsa_sha256": qsa_sha,
        "qwen_sha256": qwen_sha,
        "binding_sha256": sha256(parent_proof_path),
        "manifest_sha256": sha256(output / "round15-a-source-sha256.json"),
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
    result = build(
        parent=args.parent,
        output=args.output,
        version=args.version,
        scripts_root=args.scripts_root,
        component_result=args.component_result,
    )
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["version"], result["implementation"], flush=True)


if __name__ == "__main__":
    main()
