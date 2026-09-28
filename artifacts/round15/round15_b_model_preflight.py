#!/usr/bin/env python3
"""CPU-only fail-closed preflight for one Round15 B model snapshot."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


ROLES = {354: ("screen", "baseline", "S"), 360: ("screen", "candidate", "S")}
COMPONENT_RESULT_SHA256 = "b30845f4f3d0e6728ec4449d898473731c820ded2f2a6e7af15d08a2d20ecc25"
MUTABLE = {
    "round15-a-source-sha256.json", "round15-b-source-sha256.json",
    "execution-budget.json", "execution-budget.lock", "host-controller.json",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(root):
    root = Path(root)
    return {
        str(path.relative_to(root)): sha256(path)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and path.suffix != ".pyc" and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def approved_binding(path):
    module = ast.parse(Path(path).read_text())
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "APPROVED_QSA_BINDING"
            for target in statement.targets
        ):
            return ast.literal_eval(statement.value)
    raise RuntimeError(f"APPROVED_QSA_BINDING missing in {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    devices = sorted(str(path) for path in Path("/dev").glob("davinci[0-9]*"))
    if devices:
        raise RuntimeError(f"Round15 B preflight exposed NPU devices: {devices}")

    manifest = root / "round15-b-source-sha256.json"
    if tree(root) != json.loads(manifest.read_text()):
        raise RuntimeError("Round15 B model immutable manifest mismatch")
    contract = json.loads((root / "ROUND15_B_MODEL_CONTRACT.json").read_text())
    role = ROLES.get(contract.get("version"))
    if role != (contract.get("kind"), contract.get("implementation"), contract.get("block")):
        raise RuntimeError("Round15 B model role mismatch")
    candidate = contract["implementation"] == "candidate"
    if (
        contract.get("generation_limit") != 14 or contract.get("abort_limit") != 1
        or contract.get("automatic_retries") != 0 or contract.get("profiler") != "OFF"
        or contract.get("graph") is not candidate or contract.get("threshold") != 129
        or contract.get("request_evidence_class") != "DIRECTIONAL_NOT_FOUR_PROCESS_ABBA"
    ):
        raise RuntimeError("Round15 B model budget/profile mismatch")
    hook = contract.get("startup_cache_hook", {})
    if (
        hook.get("scope") != "MODEL_CHECKPOINT_ONLY_131_FILES"
        or hook.get("trigger") != "AFTER_LOAD_AND_GLOBAL_FD_MMAP_REFERENCE_COUNT_ZERO"
        or hook.get("global_drop_caches") is not False
    ):
        raise RuntimeError("Round15 B bounded startup cache hook contract mismatch")
    budget = json.loads((root / "execution-budget.json").read_text())
    if (
        budget.get("stage") != "ROUND15_B_GRAPH_FULL_MODEL"
        or budget.get("generation_limit") != 14 or budget.get("abort_limit") != 1
        or budget.get("automatic_retries_limit") != 0
    ):
        raise RuntimeError("Round15 B model execution budget mismatch")

    binding = json.loads((root / "ROUND15_B_GRAPH_SOURCE_BINDING.json").read_text())
    overlay = root / "overlay-python"
    paths = {
        "ktep_sha256": overlay / "sglang/srt/layers/moe/kt_ep_wrapper.py",
        "qwen_sha256": overlay / "sglang/srt/models/qwen4_exp.py",
        "graph_helper_sha256": overlay / "sglang/srt/layers/moe/round15_b_resident_graph.py",
        "policy_sha256": overlay / "sglang/srt/layers/moe/round15_b_segment_policy.py",
    }
    if binding.get("implementation") != contract["implementation"]:
        raise RuntimeError("Round15 B binding implementation mismatch")
    for key, path in paths.items():
        if sha256(path) != binding.get(key):
            raise RuntimeError(f"Round15 B source binding mismatch: {key}")
    if binding.get("component_result_sha256") != COMPONENT_RESULT_SHA256:
        raise RuntimeError("Round15 B component result binding mismatch")
    if binding.get("scope") != "SERIAL_TP1_Q32_BF16_NZ29_DECODE_BATCH1_TOPK_IDS_INT32_ONLY":
        raise RuntimeError("Round15 B production route dtype scope mismatch")
    expected_digest = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if contract.get("source_binding_digest") != expected_digest:
        raise RuntimeError("Round15 B contract/binding digest mismatch")

    ktep_text = paths["ktep_sha256"].read_text()
    qwen_text = paths["qwen_sha256"].read_text()
    if ("apply_resident_moe_graph" in ktep_text) is not candidate:
        raise RuntimeError("Round15 B candidate KTEP call-site mismatch")
    if ("_round15_forward_mode" in qwen_text) is not candidate:
        raise RuntimeError("Round15 B explicit forward-mode binding mismatch")
    helper_text = paths["graph_helper_sha256"].read_text()
    if (
        "ROUND15_B_RESIDENT_GRAPH_ENABLE" not in helper_text
        or "max_running_requests" not in helper_text
        or "CAPTURE_AND_SELF_CHECK" not in helper_text
    ):
        raise RuntimeError("Round15 B bounded graph helper contract missing")

    qsa_proof = json.loads((root / "ROUND14A_QSA_GDN_SOURCE_BINDING.json").read_text())
    if sha256(paths["qwen_sha256"]) != qsa_proof.get("candidate_qwen4_exp_sha256"):
        raise RuntimeError("Round15 B QSA consumer binding mismatch")
    for gate in (
        root / "source/runtime/context_reference_gate.py",
        overlay / "sglang/srt/layers/round9/context_reference_gate.py",
    ):
        if approved_binding(gate) != qsa_proof:
            raise RuntimeError("Round15 B QSA gate differs from binding")

    launcher = (root / "source/tools/launch_hybrid.py").read_text()
    enabled = "ROUND15_B_RESIDENT_GRAPH_ENABLE=1" in launcher
    if (
        enabled is not candidate
        or "ROUND15_B_GRAPH_EVENTS=/out/round15-b-graph-events.jsonl" not in launcher
        or "Round15 B graph source binding mismatch" not in launcher
        or "ROUND15_B_APPROVED_KTEP_SOURCE" not in launcher
        or "ROUND13_ATTRIBUTION_ENABLE" in launcher
    ):
        raise RuntimeError("Round15 B launcher graph/profiler/source binding mismatch")

    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        receipt = json.loads((overlay / "round9-candidate-overlay.json").read_text())
        if receipt["candidate_tree_sha256"] != tree_digest(tree_files(overlay)):
            raise RuntimeError("Round15 B overlay receipt mismatch")
    finally:
        sys.path.remove(str(tools))
    for name in (
        "round15_b_model_driver.py", "round15_b_model_controller.py",
        "round15_b_model_supervisor.py", "round15_b_model_preflight.py",
        "source/tools/launch_hybrid.py",
    ):
        compile((root / name).read_text(), name, "exec")

    result = {
        "schema": 1, "status": "PASS_ROUND15_B_MODEL_OFFLINE_PREFLIGHT",
        "version": contract["version"], "implementation": contract["implementation"],
        "physical_devices_exposed": devices, "manifest_sha256": sha256(manifest),
        "binding_sha256": sha256(root / "ROUND15_B_GRAPH_SOURCE_BINDING.json"),
        "candidate_enabled": enabled, "profiler": "OFF", "graph": candidate,
        "generation_limit": 14, "abort_limit": 1,
        "request_evidence_class": contract["request_evidence_class"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["version"], result["implementation"], flush=True)


if __name__ == "__main__":
    main()
