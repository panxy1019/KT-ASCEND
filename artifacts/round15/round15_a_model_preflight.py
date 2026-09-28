#!/usr/bin/env python3
"""No-device same-image preflight for a Round15 QSA request screen."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


ROLES = {
    341: ("screen", "baseline", "S"),
    342: ("screen", "candidate", "S"),
}
MUTABLE = {
    "round15-a-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def approved_binding(path):
    module = ast.parse(Path(path).read_text())
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "APPROVED_QSA_BINDING"
                   for target in statement.targets):
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
        raise RuntimeError(f"Round15 A model preflight exposed NPU devices: {devices}")

    manifest = root / "round15-a-source-sha256.json"
    if tree(root) != json.loads(manifest.read_text()):
        raise RuntimeError("Round15 A model immutable manifest mismatch")
    contract = json.loads((root / "ROUND15_A_MODEL_CONTRACT.json").read_text())
    version = contract["version"]
    role = ROLES.get(version)
    if role != (contract["kind"], contract["implementation"], contract["block"]):
        raise RuntimeError("Round15 A model role mismatch")
    if (
        contract.get("generation_limit") != 14
        or contract.get("abort_limit") != 1
        or contract.get("automatic_retries") != 0
        or contract.get("profiler") != "OFF"
        or contract.get("graph") is not False
        or contract.get("threshold") != 129
        or contract.get("request_evidence_class") != "DIRECTIONAL_NOT_FOUR_PROCESS_ABBA"
    ):
        raise RuntimeError("Round15 A model budget/profile mismatch")
    budget = json.loads((root / "execution-budget.json").read_text())
    if (
        budget.get("stage") != "ROUND15_A_QSA_FULL_MODEL"
        or budget.get("generation_limit") != 14
        or budget.get("abort_limit") != 1
        or budget.get("automatic_retries_limit") != 0
    ):
        raise RuntimeError("Round15 A model execution budget mismatch")

    qsa = root / "overlay-python/sglang/srt/layers/round9/qsa.py"
    runtime_qsa = root / "source/runtime/qsa.py"
    qwen = root / "overlay-python/sglang/srt/models/qwen4_exp.py"
    binding_path = root / "ROUND14A_QSA_GDN_SOURCE_BINDING.json"
    binding = json.loads(binding_path.read_text())
    if (
        sha256(qsa) != binding.get("candidate_qsa_sha256")
        or sha256(runtime_qsa) != sha256(qsa)
        or sha256(qwen) != binding.get("candidate_qwen4_exp_sha256")
        or sha256(qwen) != contract.get("frozen_qwen_sha256")
        or binding.get("candidate") != "FULL_SET_CAUSAL_SHORT_CONTEXT"
    ):
        raise RuntimeError("Round15 A QSA/Qwen source binding mismatch")
    for gate in (
        root / "source/runtime/context_reference_gate.py",
        root / "overlay-python/sglang/srt/layers/round9/context_reference_gate.py",
    ):
        if approved_binding(gate) != binding:
            raise RuntimeError("Round15 A runtime source gate differs from binding")
    qsa_text = qsa.read_text()
    if (
        'os.environ.get("ROUND15_QSA_FULL_SET_ENABLE", "0") == "1"' not in qsa_text
        or "index_query.shape[0] == 1" not in qsa_text
        or "compressed_keys.shape[0] <= topk_blocks" not in qsa_text
    ):
        raise RuntimeError("Round15 A exact full-set branch missing")

    launcher = (root / "source/tools/launch_hybrid.py").read_text()
    enabled = "ROUND15_QSA_FULL_SET_ENABLE=1" in launcher
    if (
        enabled != (contract["implementation"] == "candidate")
        or "ROUND15_QSA_SELECTION_EVENTS=/out/round15-qsa-selection.jsonl" not in launcher
        or "ROUND14_GDN_BACKEND_ENABLE=1" not in launcher
        or "ROUND13_ATTRIBUTION_ENABLE" in launcher
    ):
        raise RuntimeError("Round15 A launcher candidate/profiler binding mismatch")

    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files
        overlay = root / "overlay-python"
        receipt = json.loads((overlay / "round9-candidate-overlay.json").read_text())
        if receipt["candidate_tree_sha256"] != tree_digest(tree_files(overlay)):
            raise RuntimeError("Round15 A overlay receipt mismatch")
    finally:
        sys.path.remove(str(tools))
    for name in (
        "round15_a_model_driver.py",
        "round15_a_model_controller.py",
        "round15_a_model_supervisor.py",
        "round15_a_model_preflight.py",
        "source/tools/launch_hybrid.py",
    ):
        compile((root / name).read_text(), name, "exec")

    result = {
        "schema": 1,
        "status": "PASS_ROUND15_A_MODEL_OFFLINE_PREFLIGHT",
        "version": version,
        "kind": role[0],
        "implementation": role[1],
        "block": role[2],
        "physical_devices_exposed": devices,
        "manifest_sha256": sha256(manifest),
        "qsa_sha256": sha256(qsa),
        "qwen_sha256": sha256(qwen),
        "candidate_enabled": enabled,
        "profiler": "OFF",
        "graph": False,
        "generation_limit": 14,
        "abort_limit": 1,
        "request_evidence_class": contract["request_evidence_class"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], version, role[1], flush=True)


if __name__ == "__main__":
    main()
