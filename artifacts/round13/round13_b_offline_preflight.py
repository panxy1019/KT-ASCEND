#!/usr/bin/env python3
"""Fail closed on a Round13 B model snapshot before loading weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROLES = {
    217: ("baseline", "smoke", "A"),
    218: ("candidate", "smoke", "B"),
    219: ("baseline", "request", "A"),
    220: ("candidate", "request", "A"),
    221: ("candidate", "request", "B"),
    222: ("baseline", "request", "B"),
}
MUTABLE = {
    "round13-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}
QSA_SHA256 = "0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35"
QWEN_SHA256 = "d4583c5abe9791b09c48d1dd8f8491a2988eb5147d45df188e0e48908c699f1f"
BINDING_SHA256 = "97eb1b60d0ee24305ad1a052884b79a3045fa3574fcfc85e8b2c4ccfdbfc0c11"
BASELINE_RUNTIME_SHA256 = "0b6e6a7fad90e3856ffdcbea233a13baaf1bb342837c68ff344579370773c028"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_tree(root):
    return {
        str(path.relative_to(root)): sha(path)
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def load_module(path):
    spec = importlib.util.spec_from_file_location("round13_b_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(snapshot, device_gate):
    snapshot, device_gate = Path(snapshot), Path(device_gate)
    contract = json.loads((snapshot / "ROUND13_MODEL_CONTRACT.json").read_text())
    version = int(contract["version"])
    if ROLES.get(version) != (
        contract["implementation"],
        contract["mode"],
        contract["block"],
    ):
        raise RuntimeError("Round13 B role/version mismatch")
    manifest = snapshot / "round13-source-sha256.json"
    if snapshot_tree(snapshot) != json.loads(manifest.read_text()):
        raise RuntimeError("Round13 B snapshot manifest mismatch")
    compiled = 0
    for path in snapshot.rglob("*.py"):
        if "__pycache__" not in path.parts:
            compile(path.read_bytes(), str(path), "exec")
            compiled += 1

    overlay = snapshot / "overlay-python"
    srt = overlay / "sglang/srt"
    tools = snapshot / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files

        overlay_receipt = json.loads(
            (overlay / "round9-candidate-overlay.json").read_text()
        )
        if overlay_receipt["candidate_tree_sha256"] != tree_digest(
            tree_files(overlay)
        ):
            raise RuntimeError("Round13 B overlay receipt mismatch")
    finally:
        sys.path.remove(str(tools))

    paths = {
        "qsa": srt / "layers/round9/qsa.py",
        "qwen": srt / "models/qwen4_exp.py",
        "runtime": srt / "layers/round9/context_reference_runtime.py",
        "binding": srt / "layers/round9/context_reference_binding.py",
        "gate": srt / "layers/round9/context_reference_gate.py",
    }
    actual = {name: sha(path) for name, path in paths.items()}
    if (
        actual["qsa"] != QSA_SHA256
        or actual["qwen"] != QWEN_SHA256
        or actual["binding"] != BINDING_SHA256
        or actual["runtime"] != contract["runtime_sha256"]
    ):
        raise RuntimeError("Round13 B promoted QSA tuple drifted")
    implementation = contract["implementation"]
    runtime_source = paths["runtime"].read_text()
    has_direct = "DIRECT_BF16_ROW" in runtime_source
    if (implementation == "candidate") != has_direct:
        raise RuntimeError("Round13 B implementation/runtime mismatch")
    if implementation == "baseline" and actual["runtime"] != BASELINE_RUNTIME_SHA256:
        raise RuntimeError("Round13 B baseline runtime changed")

    proof_path = snapshot / "ROUND12_QSA_SOURCE_BINDING.json"
    proof = json.loads(proof_path.read_text())
    if (
        proof.get("candidate_qsa_sha256") != actual["qsa"]
        or proof.get("candidate_qwen4_exp_sha256") != actual["qwen"]
        or proof.get("candidate_context_reference_runtime_sha256")
        != actual["runtime"]
        or proof.get("candidate_context_reference_binding_sha256")
        != actual["binding"]
    ):
        raise RuntimeError("Round13 B binding tuple mismatch")
    receipt = json.loads(device_gate.read_text())
    gate = load_module(paths["gate"])
    actual_sources = {
        name: sha(srt / "layers/round9" / name) for name in gate.SOURCE_FILES
    }
    _validated, qsa_binding = gate._bind_qsa_source(
        receipt["source_sha256"],
        actual_sources,
        binding_path=proof_path,
        binding_sha256=sha(proof_path),
        consumer_sha256=actual["qwen"],
        runtime_source_sha256=actual["runtime"],
        binding_source_sha256=actual["binding"],
    )
    if qsa_binding is None or qsa_binding.get("mode") != "APPROVED_CANDIDATE":
        raise RuntimeError("Round13 B runtime tuple did not cross source gate")

    framework = json.loads((snapshot / "framework-profile.json").read_text())
    streaming = json.loads((snapshot / "streaming-profile.json").read_text())
    budget = json.loads((snapshot / "execution-budget.json").read_text())
    if (
        framework["prefill"]["threshold_tokens"] != 513
        or framework["ple"]["mode"] != "ASYNC"
        or streaming["threshold_tokens"] != 513
        or budget.get("stage") != "ROUND13_B_QSA_DECODE"
        or budget.get("generation_limit") != contract["generation_limit"]
        or budget.get("abort_limit") != contract["abort_limit"]
    ):
        raise RuntimeError("Round13 B policy/budget mismatch")
    launcher = (snapshot / "source/tools/launch_hybrid.py").read_text()
    for token in (
        f"if a.version != {version}",
        contract["container"],
        contract["evidence"],
        "ROUND13_B_QSA_DECODE",
        "ROUND9_QSA_SOURCE_BINDING=",
    ):
        if token not in launcher:
            raise RuntimeError(f"Round13 B launcher gate missing: {token}")
    return {
        "schema": 1,
        "status": "PASS_ROUND13_B_OFFLINE_PREFLIGHT",
        "version": version,
        "implementation": implementation,
        "mode": contract["mode"],
        "block": contract["block"],
        "compiled_python_files": compiled,
        "snapshot_manifest_sha256": sha(manifest),
        "overlay_tree_sha256": overlay_receipt["candidate_tree_sha256"],
        "qsa_tuple": actual,
        "source_gate": qsa_binding,
        "generation_limit": contract["generation_limit"],
        "abort_limit": contract["abort_limit"],
        "threshold": 513,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(args.snapshot, args.device_gate)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        report["status"],
        report["version"],
        report["implementation"],
        flush=True,
    )


if __name__ == "__main__":
    main()
