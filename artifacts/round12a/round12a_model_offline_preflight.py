#!/usr/bin/env python3
"""Fail closed on a Round12a model snapshot before loading weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROLES = {
    173: ("NON_KT", "SMOKE"),
    174: ("KT", "A"),
    175: ("NON_KT", "A"),
    176: ("NON_KT", "B"),
    177: ("KT", "B"),
}
MUTABLE = {
    "round12-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}
PROMOTED = {
    "qsa.py": "0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35",
    "qwen4_exp.py": "d4583c5abe9791b09c48d1dd8f8491a2988eb5147d45df188e0e48908c699f1f",
    "context_reference_runtime.py": "0b6e6a7fad90e3856ffdcbea233a13baaf1bb342837c68ff344579370773c028",
    "context_reference_binding.py": "97eb1b60d0ee24305ad1a052884b79a3045fa3574fcfc85e8b2c4ccfdbfc0c11",
}
TIED_OFFLOADER_SHA256 = "8b68d4012a9bd0fab3bab70ab28db1d7c09b6242af912066d2aedd3f3a3c9006"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_tree(root: Path) -> dict:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def validate_gate_sources(overlay: Path, receipt: dict) -> None:
    for name, expected in receipt["source_sha256"].items():
        path = overlay / "sglang/srt/layers/round9" / name
        if sha256(path) != expected and name != "qsa.py":
            raise RuntimeError(f"device-gated non-QSA source changed: {name}")


def run(snapshot: Path, round9_root: Path) -> dict:
    snapshot, round9_root = Path(snapshot), Path(round9_root)
    contract = json.loads((snapshot / "ROUND12A_MODEL_CONTRACT.json").read_text())
    version = int(contract["version"])
    if ROLES.get(version) != (contract["mode"], contract["block"]):
        raise RuntimeError("Round12a role/version mismatch")
    manifest_path = snapshot / "round12-source-sha256.json"
    if snapshot_tree(snapshot) != json.loads(manifest_path.read_text()):
        raise RuntimeError("Round12a snapshot manifest mismatch")
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
            raise RuntimeError("Round12a overlay receipt mismatch")
    finally:
        sys.path.remove(str(tools))

    actual_promoted = {
        "qsa.py": sha256(srt / "layers/round9/qsa.py"),
        "qwen4_exp.py": sha256(srt / "models/qwen4_exp.py"),
        "context_reference_runtime.py": sha256(
            srt / "layers/round9/context_reference_runtime.py"
        ),
        "context_reference_binding.py": sha256(
            srt / "layers/round9/context_reference_binding.py"
        ),
    }
    if actual_promoted != PROMOTED:
        raise RuntimeError("Round12a promoted QSA tuple changed")
    binding = json.loads((snapshot / "ROUND12_QSA_SOURCE_BINDING.json").read_text())
    if (
        binding.get("status") != "PASS_ROUND12_QSA_SOURCE_BINDING"
        or binding.get("candidate_qsa_sha256") != PROMOTED["qsa.py"]
        or binding.get("candidate_qwen4_exp_sha256") != PROMOTED["qwen4_exp.py"]
        or binding.get("candidate_context_reference_runtime_sha256")
        != PROMOTED["context_reference_runtime.py"]
        or binding.get("candidate_context_reference_binding_sha256")
        != PROMOTED["context_reference_binding.py"]
    ):
        raise RuntimeError("Round12a QSA binding receipt changed")

    ple = json.loads(
        (round9_root / "evidence/20260921-ple-v2/result.json").read_text()
    )
    context = json.loads(
        (round9_root / "evidence/20260921-ple-v4/result.json").read_text()
    )
    validate_gate_sources(overlay, ple)
    validate_gate_sources(overlay, context)
    source_identity = context["source_identity"]

    mode = contract["mode"]
    if mode == "KT":
        framework = json.loads((snapshot / "framework-profile.json").read_text())
        streaming = json.loads((snapshot / "streaming-profile.json").read_text())
        bank = json.loads((snapshot / "rebound/bank-receipt.json").read_text())
        if (
            framework["prefill"]["threshold_tokens"] != 129
            or framework["ple"]["mode"] != "ASYNC"
            or streaming["threshold_tokens"] != 129
            or bank["source_identity"] != source_identity
        ):
            raise RuntimeError("Round12a KT policy/source differs from v154")
        backend = "KT_C128_MXFP4_Q32"
    else:
        framework = json.loads(
            (snapshot / "nonkt-framework-profile.json").read_text()
        )
        profile = json.loads((snapshot / "nonkt-server-profile.json").read_text())
        identity = json.loads((snapshot / "ROUND12A_NONKT_IDENTITY.json").read_text())
        offloader_sha = sha256(srt / "utils/offloader.py")
        if (
            framework["prefill"] is not None
            or framework["ple"]["mode"] != "ASYNC"
            or profile["kt_weight_path"] is not None
            or profile["cpu_offload_gb"] != 233
            or profile["mem_fraction_static"] != 0.45
            or identity["source_identity"] != source_identity
            or identity["loaded_main_layer_bytes"] != 248931383320
            or identity["largest_layer_stage_bytes"] != 5253637848
            or identity["offloader_sha256"] != TIED_OFFLOADER_SHA256
            or offloader_sha != TIED_OFFLOADER_SHA256
        ):
            raise RuntimeError("Round12a non-KT capacity/source profile changed")
        required = {
            "candidate_bank.py": "bind_round12a_nonkt_source_identity",
            "ple_binding.py": "bind_round12a_nonkt_identity(runner)",
            "round12a_nonkt_identity.py": "PASS_ROUND12A_NONKT_OFFLOADER_BINDING",
            "framework_hooks.py": "PASS_ROUND12A_NONKT_OFFLOADER_BINDING",
        }
        for name, token in required.items():
            if token not in (srt / "layers/round9" / name).read_text():
                raise RuntimeError(f"non-KT source gate missing in {name}")
        launcher = (snapshot / "round12a_nonkt_launcher.py").read_text()
        for token in (
            "ROUND9_QSA_SOURCE_BINDING=",
            "ROUND12A_NONKT_IDENTITY=",
            "ROUND12A_NONKT_MODE=1",
        ):
            if token not in launcher:
                raise RuntimeError(f"non-KT launcher environment missing: {token}")
        backend = "SGLANG_TIED_AWARE_OFFLOADERV1_FULL_BF16"
    if source_identity != ple["source_identity"]:
        raise RuntimeError("Round12a source differs from PLE receipt")

    generation_limit = int(contract["generation_limit"])
    if version == 173:
        if generation_limit != 1 or (snapshot / "ROUND12A_ABBA_QUEUE.json").exists():
            raise RuntimeError("Round12a smoke budget/queue mismatch")
        queue_sha = None
    else:
        queue_path = snapshot / "ROUND12A_ABBA_QUEUE.json"
        queue = json.loads(queue_path.read_text())
        if (
            queue.get("schema") != 1
            or queue.get("status") != "FROZEN_ROUND12A_ABBA_QUEUE"
            or queue.get("source_smoke_version") != 173
            or queue.get("workloads") != ["W1"]
            or queue.get("workload_definitions")
            != {
                "W1": {
                    "case_id": "edge64",
                    "input_tokens": 64,
                    "decode_tokens": 2,
                }
            }
            or queue.get("request_wall_limit_seconds") != 1800
        ):
            raise RuntimeError("Round12a frozen ABBA queue changed")
        expected_limit = 4 * len(queue["workloads"])
        if generation_limit != expected_limit:
            raise RuntimeError("Round12a ABBA queue/budget mismatch")
        queue_sha = sha256(queue_path)
    return {
        "schema": 1,
        "status": "PASS_ROUND12A_MODEL_OFFLINE_PREFLIGHT",
        "version": version,
        "mode": mode,
        "block": contract["block"],
        "backend": backend,
        "compiled_python_files": compiled,
        "snapshot_manifest_sha256": sha256(manifest_path),
        "overlay_tree_sha256": overlay_receipt["candidate_tree_sha256"],
        "promoted_qsa_tuple": actual_promoted,
        "tied_offloader_sha256": (
            TIED_OFFLOADER_SHA256 if mode == "NON_KT" else None
        ),
        "generation_limit": generation_limit,
        "queue_sha256": queue_sha,
        "source_identity": source_identity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--round9-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.snapshot, args.round9_root)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        report["status"], report["version"], report["backend"], flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
