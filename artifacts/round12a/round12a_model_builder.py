#!/usr/bin/env python3
"""Build immutable Round12a promoted-QSA KT/non-KT model snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys


ROLES = {
    173: ("NON_KT", "SMOKE"),
    174: ("KT", "A"),
    175: ("NON_KT", "A"),
    176: ("NON_KT", "B"),
    177: ("KT", "B"),
}
PROMOTED_QSA_SHA256 = "0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35"
PROMOTED_QWEN_SHA256 = "d4583c5abe9791b09c48d1dd8f8491a2988eb5147d45df188e0e48908c699f1f"
PROMOTED_CONTEXT_RUNTIME_SHA256 = "0b6e6a7fad90e3856ffdcbea233a13baaf1bb342837c68ff344579370773c028"
PROMOTED_CONTEXT_BINDING_SHA256 = "97eb1b60d0ee24305ad1a052884b79a3045fa3574fcfc85e8b2c4ccfdbfc0c11"
TIED_OFFLOADER_SHA256 = "8b68d4012a9bd0fab3bab70ab28db1d7c09b6242af912066d2aedd3f3a3c9006"
SOURCE_IDENTITY = {
    "revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
    "manifest_sha256": "62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e",
    "receipt_sha256": "7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239",
    "model_path": "/models/de4b8e4d43b917e7706784d8bb445c9af86a3540",
    "file_sizes_checked": 144,
    "verification_scope": "TRUSTED_FULL_RECEIPT_PLUS_LIVE_SIZES_AND_METADATA_HASHES",
}
MUTABLE = {
    "round12-source-sha256.json",
    "execution-budget.json",
    "execution-budget.lock",
    "host-controller.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"{label} precondition mismatch")
    return source.replace(old, new)


def source_manifest(root: Path) -> dict:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def _identity(version: int) -> tuple[str, str]:
    mode, block = ROLES[version]
    kind = mode.lower().replace("_", "")
    suffix = "smoke" if block == "SMOKE" else block.lower()
    return (
        f"kt-round12a-{kind}-{suffix}-v{version}",
        f"20260927-round12a-{kind}-{suffix}-v{version}",
    )


def patch_kt_launcher(source: str, *, version: int) -> str:
    mode, block = ROLES[version]
    if mode != "KT" or block not in {"A", "B"}:
        raise ValueError("KT launcher patch used for a non-KT role")
    name, evidence = _identity(version)
    source = replace_once(
        source,
        "if a.version != 154 or a.cost_mode != 'OFF':",
        f"if a.version != {version} or a.cost_mode != 'OFF':",
        "Round12a KT launcher version",
    )
    source = replace_once(
        source,
        "name='kt-round12-p2-candidate-v154';out=root/'evidence'/'20260926-round12-p2-candidate-v154'",
        f"name='{name}';out=root/'evidence'/'{evidence}'",
        "Round12a KT launcher identity",
    )
    source = replace_once(
        source,
        "if budget.get('stage')!='ROUND12_P2_SMOKE':raise RuntimeError('wrong Round12 model budget')",
        "if budget.get('stage')!='ROUND12A_ABBA':raise RuntimeError('wrong Round12a ABBA budget')",
        "Round12a KT launcher budget",
    )
    compile(source, "launch_hybrid.py", "exec")
    return source


def patch_candidate_bank(source: str) -> str:
    old = '''def bind_verified_banks(profile,bank_root,model):
    """Require the exact CPU verifier receipt and unchanged read-only payload."""
    path=os.environ.get('ROUND9_BANK_RECEIPT');sha=os.environ.get('ROUND9_BANK_RECEIPT_SHA256')'''
    new = '''def bind_verified_banks(profile,bank_root,model):
    """Require a verified KT bank or the frozen Round12a checkpoint identity."""
    if os.environ.get('ROUND12A_NONKT_MODE') == '1':
        from .round12a_nonkt_identity import bind_round12a_nonkt_source_identity
        return bind_round12a_nonkt_source_identity(model)
    path=os.environ.get('ROUND9_BANK_RECEIPT');sha=os.environ.get('ROUND9_BANK_RECEIPT_SHA256')'''
    source = replace_once(source, old, new, "Round12a candidate-bank branch")
    compile(source, "candidate_bank.py", "exec")
    return source


def patch_nonkt_ple_binding(source: str) -> str:
    old = '''    from .candidate_bank import bind_verified_banks
    binding=bind_verified_banks(os.environ['ROUND9_CANDIDATE_PROFILE'],args.kt_weight_path,args.model_path)'''
    new = '''    if os.environ.get('ROUND12A_NONKT_MODE') == '1':
        from .round12a_nonkt_identity import bind_round12a_nonkt_identity
        binding = bind_round12a_nonkt_identity(runner)
    else:
        from .candidate_bank import bind_verified_banks
        binding=bind_verified_banks(os.environ['ROUND9_CANDIDATE_PROFILE'],args.kt_weight_path,args.model_path)'''
    source = replace_once(source, old, new, "Round12a non-KT PLE identity")
    compile(source, "ple_binding.py", "exec")
    return source


def patch_nonkt_model_runner(source: str) -> str:
    old = "        set_offloader(create_offloader_from_server_args(server_args, dp_rank=dp_rank))"
    new = '''        if os.environ.get("ROUND12A_NONKT_MODE") == "1":
            from sglang.srt.layers.round9.round12a_nonkt_identity import install_interleave_policy
            self.round12a_nonkt_numa_policy = install_interleave_policy()
        set_offloader(create_offloader_from_server_args(server_args, dp_rank=dp_rank))'''
    source = replace_once(source, old, new, "Round12a model-runner NUMA policy")
    compile(source, "model_runner.py", "exec")
    return source


def patch_nonkt_framework_hooks(source: str) -> str:
    if source.count("PASS_ROUND12_NONKT_OFFLOADER_BINDING") != 1:
        raise ValueError("Round12a framework binding status precondition mismatch")
    if source.count("ROUND12_NONKT_MODE") != 1:
        raise ValueError("Round12a framework mode precondition mismatch")
    source = source.replace(
        "PASS_ROUND12_NONKT_OFFLOADER_BINDING",
        "PASS_ROUND12A_NONKT_OFFLOADER_BINDING",
    ).replace("ROUND12_NONKT_MODE", "ROUND12A_NONKT_MODE")
    compile(source, "framework_hooks.py", "exec")
    return source


def refresh_overlay(root: Path, *, mode: str) -> None:
    overlay = root / "overlay-python"
    tools = root / "source/tools"
    sys.path.insert(0, str(tools))
    try:
        from install_candidate_overlay import tree_digest, tree_files

        receipt_path = overlay / "round9-candidate-overlay.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["candidate_model_sha256"] = sha256(
            overlay / "sglang/srt/models/qwen4_exp.py"
        )
        receipt["candidate_files"] = tree_files(overlay)
        receipt["candidate_tree_sha256"] = tree_digest(receipt["candidate_files"])
        receipt["round12a"] = {
            "mode": mode,
            "qsa": "ROUND12_PROMOTED_CANDIDATE",
            "qsa_sha256": sha256(overlay / "sglang/srt/layers/round9/qsa.py"),
            "offloader_sha256": sha256(overlay / "sglang/srt/utils/offloader.py"),
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        sys.path.remove(str(tools))


def _generation_limit(version: int, queue: dict | None) -> int:
    if version == 173:
        return 1
    if (
        queue is None
        or queue.get("workloads") != ["W1"]
        or queue.get("workload_definitions")
        != {"W1": {"case_id": "edge64", "input_tokens": 64, "decode_tokens": 2}}
        or queue.get("request_wall_limit_seconds") != 1800
    ):
        raise ValueError("ABBA snapshots require a frozen queue")
    return 4 * len(queue["workloads"])


def prepare_common(
    root: Path,
    inputs: Path,
    *,
    version: int,
    queue_path: Path | None,
) -> None:
    mode, block = ROLES[version]
    for name in (
        "controller-F.json",
        "execution-budget.lock",
        "host-controller.json",
        "round10-source-sha256.json",
        "round12-source-sha256.json",
    ):
        path = root / name
        if path.exists():
            path.unlink()
    for source, target in (
        ("artifacts/round12a/round12a_monitor_owned.py", "source/tools/monitor_owned_server.py"),
        ("artifacts/round12a/round12a_model_driver.py", "round12a_model_driver.py"),
        ("artifacts/round12a/round12a_model_controller.py", "round12a_model_controller.py"),
        ("artifacts/round12a/round12a_model_supervisor.py", "round12a_model_supervisor.py"),
        ("artifacts/round11/round11_t4_driver.py", "round11_t4_driver.py"),
        ("artifacts/round10/round10-t1-fixture.json", "round10-t1-fixture.json"),
    ):
        shutil.copy2(inputs / source, root / target)
    queue = None
    if queue_path is not None:
        queue = json.loads(queue_path.read_text())
        shutil.copy2(queue_path, root / "ROUND12A_ABBA_QUEUE.json")
    generation_limit = _generation_limit(version, queue)
    stage = "ROUND12A_SMOKE" if version == 173 else "ROUND12A_ABBA"
    budget = {
        "schema": 1,
        "stage": stage,
        "version": version,
        "mode": mode,
        "block": block,
        "model_starts_limit": 1,
        "model_starts_used": 0,
        "starts": [],
        "generation_limit": generation_limit,
        "abort_limit": 0,
        "automatic_retries_limit": 0,
    }
    (root / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    name, evidence = _identity(version)
    command = [
        "{DRIVER_ROOT}/round12a_model_driver.py",
        "--mode",
        mode,
        "--block",
        block,
        "--version",
        str(version),
        "--out",
        "/out",
        "--fixture",
        "{DRIVER_ROOT}/round10-t1-fixture.json",
    ]
    if version != 173:
        command += ["--queue", "{DRIVER_ROOT}/ROUND12A_ABBA_QUEUE.json"]
    contract = {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": (
            "P2_FIRST_NONKT_REQUEST" if version == 173 else "P3_KT_NONKT_ABBA"
        ),
        "budget_stage": stage,
        "version": version,
        "mode": mode,
        "block": block,
        "container": name,
        "evidence": evidence,
        "parent": "ROUND12_P2_CANDIDATE_V154_PROMOTED_QSA",
        "qsa_source": "ROUND12_PROMOTED_CANDIDATE",
        "default_threshold": 129,
        "profiler": "OFF",
        "generation_limit": generation_limit,
        "abort_limit": 0,
        "automatic_retries": 0,
        "raw_result": "round12a-model-raw.json",
        "driver_pass_status": "PASS_ROUND12A_MODEL_DRIVER_RAW",
        "driver_command": command,
    }
    (root / "ROUND12A_MODEL_CONTRACT.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )


def _assert_promoted_tuple(root: Path) -> None:
    overlay = root / "overlay-python/sglang/srt"
    expected = {
        overlay / "layers/round9/qsa.py": PROMOTED_QSA_SHA256,
        overlay / "models/qwen4_exp.py": PROMOTED_QWEN_SHA256,
        overlay / "layers/round9/context_reference_runtime.py": (
            PROMOTED_CONTEXT_RUNTIME_SHA256
        ),
        overlay / "layers/round9/context_reference_binding.py": (
            PROMOTED_CONTEXT_BINDING_SHA256
        ),
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"promoted QSA tuple changed: {path}")


def prepare_kt(root: Path, *, version: int) -> None:
    launcher = root / "source/tools/launch_hybrid.py"
    launcher.write_text(patch_kt_launcher(launcher.read_text(), version=version))
    _assert_promoted_tuple(root)
    refresh_overlay(root, mode="KT")


def prepare_nonkt(root: Path, inputs: Path) -> None:
    overlay = root / "overlay-python/sglang/srt"
    round9 = overlay / "layers/round9"
    framework_hooks = round9 / "framework_hooks.py"
    framework_hooks.write_text(
        patch_nonkt_framework_hooks(
            (inputs / "migration_bridge/round9/runtime/framework_hooks.py").read_text()
        )
    )
    shutil.copy2(
        inputs / "artifacts/round12a/round12a_nonkt_identity.py",
        round9 / "round12a_nonkt_identity.py",
    )
    candidate_bank = round9 / "candidate_bank.py"
    candidate_bank.write_text(patch_candidate_bank(candidate_bank.read_text()))
    ple = round9 / "ple_binding.py"
    ple.write_text(patch_nonkt_ple_binding(ple.read_text()))
    model_runner = overlay / "model_executor/model_runner.py"
    model_runner.write_text(patch_nonkt_model_runner(model_runner.read_text()))
    shutil.copy2(
        inputs / "artifacts/round12a/fixtures/offloader_v1_tied_candidate.py",
        overlay / "utils/offloader.py",
    )
    for source in (
        "round12a_nonkt_server.py",
        "round12a_nonkt_launcher.py",
    ):
        shutil.copy2(inputs / "artifacts/round12a" / source, root / source)
    _assert_promoted_tuple(root)
    if sha256(overlay / "utils/offloader.py") != TIED_OFFLOADER_SHA256:
        raise ValueError("Round12a tied Offloader source changed")

    profile = {
        "model_path": "/models/de4b8e4d43b917e7706784d8bb445c9af86a3540",
        "device": "npu",
        "tp_size": 1,
        "enable_multimodal": False,
        "dtype": "bfloat16",
        "load_format": "safetensors",
        "disable_cuda_graph": True,
        "disable_radix_cache": True,
        "disable_overlap_schedule": True,
        "disable_shared_experts_fusion": True,
        "max_running_requests": 1,
        "max_total_tokens": 4100,
        "context_length": 4098,
        "max_mamba_cache_size": 1,
        "max_prefill_tokens": 32,
        "page_size": 4,
        "chunked_prefill_size": 32,
        "attention_backend": "ascend",
        "mem_fraction_static": 0.45,
        "kt_weight_path": None,
        "cpu_offload_gb": 233,
        "offload_group_size": -1,
        "host": "127.0.0.1",
        "port": 30091,
        "random_seed": 20260920,
        "watchdog_timeout": 1800,
        "skip_server_warmup": True,
        "served_model_name": "QWEN38-ROUND12A-NONKT-BF16-OFFLOADERV1",
    }
    (root / "nonkt-server-profile.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    framework = {
        "schema": 3,
        "resident_capacity": 32,
        "route_collection": False,
        "frequency": None,
        "prefill": None,
        "ple": {
            "mode": "ASYNC",
            "device_gate": {
                "path": "/round9/evidence/20260921-ple-v2/result.json",
                "sha256": "b753270612776c26b15d2c00e9bb9817bf9d86ca8bcf7e12c183e455ffa2e551",
            },
        },
    }
    (root / "nonkt-framework-profile.json").write_text(
        json.dumps(framework, indent=2) + "\n"
    )
    context_profile = {
        "schema": 1,
        "mode": "BF16_COMPACT",
        "device_gate": {
            "path": "/round9/evidence/20260921-ple-v4/result.json",
            "sha256": "e1a8213bf8301bf3758df4a67d9031fedcfdacaa460b5787c0c2951723b31811",
        },
    }
    (root / "context-reference-profile.json").write_text(
        json.dumps(context_profile, indent=2) + "\n"
    )
    capacity = json.loads(
        (inputs / "artifacts/round12/ROUND12_NON_KT_CAPACITY_RAW.json").read_text()
    )
    identity = {
        "schema": 1,
        "status": "FROZEN_ROUND12A_NONKT_IDENTITY",
        "model_revision": SOURCE_IDENTITY["revision"],
        "source_identity": SOURCE_IDENTITY,
        "cpu_offload_gib": 233,
        "loaded_main_layer_bytes": capacity["categories"]["loaded_main_layers"],
        "largest_layer_stage_bytes": capacity["largest_decoder_layer_bytes"],
        "offloader_sha256": sha256(overlay / "utils/offloader.py"),
        "qwen4_exp_sha256": sha256(overlay / "models/qwen4_exp.py"),
        "model_runner_sha256": sha256(model_runner),
        "qsa_sha256": sha256(round9 / "qsa.py"),
        "context_reference_runtime_sha256": sha256(
            round9 / "context_reference_runtime.py"
        ),
        "context_reference_binding_sha256": sha256(
            round9 / "context_reference_binding.py"
        ),
    }
    (root / "ROUND12A_NONKT_IDENTITY.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    shutil.copy2(
        inputs / "artifacts/round12/ROUND12_NON_KT_CAPACITY_RAW.json",
        root / "ROUND12_NON_KT_CAPACITY_RAW.json",
    )
    refresh_overlay(root, mode="NON_KT")


def build(
    parent: Path,
    output: Path,
    inputs: Path,
    *,
    version: int,
    queue_path: Path | None = None,
) -> Path:
    if version not in ROLES:
        raise ValueError("unsupported Round12a model version")
    parent, output, inputs = map(Path, (parent, output, inputs))
    if output.exists():
        raise FileExistsError(output)
    if version == 173 and queue_path is not None:
        raise ValueError("v173 smoke cannot consume an ABBA queue")
    if version != 173 and queue_path is None:
        raise ValueError("Round12a ABBA snapshot requires a queue")
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__"))
    for path in (output, *output.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    prepare_common(
        output, inputs, version=version, queue_path=Path(queue_path) if queue_path else None
    )
    mode, unused_block = ROLES[version]
    if mode == "KT":
        prepare_kt(output, version=version)
    else:
        prepare_nonkt(output, inputs)
    manifest = source_manifest(output)
    (output / "round12-source-sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--queue", type=Path)
    args = parser.parse_args()
    result = build(
        args.parent,
        args.output,
        args.inputs,
        version=args.version,
        queue_path=args.queue,
    )
    print(
        "PASS_ROUND12A_MODEL_SNAPSHOT",
        args.version,
        ROLES[args.version],
        sha256(result / "round12-source-sha256.json"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
