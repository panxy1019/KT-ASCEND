"""Build a fresh S0 root-cause profile snapshot from Round16 A1 evidence.

Run this program on the A3 host.  The donor is copied; historical evidence and
the admitted baseline runtime are never modified.  The resulting launcher is
bound to the attribution-only native runtime and a new container/evidence ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


TRACE_NATIVE_SHA256 = "bccee617542c1a23ed1531e1099d1bff6190f8212c91f806b9ea705bec2b8a62"
BASELINE_NATIVE_SHA256 = "614ae60f70ff70d1e47d7fd32e857f887f874b6248eb332172c8a637cb471a6d"
TRACE_BUILD_RECEIPT = Path("/mnt/data1/kt-round9/kt-root-cause-build-v1/build-receipt.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new)


def patch_launcher(
    text: str,
    *,
    block: str,
    run_version: int = 501,
    binding_sha: str = "0" * 64,
    baseline_q32_sha: str | None = None,
    diagnostic_q32_sha: str | None = None,
    baseline_layerwise_sha: str | None = None,
    diagnostic_layerwise_sha: str | None = None,
) -> str:
    suffix = "a" if block == "A" else "b"
    old_identity = (
        "name='kt-round16-a1-a-v003';out=root/'evidence'/'20260929-round16-a1-fresh-profile-a-v003'"
        if block == "A"
        else "name='kt-round16-a1-b-v003';out=root/'evidence'/'20260929-round16-a1-fresh-profile-b-v003'"
    )
    text = replace_once(
        text,
        "if a.version != 295 or a.cost_mode != 'OFF':",
        f"if a.version != {run_version} or a.cost_mode != 'OFF':",
        "version gate",
    )
    text = replace_once(
        text,
        old_identity,
        f"name='kt-root-cause-profile-{suffix}-v{run_version}';out=root/'evidence'/'20260929-kt-root-cause-profile-{suffix}-v{run_version}'",
        "run identity",
    )
    text = replace_once(
        text,
        "runtime=root/'candidate-runtime-python'\n"
        "if json.loads((a.device_gate.parent/'runtime-source-sha256.json').read_text())!=tree(runtime):raise RuntimeError('native runtime differs from device gate')",
        "baseline_runtime=root/'candidate-runtime-python'\n"
        "if json.loads((a.device_gate.parent/'runtime-source-sha256.json').read_text())!=tree(baseline_runtime):raise RuntimeError('baseline native runtime differs from device gate')\n"
        "runtime=root/'round11-t2-attribution-runtime-v3'\n"
        "baseline_non_native={k:v for k,v in tree(baseline_runtime).items() if not (k.startswith('kt_kernel/kt_kernel_ext.') and k.endswith('.so'))}\n"
        "trace_non_native={k:v for k,v in tree(runtime).items() if not (k.startswith('kt_kernel/kt_kernel_ext.') and k.endswith('.so'))}\n"
        "if trace_non_native!=baseline_non_native:raise RuntimeError('trace runtime changed non-native payload')\n"
        f"if sum(digest(p)=='{TRACE_NATIVE_SHA256}' for p in runtime.rglob('*') if p.is_file())!=1:raise RuntimeError('trace native extension identity mismatch')",
        "runtime binding",
    )
    text = replace_once(
        text,
        "if sum(digest(path)==native_sha256 for path in runtime.rglob('*') if path.is_file())!=1:\n"
        "  raise RuntimeError('Q32 native extension is not uniquely present in admitted runtime')",
        "if sum(digest(path)==native_sha256 for path in baseline_runtime.rglob('*') if path.is_file())!=1:\n"
        "  raise RuntimeError('Q32 baseline native extension is not uniquely present in admitted runtime')",
        "q32 baseline gate",
    )
    text = replace_once(
        text,
        "args+=['-e','ROUND14A_GDN_BRANCH_PATH=/out/round14a-gdn-branch.jsonl']",
        "args+=['-e','ROUND14A_GDN_BRANCH_PATH=/out/round14a-gdn-branch.jsonl']\n"
        "args+=['-e','KT_ROOT_TASK_TRACE_PATH=/out/native-task-trace.jsonl']\n"
        f"args+=['-e','KT_ROOT_TRACE_BINDING_PATH=/round9/kt-root-cause-profile-{suffix}-v{run_version}/TRACE_NATIVE_BINDING.json']",
        "trace environment",
    )
    text = replace_once(
        text,
        "if budget.get('stage')!='ROUND16_A1_FRESH_PROFILE':raise RuntimeError('wrong Round16 A1 model budget')",
        "if budget.get('stage')!='KT_ROOT_CAUSE_PROFILE':raise RuntimeError('wrong KT root-cause model ledger')",
        "diagnostic ledger stage",
    )
    if baseline_q32_sha is not None or diagnostic_q32_sha is not None:
        if baseline_q32_sha is None or diagnostic_q32_sha is None:
            raise ValueError("both Q32 gate identities are required")
        text = patch_overlay_gate(
            text,
            baseline_q32_sha=baseline_q32_sha,
            diagnostic_q32_sha=diagnostic_q32_sha,
            baseline_layerwise_sha=baseline_layerwise_sha,
            diagnostic_layerwise_sha=diagnostic_layerwise_sha,
        )
    return text


def patch_overlay_gate(
    text: str,
    *,
    baseline_q32_sha: str,
    diagnostic_q32_sha: str,
    baseline_layerwise_sha: str | None = None,
    diagnostic_layerwise_sha: str | None = None,
) -> str:
    """Verify the donor overlay after normalizing exactly one diagnostic file."""
    if len(baseline_q32_sha) != 64 or len(diagnostic_q32_sha) != 64:
        raise ValueError("Q32 gate SHA-256 is malformed")
    if (baseline_layerwise_sha is None) != (diagnostic_layerwise_sha is None):
        raise ValueError("both layerwise gate identities are required")
    if baseline_layerwise_sha is not None and (
        len(baseline_layerwise_sha) != 64 or len(diagnostic_layerwise_sha) != 64
    ):
        raise ValueError("layerwise gate SHA-256 is malformed")
    old = (
        "if overlay['baseline_tree_sha256']!=tree_digest(tree_files(root/'src/sglang-round9-port/python')) or overlay['candidate_tree_sha256']!=tree_digest(tree_files(sglang)):\n"
        " raise RuntimeError('installed overlay differs from recorded baseline/candidate payload')"
    )
    layerwise = ""
    if baseline_layerwise_sha is not None:
        layerwise = (
            "_layerwise_relative='sglang/srt/layers/round9/layerwise_gate.py'\n"
            f"if candidate_files.get(_layerwise_relative)!='{diagnostic_layerwise_sha}':\n"
            " raise RuntimeError('diagnostic layerwise gate differs from bound source')\n"
            f"candidate_files[_layerwise_relative]='{baseline_layerwise_sha}'\n"
        )
    new = (
        "candidate_files=tree_files(sglang)\n"
        "_q32_relative='sglang/srt/layers/round9/q32_gate.py'\n"
        f"if candidate_files.get(_q32_relative)!='{diagnostic_q32_sha}':\n"
        " raise RuntimeError('diagnostic Q32 gate differs from bound source')\n"
        f"candidate_files[_q32_relative]='{baseline_q32_sha}'\n" + layerwise +
        "if overlay['baseline_tree_sha256']!=tree_digest(tree_files(root/'src/sglang-round9-port/python')) or overlay['candidate_tree_sha256']!=tree_digest(candidate_files):\n"
        " raise RuntimeError('installed overlay differs outside diagnostic Q32 gate')"
    )
    return replace_once(text, old, new, "diagnostic overlay source gate")


def patch_candidate_server(text: str, *, binding_sha: str) -> str:
    """Keep the numeric Q32 gate and separately authenticate the trace-only SO."""
    if len(binding_sha) != 64:
        raise ValueError("diagnostic binding SHA-256 is malformed")
    old = (
        "native_sha256=hashlib.sha256(Path(kt_kernel_ext.__file__).resolve().read_bytes()).hexdigest()\n"
        "    require_capacity_device_gate(capacity,candidate,identity,payload,native_sha256)"
    )
    new = (
        "actual_native_sha256=hashlib.sha256(Path(kt_kernel_ext.__file__).resolve().read_bytes()).hexdigest()\n"
        "    trace_binding_path=os.environ.get('KT_ROOT_TRACE_BINDING_PATH')\n"
        "    if not trace_binding_path: raise RuntimeError('diagnostic native binding is required')\n"
        "    trace_binding_raw=Path(trace_binding_path).read_bytes()\n"
        f"    if hashlib.sha256(trace_binding_raw).hexdigest()!='{binding_sha}': raise RuntimeError('diagnostic native binding SHA mismatch')\n"
        "    binding=json.loads(trace_binding_raw)\n"
        "    if (binding.get('schema')!=1 or binding.get('status')!='PASS_KT_ROOT_DIAGNOSTIC_NATIVE_BINDING'\n"
        "            or binding.get('diagnostic_native_sha256')!=actual_native_sha256\n"
        "            or binding.get('semantic_scope')!='ENV_GATED_TASK_QUEUE_TIMESTAMPS_ONLY'\n"
        "            or binding.get('math_changes')!=0 or binding.get('synchronization_changes')!=0):\n"
        "        raise RuntimeError('diagnostic native binding contract mismatch')\n"
        "    require_capacity_device_gate(capacity,candidate,identity,payload,binding['baseline_native_sha256'])"
    )
    return replace_once(text, old, new, "candidate server diagnostic native gate")


def patch_q32_gate(text: str, *, binding_sha: str) -> str:
    """Extend the existing source-bound gate for one authenticated trace build."""
    if len(binding_sha) != 64:
        raise ValueError("diagnostic binding SHA-256 is malformed")
    old = (
        "    if (gate.get('storage') != storage or gate.get('cpu_workers') != workers or\n"
        "            gate.get('source_identity') != source_identity or\n"
        "            gate.get('bank_sha256') != layer0_sha256 or\n"
        "            not _sha256(native_sha256) or gate.get('native_sha256') != native_sha256):\n"
        "        raise RuntimeError('Q32 receipt source, storage, workers, bank or native identity mismatch')"
    )
    new = (
        "    validated_native_sha256 = native_sha256\n"
        "    if gate.get('native_sha256') != validated_native_sha256:\n"
        "        import os\n"
        "        binding_path = os.environ.get('KT_ROOT_TRACE_BINDING_PATH')\n"
        "        if not binding_path:\n"
        "            raise RuntimeError('Q32 diagnostic native binding is required')\n"
        "        binding_raw = Path(binding_path).read_bytes()\n"
        f"        if hashlib.sha256(binding_raw).hexdigest() != '{binding_sha}':\n"
        "            raise RuntimeError('Q32 diagnostic native binding SHA mismatch')\n"
        "        binding = json.loads(binding_raw)\n"
        "        if (binding.get('schema') != 1 or\n"
        "                binding.get('status') != 'PASS_KT_ROOT_DIAGNOSTIC_NATIVE_BINDING' or\n"
        "                binding.get('diagnostic_native_sha256') != native_sha256 or\n"
        "                binding.get('baseline_native_sha256') != gate.get('native_sha256') or\n"
        "                binding.get('semantic_scope') != 'ENV_GATED_TASK_QUEUE_TIMESTAMPS_ONLY' or\n"
        "                binding.get('math_changes') != 0 or\n"
        "                binding.get('synchronization_changes') != 0):\n"
        "            raise RuntimeError('Q32 diagnostic native binding contract mismatch')\n"
        "        validated_native_sha256 = binding['baseline_native_sha256']\n"
        "    if (gate.get('storage') != storage or gate.get('cpu_workers') != workers or\n"
        "            gate.get('source_identity') != source_identity or\n"
        "            gate.get('bank_sha256') != layer0_sha256 or\n"
        "            not _sha256(validated_native_sha256) or\n"
        "            gate.get('native_sha256') != validated_native_sha256):\n"
        "        raise RuntimeError('Q32 receipt source, storage, workers, bank or native identity mismatch')"
    )
    return replace_once(text, old, new, "shared Q32 diagnostic native gate")


def patch_layerwise_gate(text: str, *, binding_sha: str) -> str:
    """Keep the layerwise numeric/source gate while binding the trace-only SO."""
    if len(binding_sha) != 64:
        raise ValueError("diagnostic binding SHA-256 is malformed")
    old = (
        "    if (not isinstance(source_identity, dict) or not source_identity or\n"
        "            gate.get('source_identity') != source_identity or\n"
        "            not _sha256(native_sha256) or gate.get('native_sha256') != native_sha256):\n"
        "        raise RuntimeError('layerwise original model or native identity mismatch')"
    )
    new = (
        "    validated_native_sha256 = native_sha256\n"
        "    if gate.get('native_sha256') != validated_native_sha256:\n"
        "        import os\n"
        "        binding_path = os.environ.get('KT_ROOT_TRACE_BINDING_PATH')\n"
        "        if not binding_path:\n"
        "            raise RuntimeError('layerwise diagnostic native binding is required')\n"
        "        binding_raw = Path(binding_path).read_bytes()\n"
        f"        if hashlib.sha256(binding_raw).hexdigest() != '{binding_sha}':\n"
        "            raise RuntimeError('layerwise diagnostic native binding SHA mismatch')\n"
        "        binding = json.loads(binding_raw)\n"
        "        if (binding.get('schema') != 1 or\n"
        "                binding.get('status') != 'PASS_KT_ROOT_DIAGNOSTIC_NATIVE_BINDING' or\n"
        "                binding.get('diagnostic_native_sha256') != native_sha256 or\n"
        "                binding.get('baseline_native_sha256') != gate.get('native_sha256') or\n"
        "                binding.get('semantic_scope') != 'ENV_GATED_TASK_QUEUE_TIMESTAMPS_ONLY' or\n"
        "                binding.get('math_changes') != 0 or\n"
        "                binding.get('synchronization_changes') != 0):\n"
        "            raise RuntimeError('layerwise diagnostic native binding contract mismatch')\n"
        "        validated_native_sha256 = binding['baseline_native_sha256']\n"
        "    if (not isinstance(source_identity, dict) or not source_identity or\n"
        "            gate.get('source_identity') != source_identity or\n"
        "            not _sha256(validated_native_sha256) or\n"
        "            gate.get('native_sha256') != validated_native_sha256):\n"
        "        raise RuntimeError('layerwise original model or native identity mismatch')"
    )
    return replace_once(text, old, new, "shared layerwise diagnostic native gate")


def make_trace_binding() -> dict:
    raw = TRACE_BUILD_RECEIPT.read_bytes()
    build = json.loads(raw)
    if (
        build.get("status") != "PASS_ROUND11_T2_ATTRIBUTION_BUILD"
        or build.get("baseline_native_sha256") != BASELINE_NATIVE_SHA256
        or build.get("attribution_native_sha256") != TRACE_NATIVE_SHA256
        or build.get("semantic_scope") != "ENV_GATED_TASK_QUEUE_TIMESTAMPS_ONLY"
        or build.get("optimization_candidate") is not False
    ):
        raise RuntimeError("native trace build receipt is outside the diagnostic contract")
    return {
        "schema": 1,
        "status": "PASS_KT_ROOT_DIAGNOSTIC_NATIVE_BINDING",
        "baseline_native_sha256": BASELINE_NATIVE_SHA256,
        "diagnostic_native_sha256": TRACE_NATIVE_SHA256,
        "native_build_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "source_receipt_sha256": build["source_receipt_sha256"],
        "semantic_scope": "ENV_GATED_TASK_QUEUE_TIMESTAMPS_ONLY",
        "math_changes": 0,
        "synchronization_changes": 0,
        "optimization_candidate": False,
    }


def patch_monitor(text: str, *, run_version: int = 501) -> str:
    return replace_once(
        text,
        "r'|kt-round16-a1-[ab]-v003)', name) is not None",
        f"r'|kt-round16-a1-[ab]-v003|kt-root-cause-profile-[ab]-v{run_version})', name) is not None",
        "monitor owner allow-list",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--block", choices=["A", "B"], required=True)
    parser.add_argument("--run-version", type=int, default=501)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("snapshot output and receipt must be fresh")
    shutil.copytree(args.donor, args.output, symlinks=True)
    launcher = args.output / "source/tools/launch_hybrid.py"
    monitor = args.output / "source/tools/monitor_owned_server.py"
    candidate_server = args.output / "source/runtime/candidate_server.py"
    q32_gate = args.output / "overlay-python/sglang/srt/layers/round9/q32_gate.py"
    layerwise_gate = args.output / "overlay-python/sglang/srt/layers/round9/layerwise_gate.py"
    binding_path = args.output / "TRACE_NATIVE_BINDING.json"
    if not 501 <= args.run_version <= 999:
        raise ValueError("diagnostic run version must be in [501, 999]")
    expected_name = f"kt-root-cause-profile-{args.block.lower()}-v{args.run_version}"
    if args.output.name != expected_name:
        raise ValueError(f"snapshot output name must be {expected_name}")
    binding_path.write_text(json.dumps(make_trace_binding(), indent=2) + "\n")
    binding_sha = sha256(binding_path)
    baseline_q32_sha = sha256(q32_gate)
    q32_gate.write_text(patch_q32_gate(q32_gate.read_text(), binding_sha=binding_sha))
    diagnostic_q32_sha = sha256(q32_gate)
    baseline_layerwise_sha = sha256(layerwise_gate)
    layerwise_gate.write_text(
        patch_layerwise_gate(layerwise_gate.read_text(), binding_sha=binding_sha)
    )
    diagnostic_layerwise_sha = sha256(layerwise_gate)
    launcher.write_text(
        patch_launcher(
            launcher.read_text(),
            block=args.block,
            run_version=args.run_version,
            binding_sha=binding_sha,
            baseline_q32_sha=baseline_q32_sha,
            diagnostic_q32_sha=diagnostic_q32_sha,
            baseline_layerwise_sha=baseline_layerwise_sha,
            diagnostic_layerwise_sha=diagnostic_layerwise_sha,
        )
    )
    monitor.write_text(patch_monitor(monitor.read_text(), run_version=args.run_version))
    candidate_server.write_text(
        patch_candidate_server(candidate_server.read_text(), binding_sha=binding_sha)
    )
    compile(launcher.read_text(), str(launcher), "exec")
    compile(monitor.read_text(), str(monitor), "exec")
    compile(candidate_server.read_text(), str(candidate_server), "exec")
    compile(q32_gate.read_text(), str(q32_gate), "exec")
    compile(layerwise_gate.read_text(), str(layerwise_gate), "exec")
    budget_path = args.output / "execution-budget.json"
    budget = json.loads(budget_path.read_text())
    budget["stage"] = "KT_ROOT_CAUSE_PROFILE"
    budget["model_starts_limit"] = None
    budget["model_starts_used"] = 0
    budget["starts"] = []
    budget_path.write_text(json.dumps(budget, indent=2) + "\n")
    for relative in ("execution-budget.lock", "host-controller.json"):
        path = args.output / relative
        if path.exists():
            path.unlink()
    native = Path("/mnt/data1/kt-round9/round11-t2-attribution-runtime-v3/kt_kernel")
    candidates = list(native.glob("kt_kernel_ext.*.so"))
    if len(candidates) != 1 or sha256(candidates[0]) != TRACE_NATIVE_SHA256:
        raise RuntimeError("trace native runtime is absent or changed")
    receipt = {
        "schema": 1,
        "status": "PASS_KT_ROOT_CAUSE_PROFILE_SNAPSHOT",
        "created_at": time.time(),
        "block": args.block,
        "run_version": args.run_version,
        "donor": str(args.donor),
        "output": str(args.output),
        "launcher_sha256": sha256(launcher),
        "monitor_sha256": sha256(monitor),
        "candidate_server_sha256": sha256(candidate_server),
        "q32_gate_sha256": sha256(q32_gate),
        "layerwise_gate_sha256": sha256(layerwise_gate),
        "trace_binding_sha256": binding_sha,
        "trace_build_receipt_sha256": sha256(TRACE_BUILD_RECEIPT),
        "budget_sha256": sha256(budget_path),
        "baseline_native_sha256": BASELINE_NATIVE_SHA256,
        "trace_native_sha256": TRACE_NATIVE_SHA256,
        "trace_default": "ENABLED_ONLY_IN_THIS_DIAGNOSTIC_SNAPSHOT",
        "math_changes": 0,
        "synchronization_changes": 0,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt["status"])


if __name__ == "__main__":
    main()
