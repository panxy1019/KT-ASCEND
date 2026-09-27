#!/usr/bin/env python3
"""Fail closed on a native TP8 snapshot before any model/NPU process starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROLES = {180: "A", 181: "B", 182: "B", 183: "A"}
OLD_QSA_SHA256 = "76e9f34066f0b42f42545427967c0b789d5c32031dc5c085e230efaf95e3c54c"
PROMOTED_QSA_SHA256 = "0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35"
CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
TP8_GEOMETRY = {
    "global_query_heads": 24,
    "global_kv_heads": 2,
    "local_query_heads": 3,
    "local_kv_heads": 1,
    "replicated_indexer_query_heads": 4,
    "replicated_indexer_kv_heads": 1,
    "global_gdn_key_heads": 16,
    "global_gdn_value_heads": 48,
    "local_gdn_key_heads": 2,
    "local_gdn_value_heads": 6,
}
MUTABLE = {"round12-source-sha256.json", "execution-budget.json", "execution-budget.lock", "host-controller.json"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(root: Path) -> dict:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def run(snapshot: Path) -> dict:
    snapshot = Path(snapshot)
    contract = json.loads((snapshot / "ROUND12A_NATIVE_TP8_CONTRACT.json").read_text())
    version = int(contract["version"])
    if (
        ROLES.get(version) != contract.get("block") or contract.get("mode") != "NATIVE_TP8"
        or contract.get("arm_semantics") != "IDENTICAL_NATIVE_REPEAT_COHORT"
        or contract.get("tp_size") != 8 or contract.get("physical_devices") != list(range(8))
        or contract.get("generation_limit") != 4 or contract.get("automatic_retries") != 0
    ):
        raise RuntimeError("native TP8 role/version contract mismatch")
    manifest = snapshot / "round12-source-sha256.json"
    if tree(snapshot) != json.loads(manifest.read_text()):
        raise RuntimeError("native TP8 snapshot manifest mismatch")
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
        receipt = json.loads((overlay / "round9-candidate-overlay.json").read_text())
        if receipt["candidate_tree_sha256"] != tree_digest(tree_files(overlay)):
            raise RuntimeError("native TP8 overlay receipt mismatch")
    finally:
        sys.path.remove(str(tools))
    identity_path = snapshot / "ROUND12A_NATIVE_TP8_IDENTITY.json"
    identity = json.loads(identity_path.read_text())
    geometry = json.loads((snapshot / "NATIVE_TP8_MODEL_GEOMETRY.json").read_text())
    upstream = json.loads((snapshot / "NATIVE_TP8_UPSTREAM_SOURCE_PROOF.json").read_text())
    qsa = srt / "layers/round9/qsa.py"
    qwen = srt / "models/qwen4_exp.py"
    ple = srt / "layers/round9/ple_binding.py"
    hooks = srt / "layers/round9/framework_hooks.py"
    if (
        sha256(qsa) != OLD_QSA_SHA256 or sha256(qsa) == PROMOTED_QSA_SHA256
        or identity.get("status") != "FROZEN_ROUND12A_NATIVE_TP8_OLD_QSA"
        or identity.get("qsa_sha256") != OLD_QSA_SHA256
        or identity.get("promoted_qsa_rejected_sha256") != PROMOTED_QSA_SHA256
        or identity.get("patched_qwen_sha256") != sha256(qwen)
        or identity.get("ple_binding_sha256") != sha256(ple)
        or identity.get("framework_hooks_sha256") != sha256(hooks)
        or identity.get("config_sha256") != CONFIG_SHA256
        or identity.get("tp8_geometry") != TP8_GEOMETRY
        or geometry.get("config_sha256") != CONFIG_SHA256
        or geometry.get("tp8_local") != {
            "query_heads": 3, "kv_heads": 1, "kv_replication_group_ranks": 4,
            "gdn_key_heads": 2, "gdn_value_heads": 6,
        }
        or upstream.get("commit") != "4ccff141dbe992794f9da6c3aa23535b4f72000d"
        or upstream.get("downloaded_source_sha256")
        != "a3981e148bc9ede4b84dea700f40831f86211cea29b817e8cc8feca1066e9f72"
    ):
        raise RuntimeError("native TP8 old-QSA source tuple changed")
    qwen_text = qwen.read_text()
    for token in (
        "attn_tp_all_reduce", "hidden = attn_tp_all_reduce(hidden)",
        'ROUND12A_NATIVE_TP8_MODE', 'tp_size not in (1, 8)',
        "num_key_heads=self.num_k_heads // self.attn_tp_size",
        "num_value_heads=self.num_v_heads // self.attn_tp_size",
        "self.reference_config.num_value_heads",
        "NATIVE_TP8_OLD_QSA_HIT", "NATIVE_TP8_GDN_HIT",
        "NATIVE_TP8_LAYER_PHYSICAL", '"num_attention_heads": 24',
        "qsa_layers != list(range(3, 48, 4))",
    ):
        if token not in qwen_text:
            raise RuntimeError(f"native TP8 Qwen source gate missing: {token}")
    if "args.tp_size not in (1, 8)" not in ple.read_text() or "ple-async-events-rank" not in ple.read_text():
        raise RuntimeError("native TP8 PLE source gate missing")
    pool = srt / "mem_cache/round9_qsa_pool.py"
    mixin = srt / "model_executor/model_runner_kv_cache_mixin.py"
    fused_moe = srt / "layers/moe/fused_moe_triton/layer.py"
    if (
        "class Round9QSAKVPool" not in pool.read_text()
        or "get_attention_tp_size()" not in mixin.read_text()
        or "pool_class = Round9QSAKVPool" not in mixin.read_text()
        or "intermediate_size // self.moe_tp_size" not in fused_moe.read_text()
    ):
        raise RuntimeError("native TP8 TP-local KV/expert source geometry is missing")
    profile = json.loads((snapshot / "native-tp8-server-profile.json").read_text())
    if (
        profile.get("tp_size") != 8 or profile.get("device") != "npu"
        or profile.get("dtype") != "bfloat16" or profile.get("attention_backend") != "ascend"
        or profile.get("cpu_offload_gb") != 0 or profile.get("kt_weight_path") is not None
        or profile.get("mem_fraction_static") != 0.70
    ):
        raise RuntimeError("native TP8 server profile changed")
    mean_bytes = int(identity["mean_text_bytes_per_rank"])
    conservative_weight_bytes = int(identity["conservative_weight_bytes_per_rank"])
    static_bytes = int(profile["mem_fraction_static"] * 65536 * 1024**2)
    if static_bytes - conservative_weight_bytes < 7 * 1024**3:
        raise RuntimeError("native TP8 source-only conservative weight headroom fell below 7 GiB per rank")
    if (
        identity.get("pure_npu_scope") != "MAIN_AND_EXPERT_WEIGHTS_AND_COMPUTE_ON_EIGHT_NPUS"
        or identity.get("host_dependency") != "PLE_CHECKPOINT_ROW_TABLE_REMAINS_HOST_BACKED"
    ):
        raise RuntimeError("native TP8 scope statement changed")
    queue = json.loads((snapshot / "ROUND12A_ABBA_QUEUE.json").read_text())
    if (
        queue.get("workloads") != ["W1"]
        or queue.get("workload_definitions") != {"W1": {"case_id": "edge64", "input_tokens": 64, "decode_tokens": 2}}
        or queue.get("expected_output_ids") != {"W1": [3897, 248046]}
    ):
        raise RuntimeError("native TP8 queue differs from Round12a ABBA W1")
    launcher = (snapshot / "native_tp8_launcher.py").read_text()
    for token in (
        "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
        "ROUND12A_NATIVE_TP8_MODE=1", "NO_OVERLAP_WITH_PHYSICAL12_PERFORMANCE_WINDOWS",
    ):
        if token not in launcher:
            raise RuntimeError(f"native TP8 launcher gate missing: {token}")
    if "ROUND9_CONTEXT_REFERENCE_PROFILE" in launcher or PROMOTED_QSA_SHA256 in launcher:
        raise RuntimeError("native TP8 launcher accidentally binds candidate QSA/context consumer")
    image_preflight = (snapshot / "native_tp8_image_preflight.py").read_text()
    for token in (
        "PASS_ROUND12A_NATIVE_TP8_SAME_IMAGE_PREFLIGHT",
        "physical_devices_exposed", "ServerArgs(**profile)",
    ):
        if token not in image_preflight:
            raise RuntimeError(f"native TP8 same-image preflight gate missing: {token}")
    return {
        "schema": 1, "status": "PASS_ROUND12A_NATIVE_TP8_OFFLINE_PREFLIGHT",
        "version": version, "block": contract["block"], "compiled_python_files": compiled,
        "snapshot_manifest_sha256": sha256(manifest),
        "overlay_tree_sha256": receipt["candidate_tree_sha256"],
        "qsa_sha256": sha256(qsa), "patched_qwen_sha256": sha256(qwen),
        "tp_size": 8, "physical_devices": list(range(8)),
        "mean_text_bytes_per_rank": mean_bytes, "static_budget_bytes_per_rank": static_bytes,
        "mean_shard_margin_bytes_per_rank": static_bytes - mean_bytes,
        "conservative_weight_bytes_per_rank": conservative_weight_bytes,
        "conservative_weight_headroom_bytes_per_rank": static_bytes - conservative_weight_bytes,
        "capacity_qualification": "SOURCE_ARITHMETIC_PLAUSIBLE_ACTUAL_DEVICE_LOAD_REQUIRED",
        "pure_npu_scope": identity["pure_npu_scope"], "host_dependency": identity["host_dependency"],
        "overlap_disposition": "FORBIDDEN_PCI_NUMA_UNKNOWN_AND_SHARED_HOST_CACHE_CGROUP",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.snapshot)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["version"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
