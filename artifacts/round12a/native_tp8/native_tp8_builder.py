#!/usr/bin/env python3
"""Build isolated Round12a native-SGLang TP8/old-QSA snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys


ROLES = {180: "A", 181: "B", 182: "B", 183: "A"}
OLD_QSA_SHA256 = "76e9f34066f0b42f42545427967c0b789d5c32031dc5c085e230efaf95e3c54c"
OLD_QWEN_SHA256 = "09c3d027fc86071be687989bdb04901037adaa34e4c310f2471f40275091a0a4"
PROMOTED_QSA_SHA256 = "0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35"
PROMOTED_QWEN_SHA256 = "d4583c5abe9791b09c48d1dd8f8491a2988eb5147d45df188e0e48908c699f1f"
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
SOURCE_IDENTITY = {
    "revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
    "manifest_sha256": "62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e",
    "receipt_sha256": "7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239",
    "model_path": "/models/de4b8e4d43b917e7706784d8bb445c9af86a3540",
    "file_sizes_checked": 144,
    "verification_scope": "TRUSTED_FULL_RECEIPT_PLUS_LIVE_SIZES_AND_METADATA_HASHES",
}
LOADED_TEXT_BYTES = 251_487_307_800
PLE_EXTERNAL_TABLE_BYTES = 102_400_491_520
EXPERT_TEXT_BYTES = 48 * 512 * 3 * 2560 * 640 * 2
NON_EXPERT_TEXT_BYTES = LOADED_TEXT_BYTES - EXPERT_TEXT_BYTES
# BF16 expert tensors are evenly sharded on the 640/8 intermediate axis.
# Treat every other text tensor as fully replicated for a source-only upper
# bound.  Runtime allocation still requires the HBM guard and actual load.
CONSERVATIVE_WEIGHT_BYTES_PER_RANK = EXPERT_TEXT_BYTES // 8 + NON_EXPERT_TEXT_BYTES
FROZEN_INPUT_SHA256 = {
    "framework_hooks.py": "6b74c54a2ae847a590b86040f81072c17c0e098177013e5b6171d1a8e1e3a795",
    "round11_t4_driver.py": "38faa29ce4583254c59ee32bdf921329cf97aa0b04393a3a4bc5c9ef2d1ca2be",
    "round10-t1-fixture.json": "21e4f82531a58881f2fa251fda50ef3f9909c589336164b77fe732a77abdca95",
    "ROUND12A_ABBA_QUEUE.json": "ab3fc4141081318b400721e041b8ec5dc740bc23a8c7e900ffa8cabe35804637",
    "numa_policy.py": "88239e9b0b6069f3c4f81c7d5b64bf994dbcb8dc8ef99aea37fd56bde985f292",
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


def patch_qwen_tp8(source: str) -> str:
    """Add the upstream-required TP reduction and bounded warmup receipts."""
    if hashlib.sha256(source.encode()).hexdigest() != OLD_QWEN_SHA256:
        raise ValueError("old Qwen source identity changed")
    source = replace_once(
        source,
        "Profile: eager text, TP1, batch1, graph/radix/speculation/MTP off.",
        "Profile: eager text, TP1 or explicit Round12a TP8, batch1, graph/radix/speculation/MTP off.",
        "Qwen profile",
    )
    source = replace_once(source, "import re\n", "import re\nimport json\nfrom pathlib import Path\n", "Qwen imports")
    source = replace_once(
        source,
        "from sglang.srt.layers.dp_attention import get_attention_tp_size",
        "from sglang.srt.layers.dp_attention import (\n"
        "    attn_tp_all_reduce, get_attention_tp_rank, get_attention_tp_size,\n"
        ")",
        "attention TP helpers",
    )
    source = replace_once(
        source,
        '''def _guard_profile(forward_batch: ForwardBatch) -> None:\n    if get_attention_tp_size() != 1:\n        raise RuntimeError("Round9 reference model requires TP=1")''',
        '''def _guard_profile(forward_batch: ForwardBatch) -> None:\n    tp_size = get_attention_tp_size()\n    native_tp8 = os.environ.get("ROUND12A_NATIVE_TP8_MODE") == "1"\n    if tp_size not in (1, 8) or (tp_size == 8 and not native_tp8):\n        raise RuntimeError("Round9 reference model requires TP1 or gated Round12a TP8")\n    if native_tp8 and tp_size != 8:\n        raise RuntimeError("Round12a native TP8 mode requires exactly TP8")''',
        "Qwen TP guard",
    )
    source = replace_once(
        source,
        '''        self.reference_config = GDNConfig(\n            num_key_heads=self.num_k_heads,\n            num_value_heads=self.num_v_heads,\n            key_head_dim=self.head_k_dim,\n            value_head_dim=self.head_v_dim,\n            conv_kernel_size=self.conv_kernel_size,\n            eps=self.layer_norm_epsilon,\n        )''',
        '''        if (self.num_k_heads % self.attn_tp_size
                or self.num_v_heads % self.attn_tp_size):
            raise RuntimeError("GDN heads must divide the attention TP size")
        self.reference_config = GDNConfig(
            num_key_heads=self.num_k_heads // self.attn_tp_size,
            num_value_heads=self.num_v_heads // self.attn_tp_size,
            key_head_dim=self.head_k_dim,
            value_head_dim=self.head_v_dim,
            conv_kernel_size=self.conv_kernel_size,
            eps=self.layer_norm_epsilon,
        )''',
        "GDN TP-local reference geometry",
    )
    source = replace_once(
        source,
        "        z = z.reshape(-1, self.num_v_heads, self.head_v_dim)",
        "        z = z.reshape(-1, self.reference_config.num_value_heads, self.head_v_dim)",
        "GDN TP-local z reshape",
    )
    source = replace_once(
        source,
        '''        state = GDNState(conv_state=conv_state, recurrent_state=recurrent_state)\n        normalized, next_state = eager_gdn(''',
        '''        state = GDNState(conv_state=conv_state, recurrent_state=recurrent_state)\n        if (os.environ.get("ROUND12A_NATIVE_TP8_MODE") == "1"\n                and not getattr(self, "_round12a_gdn_recorded", False)):\n            rank = int(get_attention_tp_rank())\n            item = dict(\n                status="NATIVE_TP8_GDN_HIT", rank=rank, layer=int(self.layer_id),\n                key_heads=int(self.reference_config.num_key_heads),\n                value_heads=int(self.reference_config.num_value_heads),\n                z_heads=int(z.shape[1]), conv_state_shape=list(conv_state.shape),\n                recurrent_state_shape=list(recurrent_state.shape),\n                conv_state_device=conv_state.device.type,\n                recurrent_state_device=recurrent_state.device.type)\n            with Path(f"/out/native-tp8-rank{rank}.jsonl").open("a") as stream:\n                stream.write(json.dumps(item, allow_nan=False) + "\\n")\n            self._round12a_gdn_recorded = True\n        normalized, next_state = eager_gdn(''',
        "GDN warmup receipt",
    )
    source = replace_once(
        source,
        '''        self.ple = (\n            Qwen4ReferencePLE(config, layer_id)\n            if layer_id + 1 in set(config.ple_layer_ids)\n            else None\n        )''',
        '''        self.ple = (\n            Qwen4ReferencePLE(config, layer_id)\n            if layer_id + 1 in set(config.ple_layer_ids)\n            else None\n        )\n        self._round12a_layer_id = int(layer_id)\n        self._round12a_physical_recorded = False''',
        "layer receipt state",
    )
    source = replace_once(
        source,
        '''    def _after_attention(self, hidden, residual, forward_batch):\n        hidden = self.attn_hyper_connection.combine(hidden, residual)\n        hidden, residual = self.mlp_hyper_connection.mix(hidden)\n        hidden = self.mlp(hidden, forward_batch, False)\n        hidden = self.mlp_hyper_connection.combine(hidden, residual)\n        return hidden, None''',
        '''    def _after_attention(self, hidden, residual, forward_batch):\n        # RowParallel attention/GDN projections return TP-local partials.  The\n        # frozen upstream Qwen4 semantic donor reduces them before HC combine.\n        if get_attention_tp_size() > 1:\n            hidden = attn_tp_all_reduce(hidden)\n        hidden = self.attn_hyper_connection.combine(hidden, residual)\n        hidden, residual = self.mlp_hyper_connection.mix(hidden)\n        hidden = self.mlp(hidden, forward_batch, False)\n        if (os.environ.get("ROUND12A_NATIVE_TP8_MODE") == "1"\n                and not self._round12a_physical_recorded):\n            experts = self.mlp.experts\n            w13 = getattr(experts, "w13_weight", None)\n            w2 = getattr(experts, "w2_weight", None)\n            if not isinstance(w13, torch.Tensor) or not isinstance(w2, torch.Tensor):\n                raise RuntimeError("native TP8 expert physical tensors are missing")\n            rank = int(get_attention_tp_rank())\n            item = dict(\n                status="NATIVE_TP8_LAYER_PHYSICAL", rank=rank,\n                layer=self._round12a_layer_id,\n                w13_device=w13.device.type, w2_device=w2.device.type,\n                dtype=str(w13.dtype), w13_bytes=w13.numel()*w13.element_size(),\n                w2_bytes=w2.numel()*w2.element_size())\n            with Path(f"/out/native-tp8-rank{rank}.jsonl").open("a") as stream:\n                stream.write(json.dumps(item, allow_nan=False) + "\\n")\n            self._round12a_physical_recorded = True\n        hidden = self.mlp_hyper_connection.combine(hidden, residual)\n        return hidden, None''',
        "attention TP reduction",
    )
    source = replace_once(
        source,
        '''        _, selected = self.indexer(hidden_states, positions, forward_batch)\n        req_slot = forward_batch.req_pool_indices.long()[0]''',
        '''        _, selected = self.indexer(hidden_states, positions, forward_batch)\n        if (os.environ.get("ROUND12A_NATIVE_TP8_MODE") == "1"\n                and not getattr(self, "_round12a_old_qsa_recorded", False)):\n            rank = int(get_attention_tp_rank())\n            item = dict(\n                status="NATIVE_TP8_OLD_QSA_HIT", rank=rank,\n                layer=int(self.attn.layer_id), query_tokens=int(q.shape[0]),\n                query_heads=int(q.shape[1]), kv_heads=int(k.shape[1]))\n            with Path(f"/out/native-tp8-rank{rank}.jsonl").open("a") as stream:\n                stream.write(json.dumps(item, allow_nan=False) + "\\n")\n            self._round12a_old_qsa_recorded = True\n        req_slot = forward_batch.req_pool_indices.long()[0]''',
        "old QSA warmup receipt",
    )
    source = replace_once(
        source,
        '''        self.config = config\n        self.hidden_size = config.hidden_size\n        self.pp_group = get_pp_group()''',
        '''        self.config = config\n        if os.environ.get("ROUND12A_NATIVE_TP8_MODE") == "1":\n            expected = {\n                "hidden_size": 2560, "num_hidden_layers": 48,\n                "num_attention_heads": 24, "num_key_value_heads": 2,\n                "indexer_n_heads": 4, "indexer_kv_heads": 1,\n                "linear_num_key_heads": 16, "linear_num_value_heads": 48,\n                "num_experts": 512, "num_experts_per_tok": 10,\n                "indexer_compress_ratio": 4, "indexer_budget": 2048,\n                "hc_count": 4,\n            }\n            observed = {key: getattr(config, key, None) for key in expected}\n            qsa_layers = [\n                index for index, kind in enumerate(config.layers_block_type)\n                if kind in ("attention", "full_attention")\n            ]\n            if (observed != expected or list(config.ple_layer_ids) != [2]\n                    or qsa_layers != list(range(3, 48, 4))):\n                raise RuntimeError(\n                    f"Round12a native TP8 model geometry changed: "\n                    f"observed={observed}, ple={config.ple_layer_ids}, "\n                    f"qsa_layers={qsa_layers}")\n        self.hidden_size = config.hidden_size\n        self.pp_group = get_pp_group()''',
        "native TP8 model geometry",
    )
    compile(source, "qwen4_exp.py", "exec")
    return source


def patch_ple_binding(source: str) -> str:
    source = replace_once(
        source,
        "args.max_running_requests != 1 or args.tp_size != 1 or",
        "args.max_running_requests != 1 or args.tp_size not in (1, 8) or",
        "PLE TP geometry",
    )
    source = replace_once(
        source,
        '''    from .candidate_bank import bind_verified_banks\n    binding=bind_verified_banks(os.environ['ROUND9_CANDIDATE_PROFILE'],args.kt_weight_path,args.model_path)''',
        '''    if os.environ.get("ROUND12A_NATIVE_TP8_MODE") == "1":\n        if args.tp_size != 8:\n            raise RuntimeError("native TP8 PLE binding requires TP8")\n        from .round12a_native_tp8_identity import bind_native_tp8_source_identity\n        binding = bind_native_tp8_source_identity(args.model_path)\n    else:\n        from .candidate_bank import bind_verified_banks\n        binding=bind_verified_banks(os.environ['ROUND9_CANDIDATE_PROFILE'],args.kt_weight_path,args.model_path)''',
        "PLE source identity",
    )
    source = replace_once(
        source,
        "    log=Path('/out/ple-async-events.jsonl')",
        '''    if os.environ.get("ROUND12A_NATIVE_TP8_MODE") == "1":\n        from sglang.srt.layers.dp_attention import get_attention_tp_rank\n        log=Path(f"/out/ple-async-events-rank{int(get_attention_tp_rank())}.jsonl")\n    else:\n        log=Path('/out/ple-async-events.jsonl')''',
        "PLE rank event path",
    )
    compile(source, "ple_binding.py", "exec")
    return source


def patch_framework_hooks(source: str) -> str:
    old = '''        if os.environ.get('ROUND12_NONKT_MODE') == '1':\n            write_nonkt_physical()\n        else:\n            write_q32_physical(scheduler)'''
    new = '''        if os.environ.get('ROUND12A_NATIVE_TP8_MODE') == '1':\n            # Per-rank/all-layer NPU tensor receipts are emitted by qwen4_exp\n            # during the warmup request, after the TP-sharded weights are used.\n            pass\n        elif os.environ.get('ROUND12_NONKT_MODE') == '1':\n            write_nonkt_physical()\n        else:\n            write_q32_physical(scheduler)'''
    source = replace_once(source, old, new, "native framework physical branch")
    compile(source, "framework_hooks.py", "exec")
    return source


def source_manifest(root: Path) -> dict:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")
    }


def refresh_overlay(root: Path) -> dict:
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
        receipt["round12a_native_tp8"] = {
            "backend": "SGLANG_NATIVE_TP8_BF16",
            "qsa": "QUALIFIED_ORIGINAL",
            "qsa_sha256": OLD_QSA_SHA256,
            "tp_size": 8,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    finally:
        sys.path.remove(str(tools))


def _identity(version: int) -> tuple[str, str]:
    block = ROLES[version].lower()
    return (
        f"kt-round12a-native-tp8-{block}-v{version}",
        f"20260927-round12a-native-tp8-{block}-v{version}",
    )


def build(parent: Path, output: Path, inputs: Path, *, version: int, queue: Path) -> Path:
    if version not in ROLES:
        raise ValueError("unsupported native TP8 version")
    parent, output, inputs, queue = map(Path, (parent, output, inputs, queue))
    if output.exists():
        raise FileExistsError(output)
    frozen_inputs = inputs / "frozen-inputs"
    for name, expected in FROZEN_INPUT_SHA256.items():
        path = frozen_inputs / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen native TP8 build input changed: {name}")
    if sha256(queue) != FROZEN_INPUT_SHA256["ROUND12A_ABBA_QUEUE.json"]:
        raise ValueError("external queue path differs from the frozen native TP8 queue")
    queue_doc = json.loads(queue.read_text())
    expected_workload = {"W1": {"case_id": "edge64", "input_tokens": 64, "decode_tokens": 2}}
    if (
        queue_doc.get("status") != "FROZEN_ROUND12A_ABBA_QUEUE"
        or queue_doc.get("workloads") != ["W1"]
        or queue_doc.get("workload_definitions") != expected_workload
        or queue_doc.get("expected_output_ids") != {"W1": [3897, 248046]}
    ):
        raise ValueError("native TP8 requires the frozen Round12a W1 queue")
    shutil.copytree(parent, output, ignore=shutil.ignore_patterns("__pycache__"))
    for path in (output, *output.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    for name in (
        "controller-F.json", "execution-budget.lock", "host-controller.json",
        "round10-source-sha256.json", "round12-source-sha256.json",
        "ROUND12A_MODEL_CONTRACT.json",
    ):
        path = output / name
        if path.exists():
            path.unlink()

    overlay = output / "overlay-python/sglang/srt"
    qsa = overlay / "layers/round9/qsa.py"
    qwen = overlay / "models/qwen4_exp.py"
    if sha256(qsa) != OLD_QSA_SHA256 or sha256(qwen) != OLD_QWEN_SHA256:
        raise ValueError("parent is not the frozen v153 old-QSA tuple")
    qwen.write_text(patch_qwen_tp8(qwen.read_text()))
    ple = overlay / "layers/round9/ple_binding.py"
    ple.write_text(patch_ple_binding(ple.read_text()))
    hooks = overlay / "layers/round9/framework_hooks.py"
    hooks.write_text(
        patch_framework_hooks(
            (frozen_inputs / "framework_hooks.py").read_text()
        )
    )
    round9 = overlay / "layers/round9"
    shutil.copy2(
        inputs / "round12a_native_tp8_identity.py",
        round9 / "round12a_native_tp8_identity.py",
    )
    for filename in (
        "native_tp8_driver.py", "native_tp8_server.py", "native_tp8_launcher.py",
        "native_tp8_controller.py", "native_tp8_supervisor.py", "native_tp8_monitor.py",
        "native_tp8_admission.py", "native_tp8_image_preflight.py",
    ):
        shutil.copy2(inputs / filename, output / filename)
    for filename in (
        "NATIVE_TP8_MODEL_GEOMETRY.json",
        "NATIVE_TP8_UPSTREAM_SOURCE_PROOF.json",
        "NATIVE_TP8_SOURCE_CONTRACT.json",
        "NATIVE_TP8_RESOURCE_CONTRACT.json",
    ):
        shutil.copy2(inputs / filename, output / filename)
    shutil.copy2(frozen_inputs / "round11_t4_driver.py", output / "round11_t4_driver.py")
    shutil.copy2(frozen_inputs / "round10-t1-fixture.json", output / "round10-t1-fixture.json")
    shutil.copy2(frozen_inputs / "numa_policy.py", output / "numa_policy.py")
    shutil.copy2(queue, output / "ROUND12A_ABBA_QUEUE.json")

    source_receipt = json.loads((output / "rebound/bank-receipt.json").read_text())["source_receipt"]
    profile = {
        "model_path": SOURCE_IDENTITY["model_path"], "device": "npu", "tp_size": 8,
        "enable_multimodal": False, "dtype": "bfloat16", "load_format": "safetensors",
        "disable_cuda_graph": True, "disable_radix_cache": True,
        "disable_overlap_schedule": True, "disable_shared_experts_fusion": True,
        "max_running_requests": 1, "max_total_tokens": 4100, "context_length": 4098,
        "max_mamba_cache_size": 1, "max_prefill_tokens": 32, "page_size": 4,
        "chunked_prefill_size": 32, "attention_backend": "ascend",
        "mem_fraction_static": 0.70, "kt_weight_path": None, "cpu_offload_gb": 0,
        "host": "127.0.0.1", "port": 30091, "random_seed": 20260920,
        "watchdog_timeout": 1800, "skip_server_warmup": True,
        "served_model_name": "QWEN38-ROUND12A-NATIVE-TP8-OLD-QSA",
    }
    (output / "native-tp8-server-profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    framework = {
        "schema": 3, "resident_capacity": 32, "route_collection": False,
        "frequency": None, "prefill": None,
        "ple": {"mode": "ASYNC", "device_gate": {
            "path": "/round9/evidence/20260921-ple-v2/result.json",
            "sha256": "b753270612776c26b15d2c00e9bb9817bf9d86ca8bcf7e12c183e455ffa2e551",
        }},
    }
    (output / "native-tp8-framework-profile.json").write_text(json.dumps(framework, indent=2) + "\n")
    identity = {
        "schema": 1, "status": "FROZEN_ROUND12A_NATIVE_TP8_OLD_QSA",
        "source_identity": SOURCE_IDENTITY, "source_receipt": source_receipt,
        "tp_size": 8, "precision": "BF16", "attention_backend": "ascend",
        "qsa_mode": "QUALIFIED_ORIGINAL", "qsa_sha256": sha256(qsa),
        "promoted_qsa_rejected_sha256": PROMOTED_QSA_SHA256,
        "parent_qwen_sha256": OLD_QWEN_SHA256, "patched_qwen_sha256": sha256(qwen),
        "config_sha256": CONFIG_SHA256, "tp8_geometry": TP8_GEOMETRY,
        "ple_binding_sha256": sha256(ple), "framework_hooks_sha256": sha256(hooks),
        "loaded_text_bytes": LOADED_TEXT_BYTES,
        "mean_text_bytes_per_rank": (LOADED_TEXT_BYTES + 7) // 8,
        "expert_text_bytes": EXPERT_TEXT_BYTES,
        "non_expert_text_bytes": NON_EXPERT_TEXT_BYTES,
        "conservative_weight_bytes_per_rank": CONSERVATIVE_WEIGHT_BYTES_PER_RANK,
        "ple_external_table_bytes": PLE_EXTERNAL_TABLE_BYTES,
        "pure_npu_scope": "MAIN_AND_EXPERT_WEIGHTS_AND_COMPUTE_ON_EIGHT_NPUS",
        "host_dependency": "PLE_CHECKPOINT_ROW_TABLE_REMAINS_HOST_BACKED",
        "upstream_tp_semantic_donor": {
            "repository": "https://github.com/sgl-project/sglang",
            "commit": "4ccff141dbe992794f9da6c3aa23535b4f72000d",
            "contract": "attn_tp_all_reduce_before_attention_hyperconnection_combine",
        },
    }
    (output / "ROUND12A_NATIVE_TP8_IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n")
    name, evidence = _identity(version)
    contract = {
        "schema": 1, "status": "FROZEN_NOT_RUN",
        "stage": "ROUND12A_NATIVE_TP8_OLD_QSA_ABBA", "version": version,
        "mode": "NATIVE_TP8", "block": ROLES[version], "container": name,
        "evidence": evidence, "parent": "ROUND12_P2_BASELINE_V153_OLD_QSA",
        "qsa_source": "QUALIFIED_ORIGINAL", "tp_size": 8,
        "arm_semantics": "IDENTICAL_NATIVE_REPEAT_COHORT",
        "physical_devices": list(range(8)), "profiler": "OFF",
        "generation_limit": 4, "abort_limit": 0, "automatic_retries": 0,
        "raw_result": "native-tp8-raw.json",
        "driver_pass_status": "PASS_ROUND12A_NATIVE_TP8_DRIVER_RAW",
        "driver_command": [
            "{DRIVER_ROOT}/native_tp8_driver.py", "--version", str(version),
            "--block", ROLES[version], "--out", "/out", "--fixture",
            "{DRIVER_ROOT}/round10-t1-fixture.json", "--queue",
            "{DRIVER_ROOT}/ROUND12A_ABBA_QUEUE.json",
        ],
    }
    (output / "ROUND12A_NATIVE_TP8_CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")
    budget = {
        "schema": 1, "stage": contract["stage"], "version": version,
        "block": ROLES[version], "model_starts_limit": 1, "model_starts_used": 0,
        "starts": [], "generation_limit": 4, "abort_limit": 0,
        "automatic_retries_limit": 0,
    }
    (output / "execution-budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    refresh_overlay(output)
    manifest = source_manifest(output)
    (output / "round12-source-sha256.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    root = build(args.parent, args.output, args.inputs, version=args.version, queue=args.queue)
    print("PASS_ROUND12A_NATIVE_TP8_SNAPSHOT", args.version, sha256(root / "round12-source-sha256.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
