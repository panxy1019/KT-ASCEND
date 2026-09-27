"""Fail-closed binding for Round13's frozen global-budget placements.

This module owns only static startup placement.  It does not migrate experts,
change expert precision, update NZ tensors, or alter request-time routing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


LAYERS = 48
EXPERTS = 512
BUDGET = 1536
STRATEGIES = frozenset(("uniform", "frequency", "front_loading", "random_seed42"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_digest(rows) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_rows(rows) -> list[list[int]]:
    if not isinstance(rows, list) or len(rows) != LAYERS:
        raise ValueError("Round13 placement requires exactly48 rows")
    normalized = []
    for layer, row in enumerate(rows):
        if not isinstance(row, list) or row != sorted(row) or len(row) != len(set(row)):
            raise ValueError(f"Round13 placement row {layer} is not sorted and unique")
        if any(type(expert) is not int or not 0 <= expert < EXPERTS for expert in row):
            raise ValueError(f"Round13 placement row {layer} contains an invalid expert")
        normalized.append(list(row))
    if sum(map(len, normalized)) != BUDGET:
        raise ValueError("Round13 placement is not the frozen global1536 budget")
    return normalized


def load_profile(path: str | Path, expected_sha256: str, strategy: str) -> dict:
    path = Path(path)
    if strategy not in STRATEGIES:
        raise ValueError("unknown Round13 placement strategy")
    if not expected_sha256 or _sha256(path) != expected_sha256:
        raise RuntimeError("Round13 placement document SHA mismatch")
    document = json.loads(path.read_text())
    if (document.get("schema") != 2 or
            document.get("status") != "FROZEN_ROUND13_PLACEMENTS_FROM_FRESH_P0_DEV"):
        raise RuntimeError("Round13 placement document is not the fresh P0 freeze")
    entry = document.get("profiles", {}).get(strategy)
    if not isinstance(entry, dict) or entry.get("name") != strategy:
        raise RuntimeError("Round13 placement entry identity mismatch")
    rows = validate_rows(entry.get("selected_logical_experts"))
    counts = [len(row) for row in rows]
    if (entry.get("budget_experts") != BUDGET or
            entry.get("per_layer_counts") != counts or
            entry.get("profile_sha256") != _profile_digest(rows)):
        raise RuntimeError("Round13 placement entry receipt mismatch")
    return {
        "strategy": strategy,
        "rows": rows,
        "counts": counts,
        "document_sha256": expected_sha256,
        "profile_sha256": entry["profile_sha256"],
        "nominal_bf16_bytes": entry["nominal_bf16_bytes"],
    }


def profile_from_environment() -> dict | None:
    path = os.environ.get("ROUND13_PLACEMENT_PROFILE")
    digest = os.environ.get("ROUND13_PLACEMENT_PROFILE_SHA256")
    strategy = os.environ.get("ROUND13_PLACEMENT_STRATEGY")
    if not path and not digest and not strategy:
        return None
    if not path or not digest or not strategy:
        raise RuntimeError("Round13 placement path, SHA and strategy are required together")
    return load_profile(path, digest, strategy)


def bind_masks(args, wrapper, profile: dict) -> int:
    """Publish one immutable CPU bool mask before any KTEP layer allocates."""
    if (args.kt_num_gpu_experts != 32 or args.kt_gpu_experts_ratio is not None or
            args.kt_expert_placement_strategy != "uniform" or args.tp_size != 1 or
            args.kt_enable_dynamic_expert_update or
            (args.kt_max_deferred_experts_per_token or 0) != 0):
        raise RuntimeError("Round13 placement requires the frozen serial static S0 arguments")
    import torch
    masks = torch.zeros((LAYERS, EXPERTS), dtype=torch.bool, device="cpu")
    for layer, row in enumerate(profile["rows"]):
        if row:
            masks[layer, row] = True
    binding = getattr(wrapper, "_ROUND13_PLACEMENT_BINDING", None)
    expected = (profile["document_sha256"], profile["profile_sha256"], profile["strategy"])
    if binding is None:
        if wrapper._KT_GPU_EXPERTS_MASKS is not None:
            raise RuntimeError("Round13 placement must bind before KTEP mask initialization")
        wrapper._KT_GPU_EXPERTS_MASKS = masks
        wrapper._ROUND13_PLACEMENT_BINDING = expected
        print("ROUND13_PLACEMENT_BOUND", profile["strategy"], profile["profile_sha256"], flush=True)
    elif (binding != expected or wrapper._KT_GPU_EXPERTS_MASKS is None or
          not torch.equal(wrapper._KT_GPU_EXPERTS_MASKS, masks)):
        raise RuntimeError("Round13 placement changed during model allocation")
    # 32 remains the equivalent global-budget capacity used by unchanged S0
    # resource and bank gates; individual layer capacity comes from the mask.
    return 32


def validate_layer_config(cfg, layer: int, profile: dict, args) -> dict:
    if type(layer) is not int or not 0 <= layer < LAYERS:
        raise ValueError("invalid Round13 layer")
    expected = profile["counts"][layer]
    if expected == EXPERTS:
        if cfg is not None:
            raise RuntimeError("Q512 layer must use the original pure-NPU MoE method")
        provider = "PURE_NPU"
    else:
        if (cfg is None or int(cfg.gpu_experts_mask.sum().item()) != expected or
                cfg.layer_idx != layer or cfg.cpuinfer_threads != 128 or
                cfg.threadpool_count != 4 or list(cfg.numa_nodes) != [2, 3, 4, 7] or
                cfg.method != "LLAMAFILE" or (cfg.max_deferred_experts_per_token or 0) != 0 or
                cfg.kt_enable_dynamic_expert_update or cfg.gpu_prefill_token_threshold):
            raise RuntimeError("Round13 per-layer KTEP configuration mismatch")
        selected = cfg.gpu_experts_mask.nonzero().reshape(-1).tolist()
        if selected != profile["rows"][layer]:
            raise RuntimeError("Round13 per-layer logical map mismatch")
        provider = "KTEP_Q0" if expected == 0 else "KTEP_MIXED"
    return {"layer": layer, "resident_capacity": expected, "provider": provider}


def require_method_identity(method, layer: int, profile: dict, is_wrapped_method) -> dict:
    expected = profile["counts"][layer]
    wrapped = bool(is_wrapped_method(method, "kt_ep"))
    if wrapped != (expected < EXPERTS):
        raise RuntimeError("Round13 loaded MoE provider does not match the frozen placement")
    return {"layer": layer, "resident_capacity": expected,
            "provider": "KTEP" if wrapped else "PURE_NPU"}


def require_component_gate(path: str | Path, expected_sha256: str) -> dict:
    path = Path(path)
    if _sha256(path) != expected_sha256:
        raise RuntimeError("Round13 placement component gate SHA mismatch")
    gate = json.loads(path.read_text())
    expected = {
        "front_q512": ("front_loading", 0, 512, "PURE_NPU"),
        "front_q0": ("front_loading", 3, 0, "KTEP_Q0"),
        "frequency_q242": ("frequency", 2, 242, "KTEP_MIXED"),
        "random_q44": ("random_seed42", 44, 44, "KTEP_MIXED"),
    }
    if (gate.get("schema") != 1 or gate.get("status") != "PASS_ROUND13_A_COMPONENT_GATE" or
            set(gate.get("results", {})) != set(expected)):
        raise RuntimeError("Round13 placement component gate is incomplete")
    for name, identity in expected.items():
        row = gate["results"][name]
        if (row.get("status"), row.get("strategy"), row.get("layer"),
                row.get("capacity"), row.get("provider")) != (
                "PASS_ROUND13_PLACEMENT_COMPONENT", *identity):
            raise RuntimeError(f"Round13 component gate row failed: {name}")
        if (row.get("profile_document_sha256") != gate.get("placement_profiles_sha256") or
                len(set(row.get("repeat_hashes", []))) != 1 or
                not row.get("relative_l2") or max(row["relative_l2"]) > .01):
            raise RuntimeError(f"Round13 component numeric/identity gate failed: {name}")
    return gate


def collect_physical_layers(model, method_type, profile: dict) -> dict:
    """Inspect all 48 real expert owners after model loading."""
    owners = {}
    for name, module in model.named_modules():
        match = re.search(r"(?:^|\.)layers\.([0-9]+)\.mlp\.experts$", name)
        if match:
            index = int(match.group(1))
            if index in owners:
                raise RuntimeError("duplicate Round13 expert owner")
            owners[index] = module
    if set(owners) != set(range(LAYERS)):
        raise RuntimeError("Round13 physical record requires48 real expert owners")
    records = []
    total_bytes = 0
    import torch_npu
    for layer in range(LAYERS):
        owner = owners[layer]
        method = getattr(owner, "quant_method", None)
        expected = profile["counts"][layer]
        wrapped = isinstance(method, method_type)
        if wrapped != (expected < EXPERTS):
            raise RuntimeError("Round13 physical provider mismatch")
        tensors = {}
        for name, shape in (("w13_weight", [expected, 2560, 1280]),
                            ("w2_weight", [expected, 640, 2560])):
            value = getattr(owner, name, getattr(method, name, None))
            if (value is None or list(value.shape) != shape or str(value.dtype) != "torch.bfloat16" or
                    getattr(value.device, "type", None) != "npu"):
                raise RuntimeError(f"Round13 layer{layer} {name} physical tensor mismatch")
            if expected and int(torch_npu.get_npu_format(value)) != 29:
                raise RuntimeError(f"Round13 layer{layer} {name} is not NZ29")
            tensors[name] = {"shape": shape, "dtype": str(value.dtype),
                "format": int(torch_npu.get_npu_format(value)),
                "bytes": value.numel() * value.element_size()}
        selected = profile["rows"][layer]
        if wrapped:
            if (method.num_gpu_experts != expected or
                    method.gpu_index_to_logical.detach().cpu().tolist() != selected or
                    method.gpu_experts_mask.nonzero().reshape(-1).tolist() != selected):
                raise RuntimeError("Round13 KTEP physical map mismatch")
            provider = "KTEP_Q0" if expected == 0 else "KTEP_MIXED"
        else:
            provider = "PURE_NPU"
        layer_bytes = sum(tensor["bytes"] for tensor in tensors.values())
        total_bytes += layer_bytes
        records.append({"layer": layer, "resident_capacity": expected,
            "selected_logical_experts": selected, "provider": provider,
            "tensors": tensors, "total_bytes": layer_bytes})
    if total_bytes != profile["nominal_bf16_bytes"]:
        raise RuntimeError("Round13 actual device expert bytes differ from global budget")
    return {"schema": 1, "status": "PASS_ROUND13_STATIC_PLACEMENT_PHYSICAL",
        "strategy": profile["strategy"], "profile_sha256": profile["profile_sha256"],
        "document_sha256": profile["document_sha256"], "total_experts": BUDGET,
        "total_bytes": total_bytes, "layers": records}


__all__ = (
    "bind_masks", "collect_physical_layers", "load_profile", "profile_from_environment",
    "require_component_gate", "require_method_identity", "validate_layer_config", "validate_rows",
)
