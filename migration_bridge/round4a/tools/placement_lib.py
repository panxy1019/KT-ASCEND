"""Deterministic Round 4A placement construction and validation helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ANCHOR_LAYER = 17
ANCHOR_EXPERT = 8


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_dimensions(config: dict) -> tuple[int, int, list[int]]:
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["n_routed_experts"])
    first_moe = int(config.get("first_k_dense_replace", 0) or 0)
    frequency = config.get("moe_layer_freq", 1)
    if isinstance(frequency, list):
        moe_layers = [idx for idx, enabled in enumerate(frequency[:num_layers]) if enabled]
    else:
        frequency = int(frequency)
        if frequency <= 0:
            raise ValueError("moe_layer_freq must be positive")
        moe_layers = [
            idx
            for idx in range(num_layers)
            if idx >= first_moe and idx % frequency == 0
        ]
    if not moe_layers:
        raise ValueError("model config contains no MoE layers")
    return num_layers, num_experts, moe_layers


def ranked_experts(frequency: torch.Tensor, moe_layers: list[int]) -> dict[int, list[int]]:
    ranked = {}
    for layer in moe_layers:
        counts = frequency[layer].tolist()
        ranked[layer] = sorted(range(len(counts)), key=lambda expert: (-counts[expert], expert))
    return ranked


def quantile_layers(moe_layers: list[int], count: int, anchor_layer: int) -> list[int]:
    if count > len(moe_layers):
        raise ValueError(f"cannot choose {count} layers from {len(moe_layers)}")
    if count == 1:
        selected = [moe_layers[0]]
    else:
        indices = [round(i * (len(moe_layers) - 1) / (count - 1)) for i in range(count)]
        selected = list(dict.fromkeys(moe_layers[index] for index in indices))
    for candidate in moe_layers:
        if len(selected) == count:
            break
        if candidate not in selected:
            selected.append(candidate)
    if anchor_layer not in selected:
        replaceable = selected[1:-1] or selected
        victim = min(replaceable, key=lambda layer: (abs(layer - anchor_layer), layer))
        selected[selected.index(victim)] = anchor_layer
    selected = sorted(set(selected))
    if len(selected) != count:
        for candidate in sorted(moe_layers, key=lambda layer: (min(abs(layer - x) for x in selected), layer)):
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == count:
                break
    return sorted(selected)


def top_experts(
    ranking: list[int], count: int, *, anchor_expert: int | None = None
) -> list[int]:
    selected = ranking[:count]
    if anchor_expert is not None and anchor_expert not in selected:
        selected = [anchor_expert, *[expert for expert in ranking if expert != anchor_expert][: count - 1]]
    return sorted(selected)


def build_profiles(
    config: dict, frequency: torch.Tensor, anchor_layer: int = ANCHOR_LAYER, anchor_expert: int = ANCHOR_EXPERT
) -> tuple[dict[str, dict[int, list[int]]], dict[int, list[int]], list[int]]:
    num_layers, num_experts, moe_layers = model_dimensions(config)
    if tuple(frequency.shape) != (num_layers, num_experts):
        raise ValueError(
            f"frequency shape must be {(num_layers, num_experts)}, got {tuple(frequency.shape)}"
        )
    if anchor_layer not in moe_layers or not 0 <= anchor_expert < num_experts:
        raise ValueError("Round 3 anchor is outside the model MoE topology")
    rankings = ranked_experts(frequency, moe_layers)
    profiles: dict[str, dict[int, list[int]]] = {
        "p0": {anchor_layer: [anchor_expert]},
        "p1": {
            anchor_layer: top_experts(
                rankings[anchor_layer], 4, anchor_expert=anchor_expert
            )
        },
    }
    p2_layers = quantile_layers(moe_layers, 4, anchor_layer)
    p3_layers = quantile_layers(moe_layers, 8, anchor_layer)
    profiles["p2"] = {
        layer: top_experts(
            rankings[layer], 4, anchor_expert=anchor_expert if layer == anchor_layer else None
        )
        for layer in p2_layers
    }
    profiles["p3"] = {
        layer: top_experts(
            rankings[layer], 4, anchor_expert=anchor_expert if layer == anchor_layer else None
        )
        for layer in p3_layers
    }
    return profiles, rankings, moe_layers


def logical_count_for_profile(
    num_layers: int,
    num_experts: int,
    moe_layers: list[int],
    cpu_by_layer: dict[int, list[int]],
) -> torch.Tensor:
    logical_count = torch.zeros((1, num_layers, num_experts), dtype=torch.int64)
    logical_count[0, moe_layers, :] = 1
    for layer, experts in cpu_by_layer.items():
        logical_count[0, layer, experts] = 0
    return logical_count


def validate_profile(
    name: str,
    config: dict,
    cpu_by_layer: dict[int, list[int]],
    logical_count: torch.Tensor,
    anchor_layer: int = ANCHOR_LAYER,
    anchor_expert: int = ANCHOR_EXPERT,
) -> dict:
    num_layers, num_experts, moe_layers = model_dimensions(config)
    expected_cpu = {"p0": 1, "p1": 4, "p2": 16, "p3": 32}[name]
    expected_layers = {"p0": 1, "p1": 1, "p2": 4, "p3": 8}[name]
    if logical_count.dtype != torch.int64 or tuple(logical_count.shape) != (1, num_layers, num_experts):
        raise AssertionError("logical_count must be int64 with model-derived shape")
    if len(cpu_by_layer) != expected_layers:
        raise AssertionError(f"{name}: expected {expected_layers} CPU layers")
    if sum(len(experts) for experts in cpu_by_layer.values()) != expected_cpu:
        raise AssertionError(f"{name}: expected {expected_cpu} CPU experts")
    if anchor_expert not in cpu_by_layer.get(anchor_layer, []):
        raise AssertionError(f"{name}: Round 3 anchor L{anchor_layer}E{anchor_expert} missing")
    if any(layer not in moe_layers for layer in cpu_by_layer):
        raise AssertionError(f"{name}: CPU placement includes a dense layer")
    layer_rows = []
    for layer in range(num_layers):
        expected_cpu_ids = sorted(cpu_by_layer.get(layer, []))
        if len(set(expected_cpu_ids)) != len(expected_cpu_ids):
            raise AssertionError(f"{name}: duplicate CPU expert in layer {layer}")
        if any(expert < 0 or expert >= num_experts for expert in expected_cpu_ids):
            raise AssertionError(f"{name}: invalid expert ID in layer {layer}")
        row = logical_count[0, layer]
        if layer in moe_layers:
            actual_cpu_ids = torch.where(row == 0)[0].tolist()
            if actual_cpu_ids != expected_cpu_ids:
                raise AssertionError(
                    f"{name}: layer {layer} CPU IDs {actual_cpu_ids} != {expected_cpu_ids}"
                )
            if not bool(((row == 0) | (row == 1)).all()):
                raise AssertionError(f"{name}: non-binary logical_count")
            npu_count = int(row.sum().item())
            if npu_count != num_experts - len(expected_cpu_ids):
                raise AssertionError(f"{name}: physical NPU expert count mismatch")
            layer_rows.append(
                {
                    "layer": layer,
                    "cpu_experts": expected_cpu_ids,
                    "npu_expert_count": npu_count,
                }
            )
        elif bool(row.any()):
            raise AssertionError(f"{name}: dense layer {layer} must have zero frequency")
    gpu_mask = logical_count.sum(dim=0) > 0
    for layer in range(num_layers):
        if layer not in moe_layers:
            gpu_mask[layer, :] = True
    if gpu_mask.dtype != torch.bool or tuple(gpu_mask.shape) != (num_layers, num_experts):
        raise AssertionError("derived accelerator mask has invalid dtype/shape")
    return {
        "profile": name,
        "selected_layer_count": len(cpu_by_layer),
        "selected_cpu_expert_count": expected_cpu,
        "total_moe_experts": len(moe_layers) * num_experts,
        "total_npu_experts": len(moe_layers) * num_experts - expected_cpu,
        "gpu_experts_ratio": (len(moe_layers) * num_experts - expected_cpu) / (len(moe_layers) * num_experts),
        "layer_placements": [row for row in layer_rows if row["cpu_experts"]],
        "all_npu_layers": [layer for layer in moe_layers if layer not in cpu_by_layer],
        "mask_dtype": str(gpu_mask.dtype),
        "mask_shape": list(gpu_mask.shape),
    }
