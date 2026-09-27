#!/usr/bin/env python3
"""Deterministic Round13 static expert-placement definitions.

The functions in this module produce logical expert IDs only.  They do not
allocate tensors, copy weights, update NZ storage, or publish runtime maps.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Sequence


LAYERS = 48
EXPERTS_PER_LAYER = 512
BUDGET_EXPERTS = 1536
EXPERT_BF16_BYTES = 3 * 2560 * 640 * 2
NOMINAL_BF16_BYTES = BUDGET_EXPERTS * EXPERT_BF16_BYTES
RANDOM_ALGORITHM = "SHA256_COUNTER_PERMUTATION_V1"
RANDOM_DOMAIN = b"KT-ASCEND-ROUND13-RANDOM-GLOBAL-BUDGET-V1"


Profile = list[list[int]]


def _empty() -> Profile:
    return [[] for _ in range(LAYERS)]


def _from_pairs(pairs: Iterable[tuple[int, int]]) -> Profile:
    profile = _empty()
    for layer, expert in pairs:
        profile[layer].append(expert)
    for row in profile:
        row.sort()
    return profile


def validate_profile(profile: Sequence[Sequence[int]]) -> None:
    if len(profile) != LAYERS:
        raise ValueError(f"placement requires {LAYERS} layers")
    total = 0
    for layer, row in enumerate(profile):
        if not isinstance(row, (list, tuple)):
            raise TypeError(f"layer {layer} placement is not a sequence")
        if list(row) != sorted(row):
            raise ValueError(f"layer {layer} placement is not sorted")
        if len(row) != len(set(row)):
            raise ValueError(f"layer {layer} placement has duplicate experts")
        if any(type(expert) is not int or not 0 <= expert < EXPERTS_PER_LAYER for expert in row):
            raise ValueError(f"layer {layer} placement has invalid expert IDs")
        total += len(row)
    if total != BUDGET_EXPERTS:
        raise ValueError(f"placement budget {total} != {BUDGET_EXPERTS}")


def validate_counts(counts: Sequence[Sequence[int]]) -> None:
    if len(counts) != LAYERS:
        raise ValueError(f"route counts require {LAYERS} layers")
    for layer, row in enumerate(counts):
        if len(row) != EXPERTS_PER_LAYER:
            raise ValueError(f"route count layer {layer} requires {EXPERTS_PER_LAYER} experts")
        if any(type(value) is not int or value < 0 for value in row):
            raise ValueError(f"route count layer {layer} contains invalid values")


def uniform() -> Profile:
    """Round12a S0: logical experts 0..31 in every routed-expert layer."""
    profile = [list(range(32)) for _ in range(LAYERS)]
    validate_profile(profile)
    return profile


def frequency_global_budget(counts: Sequence[Sequence[int]]) -> Profile:
    """Global Top1536 by count with deterministic layer/expert tie-breaking."""
    validate_counts(counts)
    ranked = sorted(
        (
            (-counts[layer][expert], layer, expert)
            for layer in range(LAYERS)
            for expert in range(EXPERTS_PER_LAYER)
        )
    )
    profile = _from_pairs((layer, expert) for _, layer, expert in ranked[:BUDGET_EXPERTS])
    validate_profile(profile)
    return profile


def front_loading() -> Profile:
    """Three complete routed-expert layers followed by forty-five q=0 layers."""
    profile = [list(range(EXPERTS_PER_LAYER)) if layer < 3 else [] for layer in range(LAYERS)]
    validate_profile(profile)
    return profile


def _random_key(seed: int, layer: int, expert: int) -> bytes:
    if type(seed) is not int or seed < 0:
        raise ValueError("random placement seed must be a non-negative integer")
    payload = RANDOM_DOMAIN + b"|" + str(seed).encode() + b"|" + str(layer).encode() + b"|" + str(expert).encode()
    return hashlib.sha256(payload).digest()


def random_global_budget(seed: int = 42) -> Profile:
    """Version-stable hash permutation followed by sampling without replacement."""
    ranked = sorted(
        (
            (_random_key(seed, layer, expert), layer, expert)
            for layer in range(LAYERS)
            for expert in range(EXPERTS_PER_LAYER)
        )
    )
    profile = _from_pairs((layer, expert) for _, layer, expert in ranked[:BUDGET_EXPERTS])
    validate_profile(profile)
    return profile


def build_profiles(counts: Sequence[Sequence[int]]) -> dict[str, Profile]:
    profiles = {
        "uniform": uniform(),
        "frequency": frequency_global_budget(counts),
        "front_loading": front_loading(),
        "random_seed42": random_global_budget(42),
    }
    for profile in profiles.values():
        validate_profile(profile)
    return profiles


def profile_digest(profile: Sequence[Sequence[int]]) -> str:
    validate_profile(profile)
    encoded = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def hit_summary(profile: Sequence[Sequence[int]], counts: Sequence[Sequence[int]]) -> dict:
    validate_profile(profile)
    validate_counts(counts)
    total = sum(sum(row) for row in counts)
    hits = sum(
        counts[layer][expert]
        for layer, selected in enumerate(profile)
        for expert in selected
    )
    return {
        "route_hits": hits,
        "route_total": total,
        "route_hit_fraction": hits / total if total else None,
    }


def profile_receipt(name: str, profile: Sequence[Sequence[int]], counts: Sequence[Sequence[int]]) -> dict:
    validate_profile(profile)
    per_layer = [len(row) for row in profile]
    return {
        "name": name,
        "budget_experts": sum(per_layer),
        "nominal_bf16_bytes": NOMINAL_BF16_BYTES,
        "nominal_bf16_gib": NOMINAL_BF16_BYTES / 2**30,
        "per_layer_counts": per_layer,
        "selected_logical_experts": [list(row) for row in profile],
        "profile_sha256": profile_digest(profile),
        "dev_prediction": hit_summary(profile, counts),
    }


__all__ = (
    "LAYERS",
    "EXPERTS_PER_LAYER",
    "BUDGET_EXPERTS",
    "EXPERT_BF16_BYTES",
    "NOMINAL_BF16_BYTES",
    "RANDOM_ALGORITHM",
    "build_profiles",
    "frequency_global_budget",
    "front_loading",
    "hit_summary",
    "profile_digest",
    "profile_receipt",
    "random_global_budget",
    "uniform",
    "validate_counts",
    "validate_profile",
)
