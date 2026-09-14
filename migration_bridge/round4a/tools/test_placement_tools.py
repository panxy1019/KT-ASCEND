from __future__ import annotations

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from placement_lib import (  # noqa: E402
    build_profiles,
    logical_count_for_profile,
    model_dimensions,
    quantile_layers,
    ranked_experts,
    validate_profile,
)


def _config() -> dict:
    return {
        "num_hidden_layers": 27,
        "n_routed_experts": 64,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
    }


def _frequency() -> torch.Tensor:
    frequency = torch.zeros((27, 64), dtype=torch.int64)
    for layer in range(1, 27):
        for expert in range(64):
            frequency[layer, expert] = (layer * 17 + expert * 11) % 23
    return frequency


def test_ranking_tie_breaks_by_expert_id() -> None:
    frequency = torch.zeros((3, 4), dtype=torch.int64)
    frequency[1] = torch.tensor([7, 9, 9, 2])
    assert ranked_experts(frequency, [1])[1] == [1, 2, 0, 3]


def test_depth_quantiles_include_anchor() -> None:
    moe_layers = list(range(1, 27))
    assert quantile_layers(moe_layers, 4, 17) == [1, 9, 17, 26]
    assert quantile_layers(moe_layers, 8, 17) == [1, 5, 8, 12, 17, 19, 22, 26]


def test_profiles_have_exact_scale_and_anchor() -> None:
    config = _config()
    profiles, _, moe_layers = build_profiles(config, _frequency())
    assert {name: sum(map(len, rows.values())) for name, rows in profiles.items()} == {
        "p0": 1,
        "p1": 4,
        "p2": 16,
        "p3": 32,
    }
    for profile in profiles.values():
        assert 8 in profile[17]
    assert set(profiles["p2"]) == {1, 9, 17, 26}
    assert set(profiles["p3"]) == {1, 5, 8, 12, 17, 19, 22, 26}
    assert moe_layers == list(range(1, 27))


def test_mask_partition_and_physical_counts() -> None:
    config = _config()
    num_layers, num_experts, moe_layers = model_dimensions(config)
    profiles, _, _ = build_profiles(config, _frequency())
    for name, profile in profiles.items():
        logical_count = logical_count_for_profile(
            num_layers, num_experts, moe_layers, profile
        )
        result = validate_profile(name, config, profile, logical_count)
        for row in result["layer_placements"]:
            expected_npu = 64 - len(row["cpu_experts"])
            assert row["npu_expert_count"] == expected_npu
        assert result["mask_dtype"] == "torch.bool"
        assert result["mask_shape"] == [27, 64]


def test_validator_rejects_cpu_npu_overlap_encoding() -> None:
    config = _config()
    num_layers, num_experts, moe_layers = model_dimensions(config)
    profile = {17: [8]}
    logical_count = logical_count_for_profile(
        num_layers, num_experts, moe_layers, profile
    )
    logical_count[0, 17, 8] = 1
    try:
        validate_profile("p0", config, profile, logical_count)
    except AssertionError as error:
        assert "CPU IDs" in str(error)
    else:
        raise AssertionError("invalid placement unexpectedly passed")
