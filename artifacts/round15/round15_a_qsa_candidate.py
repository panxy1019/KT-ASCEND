#!/usr/bin/env python3
"""Pure policy and exact-index helpers for the Round15 QSA screen.

The qualified Round15 workload has at most 256 effective tokens.  With the
frozen ratio-four index and a 2,048-token budget, every completed compressed
group is selected.  Decode can therefore construct the complete causal set
without scoring or Top-K.  Longer histories are the only ones for which the
Lightning Indexer can affect membership; unsupported inputs retain the eager
provider before any state is changed.
"""

from __future__ import annotations


FROZEN_RATIO = 4
FROZEN_HEADS = 4
FROZEN_HEAD_DIM = 128
FROZEN_DTYPE = "bfloat16"


def select_strategy(
    *,
    query_tokens: int,
    key_groups: int,
    topk_blocks: int,
    ratio: int,
    n_heads: int,
    head_dim: int,
    dtype_name: str,
) -> str:
    """Return the only admitted implementation for the supplied geometry."""
    values = (query_tokens, key_groups, topk_blocks, ratio, n_heads, head_dim)
    if any(type(value) is not int for value in values):
        return "EAGER_REFERENCE"
    if (
        query_tokens != 1
        or key_groups < 0
        or topk_blocks < 1
        or ratio != FROZEN_RATIO
        or n_heads != FROZEN_HEADS
        or head_dim != FROZEN_HEAD_DIM
        or dtype_name != FROZEN_DTYPE
    ):
        return "EAGER_REFERENCE"
    if key_groups <= topk_blocks:
        return "FULL_SET_CAUSAL"
    return "LIGHTNING_INDEXER"


def full_set_logical_indices(
    *, logical_position: int, ratio: int, topk_blocks: int
) -> list[int]:
    """Build the frozen selected-token layout for a complete causal set.

    This helper is intentionally torch-free so the contract can be verified on
    the local development host.  The device implementation uses the same
    layout with ``arange`` and ``full`` on NPU.
    """
    if (
        type(logical_position) is not int
        or type(ratio) is not int
        or type(topk_blocks) is not int
        or logical_position < 0
        or ratio < 2
        or topk_blocks < 1
        or logical_position >= topk_blocks * ratio
    ):
        raise ValueError("logical position is outside the exact full-set path")
    width = topk_blocks * ratio + ratio - 1
    valid = list(range(logical_position + 1))
    return valid + [-1] * (width - len(valid))
