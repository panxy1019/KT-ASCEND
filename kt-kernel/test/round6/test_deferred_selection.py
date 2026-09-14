"""Round 6 Deferred Expert selection contracts.

These tests exercise the existing kt-kernel selection mechanism directly.  A3
tests add the queue, transfer and incremental-merge lifecycle around it.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from kt_kernel.experts_base import BaseMoEWrapper


def _select(expert_ids, expert_scores, protected_k):
    wrapper = SimpleNamespace(num_experts=int(expert_ids.max().item()) + 1)
    return BaseMoEWrapper.select_deferred_experts(
        wrapper,
        expert_ids=expert_ids,
        expert_scores=expert_scores,
        protected_k=protected_k,
    )


def test_immediate_and_deferred_ids_partition_every_route():
    expert_ids = torch.tensor([[6, 8, 25, 36, 11, 12]], dtype=torch.int64)
    scores = torch.tensor([[0.40, 0.25, 0.15, 0.10, 0.06, 0.04]])

    immediate, deferred = _select(expert_ids, scores, protected_k=5)

    immediate_mask = immediate.ge(0)
    deferred_mask = deferred.ge(0)
    assert not torch.any(immediate_mask & deferred_mask)
    assert torch.all(immediate_mask | deferred_mask)
    assert deferred[0, 5].item() == 12
    assert torch.equal(immediate[immediate_mask], expert_ids[immediate_mask])
    assert torch.equal(deferred[deferred_mask], expert_ids[deferred_mask])


def test_prefill_selection_uses_batch_union_expert_protection():
    expert_ids = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 0],
        ],
        dtype=torch.int64,
    )
    scores = torch.tensor(
        [
            [0.60, 0.30, 0.10],
            [0.50, 0.40, 0.10],
        ]
    )

    immediate, deferred = _select(expert_ids, scores, protected_k=2)

    # Expert 0 is protected by row 0, so its lower-scored occurrence in row 1
    # is immediate too.  The current mechanism protects expert IDs across the
    # flattened batch, rather than enforcing exactly one deferred slot per row.
    assert immediate[1, 2].item() == 0
    assert deferred[1, 2].item() == -1
    assert deferred[0, 2].item() == 2


def test_zero_and_full_protection_boundaries_are_explicit():
    expert_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    scores = torch.tensor([[0.5, 0.3, 0.2]])

    immediate, deferred = _select(expert_ids, scores, protected_k=0)
    assert torch.all(immediate.eq(-1))
    assert torch.equal(deferred, expert_ids)

    immediate, deferred = _select(expert_ids, scores, protected_k=3)
    assert torch.equal(immediate, expert_ids)
    assert torch.all(deferred.eq(-1))


def test_equal_score_selection_is_repeat_deterministic_on_active_backend():
    expert_ids = torch.tensor([[6, 8, 25, 36, 11, 12]], dtype=torch.int64)
    scores = torch.ones((1, 6), dtype=torch.float32)

    reference = _select(expert_ids, scores, protected_k=5)
    for _ in range(100):
        candidate = _select(expert_ids, scores, protected_k=5)
        assert torch.equal(candidate[0], reference[0])
        assert torch.equal(candidate[1], reference[1])
