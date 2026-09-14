"""Round 6 Deferred Expert selection contracts.

These tests exercise the existing kt-kernel selection mechanism directly.  A3
tests add the queue, transfer and incremental-merge lifecycle around it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from kt_kernel import experts_base
from kt_kernel.experts_base import (
    BaseMoEWrapper,
    _has_cpu_routes,
    _summarize_cpu_routes,
    _trace_last_row_cpu_routes,
)


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


def test_deferred_trace_reports_only_cpu_owned_routes(monkeypatch, tmp_path):
    route_ids = torch.tensor([[6, -1, 8, 25, 36, 11]], dtype=torch.int64)
    gpu_mask = torch.ones(64, dtype=torch.bool)
    gpu_mask[[6, 8, 25, 36]] = False

    valid, cpu, cpu_ids = _summarize_cpu_routes(route_ids, gpu_mask)
    assert (valid, cpu, cpu_ids) == (5, 4, [6, 8, 25, 36])
    assert _has_cpu_routes(route_ids, gpu_mask)
    assert not _has_cpu_routes(torch.tensor([[11, -1]]), gpu_mask)
    assert not _has_cpu_routes(torch.full((1, 6), -1), gpu_mask)

    trace_path = tmp_path / "deferred.jsonl"
    monkeypatch.setenv("KT_DEBUG_DEFERRED_TRACE_FILE", str(trace_path))
    monkeypatch.setenv("KT_DEBUG_DEFERRED_TRACE_LIMIT", "1")
    monkeypatch.setattr(experts_base, "_deferred_trace_count", 0)
    experts_base._trace_deferred_event("submit", layer_idx=7, cpu_routes=cpu)
    experts_base._trace_deferred_event("sync", layer_idx=7)

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert records == [{"cpu_routes": 4, "event": "submit", "layer_idx": 7}]


def test_incremental_consumer_clears_previous_layer_pending():
    wrapper = SimpleNamespace(layer_idx=8)
    BaseMoEWrapper._layer_has_pending_deferred = {7: True, 8: False}

    BaseMoEWrapper._clear_consumed_previous_pending(wrapper, incremental=True)
    assert BaseMoEWrapper._layer_has_pending_deferred == {7: False, 8: False}

    BaseMoEWrapper._layer_has_pending_deferred[7] = True
    BaseMoEWrapper._clear_consumed_previous_pending(wrapper, incremental=False)
    assert BaseMoEWrapper._layer_has_pending_deferred[7]


def test_trace_target_row_excludes_prefill_cpu_routes():
    gpu_mask = torch.ones(64, dtype=torch.bool)
    gpu_mask[[6, 8, 25, 36]] = False
    ids = torch.tensor([[6, 8, -1], [11, 25, 36]], dtype=torch.long)
    weights = torch.tensor([[0.5, 0.4, 0.0], [0.6, 0.3, 0.1]])

    cpu_ids, cpu_weights = _trace_last_row_cpu_routes(ids, weights, gpu_mask)

    assert cpu_ids == [25, 36]
    assert cpu_weights == [float(weights[1, 1]), float(weights[1, 2])]
