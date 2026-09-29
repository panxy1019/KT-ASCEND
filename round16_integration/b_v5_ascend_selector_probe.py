"""Bounded device check for the Round16-B Ascend Top-K adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.utils import enable_custom_op

from round16_integration.b2_live_bindings import make_ascend_expert_selector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.npu.set_device(0)
    generator = torch.Generator(device="cpu").manual_seed(20260929)
    logits_cpu = torch.randn(
        (32, 512), generator=generator, dtype=torch.bfloat16
    )
    hidden = torch.randn(
        (32, 2560), generator=generator, dtype=torch.bfloat16
    ).to("npu:0")
    logits = logits_cpu.to("npu:0")
    owner = SimpleNamespace(
        use_grouped_topk=False,
        renormalize=True,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
    )
    runner = SimpleNamespace(
        routed_experts=owner,
        moe_config=SimpleNamespace(experts_per_token=10),
        _quant_method=SimpleNamespace(topk_indices_dtype=torch.int32),
    )
    adapter = make_ascend_expert_selector(
        select_experts,
        enable_custom_op=enable_custom_op,
    )
    with torch.inference_mode():
        weights, expert_ids = adapter(
            runner=runner,
            hidden_states=hidden,
            router_logits=logits,
            input_ids=None,
        )
        repeat_weights, repeat_expert_ids = adapter(
            runner=runner,
            hidden_states=hidden,
            router_logits=logits,
            input_ids=None,
        )
        torch.npu.synchronize()
        weights_cpu = weights.float().cpu()
        ids_cpu = expert_ids.long().cpu()
        repeat_weights_cpu = repeat_weights.cpu()
        repeat_ids_cpu = repeat_expert_ids.long().cpu()

    probabilities = logits_cpu.float().softmax(dim=-1)
    reference = logits_cpu.float().topk(10, dim=-1, sorted=True)
    reference_ids = reference.indices
    reference_sets = reference_ids.sort(dim=-1).values
    observed_sets = ids_cpu.sort(dim=-1).values
    ids_exact = torch.equal(observed_sets, reference_sets)
    ids_in_range = bool(((ids_cpu >= 0) & (ids_cpu < 512)).all())
    ids_unique = bool(
        (observed_sets[:, 1:] != observed_sets[:, :-1]).all()
    )
    boundary = reference.values[:, -1]
    observed_logits = logits_cpu.float().gather(1, ids_cpu)
    topk_boundary_valid = bool(
        (observed_logits >= boundary[:, None]).all()
    )
    reference_mask = torch.zeros((32, 512), dtype=torch.bool)
    observed_mask = torch.zeros((32, 512), dtype=torch.bool)
    reference_mask.scatter_(1, reference_ids, True)
    observed_mask.scatter_(1, ids_cpu, True)
    different = reference_mask ^ observed_mask
    boundary_only_difference = bool(
        (
            (~different)
            | (logits_cpu.float() == boundary[:, None])
        ).all()
    )
    tie_rows = int(
        (
            (logits_cpu.float() == boundary[:, None]).sum(dim=-1) > 1
        ).sum()
    )
    mismatch_rows = int((reference_mask != observed_mask).any(dim=-1).sum())
    ids_repeat_exact = torch.equal(ids_cpu, repeat_ids_cpu)
    weights_repeat_exact = torch.equal(weights.cpu(), repeat_weights_cpu)
    expected_weights = probabilities.gather(1, ids_cpu)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    relative_l2 = float(
        torch.linalg.vector_norm(weights_cpu - expected_weights)
        / torch.linalg.vector_norm(expected_weights)
    )
    finite = bool(torch.isfinite(weights_cpu).all())
    normalized_error = float((weights_cpu.sum(dim=-1) - 1.0).abs().max())
    passed = (
        ids_in_range
        and ids_unique
        and topk_boundary_valid
        and boundary_only_difference
        and ids_repeat_exact
        and weights_repeat_exact
        and finite
        and relative_l2 <= 0.01
        and normalized_error <= 0.01
    )
    payload = {
        "schema": 1,
        "status": (
            "PASS_ROUND16_B_ASCEND_SELECTOR_COMPONENT"
            if passed
            else "FAIL_ROUND16_B_ASCEND_SELECTOR_COMPONENT"
        ),
        "passed": passed,
        "logical_experts": 512,
        "top_k": 10,
        "tokens": 32,
        "ids_set_exact": ids_exact,
        "ids_in_range": ids_in_range,
        "ids_unique_per_token": ids_unique,
        "topk_boundary_valid": topk_boundary_valid,
        "boundary_only_set_difference": boundary_only_difference,
        "boundary_tie_rows": tie_rows,
        "strict_set_mismatch_rows": mismatch_rows,
        "ids_repeat_exact": ids_repeat_exact,
        "weights_repeat_exact": weights_repeat_exact,
        "ids_dtype": str(expert_ids.dtype),
        "weights_dtype": str(weights.dtype),
        "finite": finite,
        "relative_l2": relative_l2,
        "max_normalized_error": normalized_error,
        "allocator_allocated_bytes": int(torch.npu.memory_allocated()),
        "allocator_reserved_bytes": int(torch.npu.memory_reserved()),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
