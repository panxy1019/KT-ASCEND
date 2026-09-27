#!/usr/bin/env python3
"""Real-weight A/B for direct resident-BF16 QSA Decode on physical12."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch


REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
LAYERS = (3, 23, 47)
CONTEXTS = (64, 128, 512)
SELECTION_WIDTH = 2051


def make_causal_selection(position, *, width=SELECTION_WIDTH):
    if type(position) is not int or position < 0 or width <= position:
        raise ValueError("invalid causal selection fixture")
    ids = torch.arange(width, device="npu:0", dtype=torch.long)
    return torch.where(ids[None, :] <= position, ids[None, :], -1)


def take_query_half(projected, *, num_heads=24, head_dim=256):
    expected = num_heads * head_dim * 2
    if projected.ndim != 2 or projected.shape[-1] != expected:
        raise ValueError("invalid QSA query/gate projection")
    return projected.reshape(-1, num_heads, head_dim * 2)[..., :head_dim]


def relative_l2(got, want):
    return float(
        torch.linalg.vector_norm(got.double() - want.double())
        / torch.linalg.vector_norm(want.double()).clamp_min(1e-30)
    )


def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def norm(tensor, weight):
    value = tensor.float()
    return (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        * (1 + weight.float())
    ).to(tensor.dtype)


def measure(function):
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    began = time.perf_counter_ns()
    start.record()
    output = function()
    end.record()
    torch.npu.synchronize()
    return output, (time.perf_counter_ns() - began) / 1e6, float(
        start.elapsed_time(end)
    )


def paired_timings(baseline, candidate, iterations):
    for function in (baseline, candidate, candidate, baseline):
        measure(function)
    wall = {"baseline": [], "candidate": []}
    device = {"baseline": [], "candidate": []}
    functions = {"baseline": baseline, "candidate": candidate}
    pattern = ("baseline", "candidate", "candidate", "baseline")
    step = 0
    while any(len(values) < iterations for values in wall.values()):
        name = pattern[step % len(pattern)]
        step += 1
        if len(wall[name]) >= iterations:
            continue
        _output, wall_ms, device_ms = measure(functions[name])
        wall[name].append(wall_ms)
        device[name].append(device_ms)
    return wall, device


def allocator_returned_to_baseline(
    baseline_allocated, baseline_reserved, after_allocated, after_reserved
):
    return (
        after_allocated == baseline_allocated
        and after_reserved == baseline_reserved
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--identity-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 5 <= args.iterations <= 12:
        raise ValueError("iterations outside frozen component budget")

    from safetensors import safe_open
    import torch_npu

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from probe_identity import verify_probe_identity
    from runtime.context_profile import ContextStateIdentity
    from runtime.context_reference_runtime import (
        ContextReferenceRuntime,
        _consume_approved_candidate,
        _consume_reference_rows,
    )
    from runtime.qsa import sparse_gqa, sparse_gqa_candidate, text_rope

    source_identity = verify_probe_identity(
        model=args.model,
        receipt=args.identity_receipt,
        expected_model=Path("/models") / REVISION,
        expected_receipt_sha256=(
            "7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239"
        ),
        revision=REVISION,
        manifest_sha256=(
            "62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e"
        ),
    )
    weight_map = json.loads(
        (args.model / "model.safetensors.index.json").read_text()
    )["weight_map"]

    def load(key):
        with safe_open(
            str(args.model / weight_map[key]), framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(key)

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    acl = ctypes.CDLL(
        "/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so"
    )
    physical = ctypes.c_int32(-1)
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    if (
        acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical)) != 0
        or physical.value != 12
        or torch.npu.device_count() != 1
        or list(Path("/dev").glob("davinci[0-9]*")) != [Path("/dev/davinci12")]
    ):
        raise RuntimeError("Round13 B component is not isolated to physical12")

    # Initialize the provider before taking the allocator baseline.  The
    # baseline is typically zero after empty_cache, but recording it makes the
    # cleanup contract explicit and independent of provider initialization.
    torch.manual_seed(0)
    runtime_prime = torch.randn(1, device="npu:0")
    del runtime_prime
    torch.npu.synchronize()
    torch.npu.empty_cache()
    baseline_allocated = int(torch.npu.memory_allocated())
    baseline_reserved = int(torch.npu.memory_reserved())
    torch.npu.reset_peak_memory_stats()

    report = {
        "schema": 1,
        "status": "RUNNING",
        "candidate": "DIRECT_RESIDENT_BF16_ROW_DECODE",
        "block": args.block,
        "source_identity": source_identity,
        "physical_device": 12,
        "iterations": args.iterations,
        "precision": "BF16_INPUT_FP32_ATTENTION",
        "rows": [],
        "contract_checks": {},
        "automatic_retries": 0,
        "baseline_allocated": baseline_allocated,
        "baseline_reserved": baseline_reserved,
    }
    names = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "q_norm.weight",
        "k_norm.weight",
    )
    try:
        with torch.no_grad():
            for layer in LAYERS:
                identity = ContextStateIdentity(
                    f"round13-b-{args.block.lower()}-layer-{layer}",
                    None,
                    layer,
                    "QSA_KV",
                )
                prefix = f"model.language_model.layers.{layer}.self_attn."
                weights = {
                    name: load(prefix + name).to("npu:0") for name in names
                }
                seed = 21300 + layer + (1000 if args.block == "B" else 0)
                torch.manual_seed(seed)
                hidden = torch.randn(
                    max(CONTEXTS), 2560, dtype=torch.bfloat16
                ).to("npu:0")
                q_gate = hidden @ weights["q_proj.weight"].T
                all_q = take_query_half(q_gate)
                all_positions = torch.arange(max(CONTEXTS), device="npu:0")
                all_q = text_rope(
                    norm(all_q, weights["q_norm.weight"]), all_positions
                )
                all_k = text_rope(
                    norm(
                        (hidden @ weights["k_proj.weight"].T).reshape(
                            max(CONTEXTS), 2, 256
                        ),
                        weights["k_norm.weight"],
                    ),
                    all_positions,
                )
                all_v = (hidden @ weights["v_proj.weight"].T).reshape(
                    max(CONTEXTS), 2, 256
                )
                for context in CONTEXTS:
                    position = context - 1
                    positions = torch.tensor([position], device="npu:0")
                    q = all_q[position : position + 1]
                    keys = all_k[:context]
                    values = all_v[:context]
                    selected = make_causal_selection(position)

                    def baseline():
                        return _consume_reference_rows(
                            identity,
                            "BF16_COMPACT",
                            layer,
                            q,
                            keys,
                            values,
                            selected,
                            positions,
                            sparse_gqa_candidate,
                        )

                    def candidate():
                        output, strategy = _consume_approved_candidate(
                            identity,
                            "BF16_COMPACT",
                            layer,
                            q,
                            keys,
                            values,
                            selected,
                            positions,
                            sparse_gqa_candidate,
                        )
                        if strategy != "DIRECT_BF16_ROW":
                            raise RuntimeError(
                                f"Decode candidate dispatched {strategy}"
                            )
                        return output

                    oracle = sparse_gqa(q, keys, values, selected)
                    baseline_output = baseline()
                    candidate_output = candidate()
                    torch.npu.synchronize()
                    oracle_cpu = oracle.cpu()
                    baseline_cpu = baseline_output.cpu()
                    candidate_cpu = candidate_output.cpu()
                    baseline_error = relative_l2(baseline_cpu, oracle_cpu)
                    candidate_error = relative_l2(candidate_cpu, oracle_cpu)
                    if (
                        not torch.isfinite(candidate_cpu).all()
                        or baseline_error > 0.001
                        or candidate_error > 0.001
                    ):
                        raise RuntimeError("Round13 B numerical gate failed")
                    hashes = []
                    for _ in range(3):
                        repeat = candidate()
                        torch.npu.synchronize()
                        hashes.append(tensor_hash(repeat.cpu()))
                    if len(set(hashes)) != 1:
                        raise RuntimeError("Round13 B repeat hash failed")
                    wall, device = paired_timings(
                        baseline, candidate, args.iterations
                    )
                    baseline_median = statistics.median(wall["baseline"])
                    candidate_median = statistics.median(wall["candidate"])
                    report["rows"].append(
                        {
                            "layer": layer,
                            "context_tokens": context,
                            "query_tokens": 1,
                            "strategy": "DIRECT_BF16_ROW",
                            "baseline_wall_ms": wall["baseline"],
                            "candidate_wall_ms": wall["candidate"],
                            "baseline_device_ms": device["baseline"],
                            "candidate_device_ms": device["candidate"],
                            "baseline_median_ms": baseline_median,
                            "candidate_median_ms": candidate_median,
                            "candidate_over_baseline": (
                                candidate_median / baseline_median
                            ),
                            "baseline_relative_l2": baseline_error,
                            "candidate_relative_l2": candidate_error,
                            "repeat_hashes": hashes,
                        }
                    )
                    if layer == 3 and context == 128:
                        future = selected.clone()
                        future[0, 0] = context
                        try:
                            _consume_approved_candidate(
                                identity,
                                "BF16_COMPACT",
                                layer,
                                q,
                                keys,
                                values,
                                future,
                                positions,
                                sparse_gqa_candidate,
                            )
                        except ValueError as error:
                            if "future logical token" not in str(error):
                                raise
                        else:
                            raise RuntimeError("Decode candidate accepted future KV")
                        fallback, fallback_strategy = _consume_approved_candidate(
                            identity,
                            "BF16_COMPACT",
                            layer,
                            q[:, :12],
                            keys,
                            values,
                            selected,
                            positions,
                            sparse_gqa_candidate,
                        )
                        fallback_oracle = sparse_gqa(
                            q[:, :12], keys, values, selected
                        )
                        torch.npu.synchronize()
                        if (
                            fallback_strategy != "ROW"
                            or relative_l2(fallback.cpu(), fallback_oracle.cpu())
                            > 0.001
                        ):
                            raise RuntimeError("unqualified Decode fallback failed")
                        report["contract_checks"].update(
                            per_query_causal_future_rejected=True,
                            unqualified_geometry_strategy=fallback_strategy,
                            unqualified_geometry_reference=True,
                        )
                        # The negative causal fixture is an NPU tensor.  It is
                        # no longer part of the lifecycle checks below and must
                        # not remain live when allocator cleanup is measured.
                        del future
                del weights, hidden, q_gate, all_q, all_k, all_v, all_positions
                torch.npu.synchronize()
                torch.npu.empty_cache()

            events = []
            lifecycle = ContextReferenceRuntime(
                "BF16_COMPACT",
                layers=(3,),
                approved_prefill_candidate=True,
                event=events.append,
            )
            token = object()
            lifecycle.begin("round13-b-lifecycle", token)
            lifecycle.consume(
                3, q, keys, values, selected, positions, sparse_gqa_candidate
            )
            lifecycle.after(token, finished=True)
            cancel_token = object()
            lifecycle.begin("round13-b-cancel", cancel_token)
            lifecycle.abort(cancel_token)
            cancel_ready = (
                lifecycle.state == "READY"
                and not lifecycle.active
                and lifecycle.request_token is None
            )
            stale_runtime = ContextReferenceRuntime(
                "BF16_COMPACT",
                layers=(3,),
                approved_prefill_candidate=True,
            )
            stale_token = object()
            stale_runtime.begin("round13-b-stale", stale_token)
            stale_runtime.abort(stale_token)
            stale_rejected = False
            try:
                stale_runtime.after(stale_token, finished=True)
            except RuntimeError:
                stale_rejected = True
            if (
                not stale_rejected
                or stale_runtime.state != "BROKEN_RELOAD_REQUIRED"
                or not cancel_ready
            ):
                raise RuntimeError("Round13 B cancel/lease contract failed")
            report["contract_checks"].update(
                complete_batch_drained=True,
                cancel_drained=True,
                stale_completion_rejected=True,
                stale_completion_fail_stop=True,
                runtime_ready_after_cancel=cancel_ready,
                event_statuses=[row["status"] for row in events],
            )
            del q, keys, values, selected, positions, oracle, baseline_output
            del candidate_output, repeat, fallback, fallback_oracle
            torch.npu.synchronize()
            torch.npu.empty_cache()

        ratios = [row["candidate_over_baseline"] for row in report["rows"]]
        report.update(
            status="PASS_ROUND13_B_QSA_DECODE_COMPONENT",
            row_count=len(report["rows"]),
            worst_candidate_over_baseline=max(ratios),
            all_rows_improve_10pct=all(ratio <= 0.90 for ratio in ratios),
            peak_allocated=torch.npu.max_memory_allocated(),
            after_allocated=torch.npu.memory_allocated(),
            after_reserved=torch.npu.memory_reserved(),
        )
        if not allocator_returned_to_baseline(
            baseline_allocated,
            baseline_reserved,
            report["after_allocated"],
            report["after_reserved"],
        ):
            raise RuntimeError("Round13 B allocator did not return to baseline")
    except BaseException as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], args.block, len(report["rows"]), flush=True)


if __name__ == "__main__":
    main()
