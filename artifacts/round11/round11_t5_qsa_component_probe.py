#!/usr/bin/env python3
"""Real-weight Ascend A/B for the single Round11 vectorized QSA candidate."""

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
PREFILL_CONTEXTS = (32, 64, 128)
SELECTION_WIDTH = 2051


def make_causal_selection(*, positions, context_tokens, width=SELECTION_WIDTH):
    if (
        positions.ndim != 1
        or type(context_tokens) is not int
        or context_tokens < 1
        or type(width) is not int
        or width < context_tokens
        or bool((positions < 0).any())
        or bool((positions >= context_tokens).any())
    ):
        raise ValueError("invalid causal selection fixture")
    ids = torch.arange(width, device=positions.device, dtype=torch.long)
    return torch.where(ids[None, :] <= positions[:, None], ids[None, :], -1)


def take_query_half(projected, *, num_heads=24, head_dim=256):
    expected = num_heads * head_dim * 2
    if projected.ndim != 2 or projected.shape[-1] != expected:
        raise ValueError("invalid QSA query/gate projection")
    return projected.reshape(-1, num_heads, head_dim * 2)[..., :head_dim]


def _relative_l2(got, want):
    numerator = torch.linalg.vector_norm(got.double() - want.double())
    denominator = torch.linalg.vector_norm(want.double()).clamp_min(1e-30)
    return float(numerator / denominator)


def _hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _norm(tensor, weight):
    value = tensor.float()
    return (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        * (1 + weight.float())
    ).to(tensor.dtype)


def _measure(function, q, k, v, selected):
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    started = time.perf_counter_ns()
    start.record()
    output = function(q, k, v, selected)
    end.record()
    torch.npu.synchronize()
    return output, (time.perf_counter_ns() - started) / 1e6, float(start.elapsed_time(end))


def _paired_timings(baseline, candidate, q, k, v, selected, iterations):
    for function in (baseline, candidate, candidate, baseline):
        _measure(function, q, k, v, selected)
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
        _output, wall_ms, device_ms = _measure(
            functions[name], q, k, v, selected
        )
        wall[name].append(wall_ms)
        device[name].append(device_ms)
    return wall, device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--identity-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=9)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.iterations < 5 or args.iterations > 20:
        raise ValueError("iterations outside frozen component budget")
    from safetensors import safe_open
    import torch_npu

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from probe_identity import verify_probe_identity
    from runtime.qsa import sparse_gqa, sparse_gqa_candidate, text_rope

    identity = verify_probe_identity(
        model=args.model,
        receipt=args.identity_receipt,
        expected_model=Path("/models") / REVISION,
        expected_receipt_sha256="7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239",
        revision=REVISION,
        manifest_sha256="62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e",
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
        raise RuntimeError("QSA component is not isolated on physical12")

    report = {
        "schema": 1,
        "status": "RUNNING",
        "candidate": "QUERY_AWARE_VECTORIZED_SPARSE_GQA",
        "source_identity": identity,
        "physical_device": 12,
        "iterations": args.iterations,
        "precision": "BF16_INPUT_FP32_ATTENTION",
        "selection_width": SELECTION_WIDTH,
        "rows": [],
        "automatic_retries": 0,
    }
    names = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "q_norm.weight", "k_norm.weight")
    try:
        with torch.no_grad():
            for layer in LAYERS:
                prefix = f"model.language_model.layers.{layer}.self_attn."
                weights = {name: load(prefix + name).to("npu:0") for name in names}
                torch.manual_seed(17100 + layer)
                hidden = torch.randn(129, 2560, dtype=torch.bfloat16).to("npu:0")
                q_gate = hidden @ weights["q_proj.weight"].T
                q = take_query_half(q_gate)
                positions = torch.arange(129, device="npu:0")
                q = text_rope(_norm(q, weights["q_norm.weight"]), positions)
                k = text_rope(
                    _norm(
                        (hidden @ weights["k_proj.weight"].T).reshape(129, 2, 256),
                        weights["k_norm.weight"],
                    ),
                    positions,
                )
                v = (hidden @ weights["v_proj.weight"].T).reshape(129, 2, 256)
                fixtures = [
                    (context, 32, torch.arange(context - 32, context, device="npu:0"))
                    for context in PREFILL_CONTEXTS
                ]
                fixtures.append((129, 1, torch.tensor([128], device="npu:0")))
                for context, query_tokens, active_positions in fixtures:
                    selected = make_causal_selection(
                        positions=active_positions,
                        context_tokens=context,
                    )
                    active_q = q.index_select(0, active_positions)
                    keys = k[:context]
                    values = v[:context]
                    want = sparse_gqa(active_q, keys, values, selected)
                    got = sparse_gqa_candidate(active_q, keys, values, selected)
                    torch.npu.synchronize()
                    want_cpu = want.cpu()
                    got_cpu = got.cpu()
                    relative = _relative_l2(got_cpu, want_cpu)
                    if not torch.isfinite(got_cpu).all() or relative > 0.001:
                        raise RuntimeError("QSA vectorized numerical gate failed")
                    hashes = []
                    for _repeat in range(3):
                        repeat = sparse_gqa_candidate(
                            active_q, keys, values, selected
                        )
                        torch.npu.synchronize()
                        hashes.append(_hash(repeat.cpu()))
                    if len(set(hashes)) != 1:
                        raise RuntimeError("QSA vectorized repeat hash failed")
                    wall, device = _paired_timings(
                        sparse_gqa,
                        sparse_gqa_candidate,
                        active_q,
                        keys,
                        values,
                        selected,
                        args.iterations,
                    )
                    baseline_ms = statistics.median(wall["baseline"])
                    candidate_ms = statistics.median(wall["candidate"])
                    report["rows"].append({
                        "layer": layer,
                        "context_tokens": context,
                        "query_tokens": query_tokens,
                        "baseline_wall_ms": wall["baseline"],
                        "candidate_wall_ms": wall["candidate"],
                        "baseline_device_ms": device["baseline"],
                        "candidate_device_ms": device["candidate"],
                        "baseline_median_ms": baseline_ms,
                        "candidate_median_ms": candidate_ms,
                        "candidate_over_baseline": candidate_ms / baseline_ms,
                        "relative_l2": relative,
                        "candidate_repeat_hashes": hashes,
                    })
                    del (
                        selected, active_q, keys, values, want, got,
                        want_cpu, got_cpu, repeat,
                    )
                del weights, hidden, q_gate, q, k, v, positions, fixtures, active_positions
                torch.npu.synchronize()
                torch.npu.empty_cache()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        report.update(
            status="PASS_ROUND11_T5_QSA_COMPONENT",
            row_count=len(report["rows"]),
            peak_allocated=torch.npu.max_memory_allocated(),
            after_allocated=torch.npu.memory_allocated(),
            after_reserved=torch.npu.memory_reserved(),
        )
        if report["after_allocated"] != 0 or report["after_reserved"] != 0:
            raise RuntimeError("QSA component allocator did not return to baseline")
    except BaseException as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], len(report["rows"]), flush=True)


if __name__ == "__main__":
    main()
