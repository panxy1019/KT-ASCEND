#!/usr/bin/env python3
"""Validate LLAMAFILE Deferred split and adjacent incremental merge on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from kt_kernel import KTMoEWrapper


def parse_ids(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    delta = actual.float() - expected.float()
    denominator = max(float(torch.linalg.vector_norm(expected.float())), 1e-12)
    return {
        "byte_exact": tensor_hash(actual) == tensor_hash(expected),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(delta)) / denominator,
    }


def make_wrapper(
    *, layer_idx: int, cpu_experts: list[int], gguf: Path, max_deferred: int
):
    gpu_mask = torch.ones(64, dtype=torch.bool)
    gpu_mask[cpu_experts] = False
    wrapper = KTMoEWrapper(
        layer_idx=layer_idx,
        num_experts=64,
        num_experts_per_tok=6,
        hidden_size=2048,
        moe_intermediate_size=1408,
        gpu_experts_mask=gpu_mask,
        cpuinfer_threads=16,
        threadpool_count=1,
        numa_nodes=[0],
        weight_path=str(gguf),
        chunked_prefill_size=512,
        method="LLAMAFILE",
        max_deferred_experts_per_token=max_deferred,
    )
    wrapper.load_weights()
    return wrapper


def run_task(
    wrapper,
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    initial: torch.Tensor | None = None,
    incremental: bool = False,
) -> torch.Tensor:
    qlen = torch.tensor([hidden.shape[0]], dtype=torch.int32)
    output = torch.zeros(hidden.shape, dtype=torch.float32)
    if initial is not None:
        output.copy_(initial)
    task = wrapper.moe.forward_task(
        qlen.data_ptr(),
        expert_ids.shape[-1],
        expert_ids.contiguous().data_ptr(),
        weights.contiguous().data_ptr(),
        hidden.contiguous().data_ptr(),
        output.data_ptr(),
        incremental,
    )
    wrapper.cpu_infer.submit(task)
    wrapper.cpu_infer.sync(0)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--layer7-cpu", default="16,38,41,50")
    parser.add_argument("--layer8-cpu", default="8,25,33,55")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    layer7_cpu = parse_ids(args.layer7_cpu)
    layer8_cpu = parse_ids(args.layer8_cpu)
    layer7 = make_wrapper(
        layer_idx=7, cpu_experts=layer7_cpu, gguf=args.gguf, max_deferred=1
    )
    layer8 = make_wrapper(
        layer_idx=8, cpu_experts=layer8_cpu, gguf=args.gguf, max_deferred=0
    )

    generator = torch.Generator().manual_seed(6007)
    hidden7 = torch.randn((1, 2048), generator=generator).to(torch.bfloat16)
    hidden8 = torch.randn((1, 2048), generator=generator).to(torch.bfloat16)
    ids7 = torch.tensor([[16, 11, 12, 13, 14, 38]], dtype=torch.int64)
    weights7 = torch.tensor([[0.30, 0.24, 0.18, 0.13, 0.09, 0.06]], dtype=torch.float32)
    ids8 = torch.tensor([[8, 15, 17, 19, 21, 23]], dtype=torch.int64)
    weights8 = torch.tensor([[0.35, 0.23, 0.17, 0.11, 0.08, 0.06]], dtype=torch.float32)

    immediate_ids, deferred_ids = layer7.select_deferred_experts(
        ids7, weights7, protected_k=5
    )
    full7 = run_task(layer7, hidden7, ids7, weights7)
    immediate7 = run_task(layer7, hidden7, immediate_ids, weights7)
    deferred7 = run_task(layer7, hidden7, deferred_ids, weights7)
    full8 = run_task(layer8, hidden8, ids8, weights8)
    incremental8 = run_task(
        layer8,
        hidden8,
        ids8,
        weights8,
        initial=deferred7,
        incremental=True,
    )

    split_metrics = metrics(immediate7 + deferred7, full7)
    incremental_metrics = metrics(incremental8, full8 + deferred7)
    recovered_deferred_metrics = metrics(incremental8 - full8, deferred7)

    repeat_hashes = []
    for _ in range(args.repeats):
        repeat_hashes.append(
            tensor_hash(
                run_task(
                    layer8,
                    hidden8,
                    ids8,
                    weights8,
                    initial=deferred7,
                    incremental=True,
                )
            )
        )

    payload = {
        "schema_version": 1,
        "gguf_sha256": file_sha256(args.gguf),
        "layer7_cpu_experts": layer7_cpu,
        "layer8_cpu_experts": layer8_cpu,
        "layer7_ids": ids7.tolist(),
        "layer7_immediate_ids": immediate_ids.tolist(),
        "layer7_deferred_ids": deferred_ids.tolist(),
        "layer8_ids": ids8.tolist(),
        "split_reconstruction": split_metrics,
        "incremental_reconstruction": incremental_metrics,
        "recovered_deferred_contribution": recovered_deferred_metrics,
        "repeat_count": args.repeats,
        "unique_repeat_hashes": sorted(set(repeat_hashes)),
    }
    payload["pass"] = bool(
        split_metrics["relative_l2"] <= 1e-6
        and incremental_metrics["relative_l2"] <= 1e-6
        and recovered_deferred_metrics["relative_l2"] <= 1e-6
        and len(payload["unique_repeat_hashes"]) == 1
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
