#!/usr/bin/env python3
"""Run deterministic DeepSeek-V2-Lite-shaped synthetic CPU GEMM diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_gemm_backend_probe import CBlasBackend, metrics, parse_backend  # noqa: E402


SHAPES = {
    "gate": (1408, 2048),
    "up": (1408, 2048),
    "down": (2048, 1408),
}
PATTERNS = ("random", "small", "large", "near_cancellation", "sparse")


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def make_operands(pattern: str, output_size: int, reduction_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((1, reduction_size), generator=generator)
    weight = torch.randn((output_size, reduction_size), generator=generator)
    if pattern == "small":
        x.mul_(2.0**-10); weight.mul_(2.0**-10)
    elif pattern == "large":
        x.mul_(32.0); weight.mul_(32.0)
    elif pattern == "near_cancellation":
        base = torch.linspace(-1.0, 1.0, reduction_size)
        x = base.unsqueeze(0)
        weight = torch.ones((output_size, reduction_size))
        weight[:, 1::2].mul_(-1.0)
        weight.add_(torch.randn(weight.shape, generator=generator) * 2.0**-12)
    elif pattern == "sparse":
        x[torch.rand(x.shape, generator=generator) > 0.05] = 0
        weight[torch.rand(weight.shape, generator=generator) > 0.05] = 0
    elif pattern != "random":
        raise ValueError(pattern)
    return x.to(torch.bfloat16).float().contiguous(), weight.to(torch.bfloat16).float().contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", action="append", required=True, help="NAME:LIBRARY:THREADS")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    backends = [CBlasBackend(*parse_backend(value)) for value in args.backend]
    rows = []
    for shape_index, (projection, (output_size, reduction_size)) in enumerate(SHAPES.items()):
        for pattern_index, pattern in enumerate(PATTERNS):
            x, weight = make_operands(pattern, output_size, reduction_size, 4100 + shape_index * 10 + pattern_index)
            reference = (x.double() @ weight.double().T).to(torch.bfloat16).float()
            row = {
                "projection": projection,
                "pattern": pattern,
                "x_shape": list(x.shape),
                "weight_shape": list(weight.shape),
                "x_sha256": tensor_hash(x),
                "weight_sha256": tensor_hash(weight),
                "backends": {},
            }
            for backend in backends:
                outputs = [backend.gemm(x, weight).to(torch.bfloat16).float() for _ in range(args.repeats)]
                row["backends"][backend.name] = {
                    "deterministic": all(torch.equal(outputs[0], value) for value in outputs[1:]),
                    "output_sha256": tensor_hash(outputs[0]),
                    "vs_fp64_bf16_reference": metrics(outputs[0], reference),
                }
            rows.append(row)
            print(f"{projection} {pattern}", flush=True)
    payload = {"schema_version": 1, "repeats": args.repeats, "patterns": list(PATTERNS), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
