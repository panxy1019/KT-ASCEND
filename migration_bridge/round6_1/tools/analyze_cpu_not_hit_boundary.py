#!/usr/bin/env python3
"""Check zero CPU partial and NPU-only merge for a captured CPU-not-hit call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["tensor"] if isinstance(payload, dict) else payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--call", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checks = []
    for layer in (7, 8):
        suffix = f"call{args.call:04d}.pt"
        cpu = load(args.stage_dir / f"ktep_layer{layer:02d}_cpu_partial_{suffix}")
        npu = load(args.stage_dir / f"ktep_layer{layer:02d}_npu_owned_partial_{suffix}")
        merged = load(args.stage_dir / f"ktep_layer{layer:02d}_routed_merge_{suffix}")
        expected_merge = npu.to(merged.dtype)
        checks.append(
            {
                "layer": layer,
                "call": args.call,
                "shape": list(cpu.shape),
                "cpu_partial_exact_zero": bool(torch.count_nonzero(cpu).item() == 0),
                "merge_equals_npu_after_declared_cast": bool(torch.equal(merged, expected_merge)),
                "merge_max_abs": float((merged.float() - expected_merge.float()).abs().max().item()),
            }
        )
    result = {
        "schema_version": "round6.1-cpu-not-hit-boundary-v1",
        "checks": checks,
        "pass": all(
            item["cpu_partial_exact_zero"]
            and item["merge_equals_npu_after_declared_cast"]
            for item in checks
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
