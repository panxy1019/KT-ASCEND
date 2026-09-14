#!/usr/bin/env python3
"""Verify exact NPU-only merge behavior on unique real captured Layer17 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def canonical_sha(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--cpu-experts", default="6,8,25,36")
    parser.add_argument("--minimum", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu_experts = {int(value) for value in args.cpu_experts.split(",") if value}
    rows, seen = [], set()
    for path in sorted(args.capture_dir.glob("layer17-pass*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {"hidden_states", "topk_ids", "topk_weights", "cpu_output", "gpu_output", "merged_output"}
        if not required.issubset(payload):
            raise RuntimeError(f"instrumentation fields missing in {path.name}: {sorted(required - payload.keys())}")
        for index, (hidden, ids, weights, cpu, gpu, merged) in enumerate(zip(
            payload["hidden_states"], payload["topk_ids"], payload["topk_weights"],
            payload["cpu_output"], payload["gpu_output"], payload["merged_output"],
        )):
            if cpu_experts.intersection(int(value) for value in ids.tolist()):
                continue
            identity = tensor_sha(hidden)
            if identity in seen:
                continue
            seen.add(identity)
            cpu_zero = bool(torch.equal(cpu, torch.zeros_like(cpu)))
            merge_exact = bool(torch.equal(merged, gpu.to(merged.dtype)))
            rows.append({
                "capture": path.name, "row": index, "hidden_sha256": identity,
                "topk_ids": [int(value) for value in ids],
                "topk_weights": [float(value) for value in weights],
                "cpu_output_all_zero_exact": cpu_zero,
                "gpu_output_sha256": tensor_sha(gpu), "merged_output_sha256": tensor_sha(merged),
                "merged_equals_gpu_exact": merge_exact,
            })
    selected = rows[: args.minimum]
    passed = len(selected) >= args.minimum and all(
        row["cpu_output_all_zero_exact"] and row["merged_equals_gpu_exact"] for row in selected
    )
    payload = {
        "schema_version": 1, "evidence": "real production-captured Layer17 hidden/routes",
        "cpu_experts": sorted(cpu_experts), "minimum_required": args.minimum,
        "eligible_unique_positions": len(rows), "verified_positions": len(selected),
        "all_cpu_output_zero_exact": all(r["cpu_output_all_zero_exact"] for r in selected),
        "all_merged_equals_gpu_exact": all(r["merged_equals_gpu_exact"] for r in selected),
        "status": "A3_VERIFIED_READY" if passed else "BLOCKED", "rows": selected,
    }
    payload["sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit("CPU_NOT_HIT_MISMATCH or insufficient real positions")


if __name__ == "__main__":
    main()
