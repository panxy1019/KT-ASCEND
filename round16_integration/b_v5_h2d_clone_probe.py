"""Bounded device probe for the safetensors-mmap H2D staging workaround."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch_npu  # noqa: F401
from safetensors import safe_open


WEIGHT_NAME = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
EXPECTED_SHAPE = (10240, 2560)
SHARD_SIZES = (2048, 2048, 6144)


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.npu.set_device(0)
    started = time.monotonic()
    with torch.inference_mode(), safe_open(
        str(args.checkpoint), framework="pt", device="cpu"
    ) as handle:
        source = handle.get_tensor(WEIGHT_NAME)
        if tuple(source.shape) != EXPECTED_SHAPE:
            raise RuntimeError(f"unexpected source shape: {tuple(source.shape)}")
        source_storage = int(source.untyped_storage().nbytes())
        clone_started = time.monotonic()
        staged = source.clone(memory_format=torch.contiguous_format)
        clone_seconds = time.monotonic() - clone_started
        if staged.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
            raise RuntimeError("staged checkpoint tensor still aliases its mmap source")

        destination = torch.empty(
            (16384, 2560), dtype=torch.bfloat16, device="npu:0"
        )
        copy_started = time.monotonic()
        offset = 0
        for size in SHARD_SIZES:
            destination.narrow(0, offset, size).copy_(
                staged.narrow(0, offset, size)
            )
            offset += size
        torch.npu.synchronize()
        copy_seconds = time.monotonic() - copy_started

        observed = destination.narrow(0, 0, sum(SHARD_SIZES)).cpu()
        expected = staged.narrow(0, 0, sum(SHARD_SIZES))
        exact = torch.equal(observed, expected)
        payload = {
            "schema": 1,
            "status": "PASS_H2D_ANONYMOUS_CLONE" if exact else "FAIL_NUMERICAL",
            "passed": exact,
            "weight": WEIGHT_NAME,
            "source_shape": list(source.shape),
            "source_stride": list(source.stride()),
            "source_storage_bytes": source_storage,
            "source_device": str(source.device),
            "staged_aliases_source": False,
            "staged_storage_bytes": int(staged.untyped_storage().nbytes()),
            "destination_shape": list(destination.shape),
            "destination_device": str(destination.device),
            "shard_sizes": list(SHARD_SIZES),
            "clone_seconds": clone_seconds,
            "copy_and_sync_seconds": copy_seconds,
            "total_seconds": time.monotonic() - started,
            "expected_sha256": tensor_digest(expected),
            "observed_sha256": tensor_digest(observed),
            "allocator_allocated_bytes": int(torch.npu.memory_allocated()),
            "allocator_reserved_bytes": int(torch.npu.memory_reserved()),
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True), flush=True)
        if not exact:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
