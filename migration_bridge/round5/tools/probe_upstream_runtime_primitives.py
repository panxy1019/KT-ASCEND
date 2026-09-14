#!/usr/bin/env python3
"""Probe pinned upstream Ascend callback and framework-copy primitives.

This deliberately avoids model code.  It verifies only the runtime mechanisms
that must be sound before a DeepSeek-V2-Lite capability probe is attempted.
The JSON document is printed to stdout so the caller can freeze it verbatim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback

import torch
import torch_npu  # noqa: F401 - registers the NPU backend with torch

from kt_kernel import kt_kernel_ext


def _sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _stream_handle(stream: object) -> int:
    for name in ("npu_stream", "cuda_stream"):
        value = getattr(stream, name, None)
        if value is not None:
            return int(value)
    raise RuntimeError(f"cannot obtain native stream handle from {type(stream)!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--elements", type=int, default=4096)
    args = parser.parse_args()

    result: dict[str, object] = {
        "schema": "round5-upstream-runtime-primitives-v1",
        "repeats": args.repeats,
        "elements": args.elements,
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "kt_kernel_ext": getattr(kt_kernel_ext, "__file__", "unknown"),
        "status": "FAIL",
    }

    try:
        if torch.npu.device_count() != 1:
            raise RuntimeError(
                f"expected exactly one container-visible NPU, got {torch.npu.device_count()}"
            )
        torch.npu.set_device(0)
        device = torch.device("npu:0")
        stream = torch.npu.current_stream(device)
        stream_handle = _stream_handle(stream)

        required = (
            "init_ascend_callback_worker",
            "subscribe_ascend_stream",
            "shutdown_ascend_callback_worker",
            "is_ascend_callback_worker_running",
        )
        missing = [name for name in required if not hasattr(kt_kernel_ext, name)]
        if missing:
            raise RuntimeError(f"missing Ascend extension symbols: {missing}")

        before = bool(kt_kernel_ext.is_ascend_callback_worker_running())
        kt_kernel_ext.init_ascend_callback_worker()
        after_init = bool(kt_kernel_ext.is_ascend_callback_worker_running())
        kt_kernel_ext.subscribe_ascend_stream(stream_handle)
        after_subscribe = bool(kt_kernel_ext.is_ascend_callback_worker_running())
        if not after_init or not after_subscribe:
            raise RuntimeError("Ascend callback worker failed to reach running state")

        source = torch.arange(args.elements, dtype=torch.float32, device=device)
        expected_hash = _sha256(source)
        host = torch.empty(args.elements, dtype=torch.float32, pin_memory=True)
        device_return = torch.empty_like(source)

        d2h_hashes: set[str] = set()
        h2d_hashes: set[str] = set()
        for _ in range(args.repeats):
            host.copy_(source, non_blocking=True)
            torch.npu.synchronize(device)
            d2h_hashes.add(_sha256(host))

            device_return.copy_(host, non_blocking=True)
            torch.npu.synchronize(device)
            h2d_hashes.add(_sha256(device_return))

        kt_kernel_ext.shutdown_ascend_callback_worker()
        after_shutdown = bool(kt_kernel_ext.is_ascend_callback_worker_running())

        result.update(
            {
                "device_count": torch.npu.device_count(),
                "device": str(device),
                "stream_handle_nonzero": stream_handle != 0,
                "callback_running_before": before,
                "callback_running_after_init": after_init,
                "callback_running_after_subscribe": after_subscribe,
                "callback_running_after_shutdown": after_shutdown,
                "expected_hash": expected_hash,
                "d2h_unique_hashes": sorted(d2h_hashes),
                "h2d_unique_hashes": sorted(h2d_hashes),
                "d2h_deterministic": d2h_hashes == {expected_hash},
                "h2d_deterministic": h2d_hashes == {expected_hash},
            }
        )
        if after_shutdown:
            raise RuntimeError("Ascend callback worker remained running after shutdown")
        if d2h_hashes != {expected_hash} or h2d_hashes != {expected_hash}:
            raise RuntimeError("framework copy hash mismatch")
        result["status"] = "PASS"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # keep complete first-failure evidence
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
