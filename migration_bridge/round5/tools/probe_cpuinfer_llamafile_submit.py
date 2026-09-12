#!/usr/bin/env python3
"""Exercise CPUInfer submit/sync with a deterministic LLAMAFILE MoE task."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback

import torch
import torch_npu  # noqa: F401

from kt_kernel import kt_kernel_ext
from kt_kernel_ext.kvcache import ggml_type


def _hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    result: dict[str, object] = {
        "schema": "round5-cpuinfer-llamafile-submit-v1",
        "repeats": args.repeats,
        "threads": args.threads,
        "status": "FAIL",
    }

    try:
        torch.manual_seed(20260912)
        expert_num, top_k = 4, 2
        hidden_size = intermediate_size = 256
        worker_config = kt_kernel_ext.WorkerPoolConfig()
        worker_config.subpool_count = 1
        worker_config.subpool_numa_map = [0]
        worker_config.subpool_thread_count = [args.threads]
        cpu_infer = kt_kernel_ext.CPUInfer(worker_config)

        gate = (torch.randn(expert_num, intermediate_size, hidden_size) / 100).contiguous()
        up = (torch.randn(expert_num, intermediate_size, hidden_size) / 100).contiguous()
        down = (torch.randn(expert_num, hidden_size, intermediate_size) / 100).contiguous()
        owned_mask = torch.zeros(expert_num, dtype=torch.bool).contiguous()
        physical_to_logical = torch.arange(expert_num, dtype=torch.int64).contiguous()

        config = kt_kernel_ext.moe.MOEConfig(
            expert_num,
            top_k,
            hidden_size,
            intermediate_size,
            owned_mask.data_ptr(),
        )
        config.pool = cpu_infer.backend_
        config.m_block = 32
        config.group_min_len = 10
        config.group_max_len = 16
        config.max_len = 16
        config.gate_proj = gate.data_ptr()
        config.up_proj = up.data_ptr()
        config.down_proj = down.data_ptr()
        config.gate_type = ggml_type.FP32
        config.up_type = ggml_type.FP32
        config.down_type = ggml_type.FP32
        config.hidden_type = ggml_type.BF16

        moe = kt_kernel_ext.moe.MOE(config)
        cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
        cpu_infer.sync()

        qlen = torch.tensor([1], dtype=torch.int32)
        expert_ids = torch.tensor([[0, 2]], dtype=torch.int64).contiguous()
        weights = torch.tensor([[0.625, 0.375]], dtype=torch.float32).contiguous()
        hidden = torch.linspace(-1, 1, hidden_size, dtype=torch.float32).to(torch.bfloat16).reshape(1, -1).contiguous()
        output = torch.empty_like(hidden)

        hashes: set[str] = set()
        finite = True
        for _ in range(args.repeats):
            output.fill_(float("nan"))
            cpu_infer.submit(
                moe.forward_task(
                    qlen.data_ptr(),
                    top_k,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    hidden.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            cpu_infer.sync()
            finite = finite and bool(torch.isfinite(output.float()).all())
            hashes.add(_hash(output))

        result.update(
            {
                "unique_output_hashes": sorted(hashes),
                "deterministic": len(hashes) == 1,
                "finite": finite,
                "shape": {
                    "experts": expert_num,
                    "top_k": top_k,
                    "hidden": hidden_size,
                    "intermediate": intermediate_size,
                },
            }
        )
        if len(hashes) != 1 or not finite:
            raise RuntimeError("CPUInfer LLAMAFILE output was non-deterministic or non-finite")
        result["status"] = "PASS"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
