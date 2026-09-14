"""A3 CPU-only CPUInfer + LLAMAFILE smoke/reference test.

This test uses the generic LLAMAFILE MOE binding directly with in-memory F32
weights. It does not use CUDA, NPU, SGLang, a model, or KML.
"""

from __future__ import annotations

import json
import math
import time

import torch

import kt_kernel
import kt_kernel_ext


def main() -> None:
    torch.manual_seed(20260827)
    torch.set_num_threads(1)

    hidden_size = 32
    intermediate_size = 256  # LLAMAFILE requires QK_K=256 alignment.
    expert_num = 2
    top_k = 1
    iterations = 1000

    pool_config = kt_kernel_ext.WorkerPoolConfig()
    pool_config.subpool_count = 1
    pool_config.subpool_numa_map = [0]
    pool_config.subpool_thread_count = [4]
    cpuinfer = kt_kernel_ext.CPUInfer(pool_config)

    # Small magnitudes keep the BF16 output away from overflow and make the
    # numerical comparison useful while still exercising all three GEMMs.
    gate = (torch.randn(expert_num, intermediate_size, hidden_size) * 0.02).contiguous()
    up = (torch.randn(expert_num, intermediate_size, hidden_size) * 0.02).contiguous()
    down = (torch.randn(expert_num, hidden_size, intermediate_size) * 0.02).contiguous()

    config = kt_kernel_ext.moe.MOEConfig(expert_num, top_k, hidden_size, intermediate_size)
    config.pool = cpuinfer.backend_
    config.max_len = 1
    config.group_min_len = 10  # batch=1 follows forward_one/decode.
    config.group_max_len = 1
    config.m_block = 32
    config.gate_proj = gate.data_ptr()
    config.up_proj = up.data_ptr()
    config.down_proj = down.data_ptr()
    config.gate_type = 0  # GGML_TYPE_F32
    config.up_type = 0
    config.down_type = 0
    config.hidden_type = 30  # GGML_TYPE_BF16

    moe = kt_kernel_ext.moe.MOE(config)
    cpuinfer.submit(moe.load_weights_task())
    cpuinfer.sync()

    x = torch.randn(1, hidden_size, dtype=torch.bfloat16).contiguous()
    expert_ids = torch.tensor([[1]], dtype=torch.int64)
    routing_weights = torch.tensor([[0.75]], dtype=torch.float32)
    qlen = torch.tensor([1], dtype=torch.int32)
    output = torch.empty_like(x)

    task_args = (
        qlen.data_ptr(),
        top_k,
        expert_ids.data_ptr(),
        routing_weights.data_ptr(),
        x.data_ptr(),
        output.data_ptr(),
    )

    start = time.perf_counter()
    for _ in range(iterations):
        cpuinfer.submit(moe.forward_task(*task_args))
        cpuinfer.sync()
    elapsed = time.perf_counter() - start

    expert = int(expert_ids.item())
    xf = x.float().squeeze(0)
    gate_out = gate[expert] @ xf
    up_out = up[expert] @ xf
    reference_f32 = down[expert] @ (torch.nn.functional.silu(gate_out) * up_out)
    reference_f32 *= float(routing_weights.item())
    reference_bf16 = reference_f32.to(torch.bfloat16).float()
    actual = output.float().squeeze(0)
    diff = actual - reference_bf16

    max_abs = float(diff.abs().max())
    mean_abs = float(diff.abs().mean())
    denom = float(torch.linalg.vector_norm(reference_bf16))
    relative_l2 = float(torch.linalg.vector_norm(diff)) / denom if denom else math.inf

    result = {
        "status": "PASS" if torch.isfinite(actual).all() else "FAIL",
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "microseconds_per_iteration": elapsed * 1e6 / iterations,
        "cpu_variant_metadata": kt_kernel.__cpu_variant__,
        "extension": kt_kernel_ext.__file__,
        "subpool_count": 1,
        "subpool_numa_map": [0],
        "subpool_thread_count": [4],
        "shape": {
            "batch": 1,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "experts": expert_num,
            "top_k": top_k,
        },
        "max_abs_error_vs_bf16_reference": max_abs,
        "mean_abs_error_vs_bf16_reference": mean_abs,
        "relative_l2_vs_bf16_reference": relative_l2,
        "all_finite": bool(torch.isfinite(actual).all()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
