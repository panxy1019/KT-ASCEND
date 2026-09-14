# Sequential Single-Layer Hybrid MoE

## Execution

`HybridMoECoordinator.forward_sequential` is at `kt-kernel/python/hybrid_moe.py:274-292`:

1. Validate placement, dtype, device, shapes, and token count.
2. Copy hidden/IDs/weights to the CPU contract (BF16/int64/float32).
3. Run the existing synchronous LLAMAFILE routed-expert path.
4. Run the NPU-owned expert fixture.
5. Synchronize NPU work, copy CPU contribution to NPU, and add the two BF16 contributions once.

The result exposes `output`, `cpu_contribution`, and `accelerator_contribution`, which makes opposite-owner exact-zero semantics directly testable.

## Correctness matrix

`test_ascend_hybrid_sequential.py` covers:

- CPU-only `[0,2]`, weights `[0.4,0.6]`: NPU contribution exactly zero.
- NPU-only `[1,3]`, weights `[0.4,0.6]`: CPU contribution exactly zero.
- mixed `[0,1]` and reversed mixed `[3,2]`.
- contribution sum equals returned output without reweighting.
- decode/prefill lengths 1, 8, and 32.
- four mappings and four weight edge pairs.

## Numerical result

Across the sequential mapping/routing matrix on A3:

- worst max absolute error: `0.000244140625`
- observed mean absolute error: at most approximately `5.19e-05`
- observed relative L2 error: at most approximately `0.00558`

All are below the acceptance limits `2e-3`, `2e-4`, and `2e-2`. Semantic zero checks and additive merge checks are exact.
