# Routing and Weight Contract

## Mathematical contract

For token `t`, routed global IDs `e[t,j]`, and router weights `w[t,j]`:

```text
y_cpu[t] = Σ(j where e[t,j] ∈ E_cpu) w[t,j] * Expert(e[t,j], x[t])
y_npu[t] = Σ(j where e[t,j] ∈ E_npu) w[t,j] * Expert(e[t,j], x[t])
y[t]     = y_cpu[t] + y_npu[t]
```

Unselected and opposite-owner contributions are zero.

## Where weights are applied

| Plane | Masking | Weight application | Merge reweights? |
|---|---|---|---:|
| CPU/LLAMAFILE | NPU IDs become `-1`; C++ also skips mask-true IDs | `operators/llamafile/moe.hpp:445-455` or `:717-735` | No |
| NPU fixture | CPU-owned weights become zero | `kt-kernel/python/hybrid_moe.py:187-197` | No |
| Coordinator | None | None; plain BF16 addition at `hybrid_moe.py:289-292` and `:306-317` | No |

Therefore each selected route weight is applied exactly once. Global router IDs are never converted to a second local-ID namespace during execution.

## Validation and edge cases

- ID and weight shapes must match and be rank-2.
- IDs must be integer and within `[0, num_experts)`.
- Weights must be floating point and finite; NaN is rejected.
- Tested weights: `1/0`, `0/1`, `0.5/0.5`, and `0.99/0.01`.
- Tested ownership/order: CPU-only `[0,2]`, NPU-only `[1,3]`, mixed `[0,1]`, reversed mixed `[3,2]`.

Independent references in `test_ascend_hybrid_sequential.py:60-76` compute every expert directly from source F32 weights and do not call placement partition or coordinator merge logic.
