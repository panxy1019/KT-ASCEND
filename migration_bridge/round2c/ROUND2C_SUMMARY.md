# B Round 2C Summary

## Repository

- Round 2B frozen base: `6f50c3782f6002940dedcbbc74c6af980fc0d862`
- tag: `round2b-a3-verified`
- branch: `feature/kt-single-layer-hybrid-moe`
- Round 2C final: this documentation commit; use `git rev-parse HEAD`

## Placement

- fixed four-expert primary placement: CPU `{0,2}`, NPU `{1,3}`
- Ascend `gpu_experts_mask`: `True = NPU-owned`, `False = CPU-owned`
- exhaustive, disjoint ownership enforced by an immutable CPU bool mask
- invalid mask, mapping, route, NaN weight, deferred mode, and zero stream handle fail fast

## Routing

- router IDs remain global logical IDs
- `physical_to_logical_map[p] = global logical ID`
- CPU receives `-1` for NPU-owned routes; NPU receives zero weight for CPU-owned routes
- weights are applied once in each expert plane and never at merge

## CPU Expert

- existing `KTMoEWrapper -> LlamafileMoEWrapper -> CPUInfer -> LLAMAFILE MOE`
- BF16 hidden/output, int64 IDs, float32 routing weights
- no new CPU loader or C++ MoE kernel change

## NPU Expert

- deterministic ordinary torch_npu BF16 gate/up/down fixture
- exact SiLU/SwiGLU gate-up-down semantics
- resident weights reordered once into global-logical indexing
- no fused kernel, graph, TP/HCCL, quantization, full model, or private ABI

## Sequential Hybrid

- CPU-only, NPU-only, mixed, and reversed mixed: PASS
- mappings: identity, `[2,0,3,1]`, reverse, and `[1,3,0,2]`: PASS
- qlen 1/8/32 and route-weight edges: PASS
- worst max abs `0.000244140625`; mean approximately `<=5.19e-05`; relative L2 approximately `<=0.00558`

## Overlapped Hybrid

- callback-driven D2H → CPUInfer concurrent with NPU expert → callback sync → H2D → merge
- sequential/overlapped CPU, NPU, and merged contributions: bitwise equal for qlen 1/8/32
- final measured CPU interval `2.121210 ms`, NPU `2.359660 ms`, wall `3.134158 ms`
- conservative overlap lower bound `1.346712 ms`; proof only, no performance claim
- combined-matrix H2D/shared-buffer race found and fixed in isolated commit `b8ba787`

## Shared Expert

- `SHARED_EXPERT_OWNER = outer model layer`
- `SHARED_EXPERT = BYPASSED_WITH_CONTRACT`
- coordinator returns routed-only contribution; future outer integration adds shared exactly once

## Stability

- 1,000 mixed overlapped cycles: PASS, bitwise stable
- RSS `1,857,671,168 -> 1,857,671,168`, delta 0
- wrapper recreate ×20 with alternating masks: PASS
- NPU stream create/forward/destroy ×100: PASS

## Regression

- Round 2A final regression: **21 passed in 5.69s**
- Round 2B final regression: **23 passed, 1 expected CUDA-only skip in 27.50s**
- Round 2C final: two consecutive runs, **43 passed** each

## P0 Rules

- callback launches device work: NO
- callback synchronizes device: NO
- Graph/deferred/dynamic placement: OFF
- full model/SGLang integration/TP/HCCL: NOT IMPLEMENTED
- shared expert duplicate add: prevented by outer-owner contract
- host and business containers modified: NO

## Exit Gate

```text
CPU_EXPERT_PLANE = A3_VERIFIED_READY
ASCEND_RUNTIME_PLANE = A3_VERIFIED_READY
HYBRID_MOE_SINGLE_LAYER = A3_VERIFIED_READY
```

## Remaining blockers

No Round 2C synthetic single-layer blocker. This result does not claim full-model correctness, serving integration, performance, graph mode, TP/HCCL, deferred experts, dynamic placement, or shared-expert execution. Those remain later-round work. The prior generic-aarch64 ISA portability risk and non-privileged NUMA membind limitation remain unchanged.

## Commits

See `11_GIT_COMMITS.md`; Round 2C contains seven code/test/fix commits plus this documentation commit.
