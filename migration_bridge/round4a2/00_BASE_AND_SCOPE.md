# Round 4A.2 Base and Scope

## Repository freeze

- Round 4A.1 parent final / Round 4A.2 base: `ba508e4f920e99cd8cf1c0127d9aa6e5e0ac2559`
- parent branch: `feature/kt-round4a2-cpu-gemm-numerical-closure`
- SGLang Round 4A.1 final / Round 4A.2 base: `8a4bb4c850860377d9880b8c0fa570ca2d0d19e1`
- child branch: `feature/kt-ep-round4a2-cpu-gemm-numerical-closure`
- SGLang production diff: none

The two Round 4A.1 production fixes—BF16-visible expert boundaries and FP32
Hybrid partial accumulation—were retained throughout every experiment.

## Frozen scope

- model: DeepSeek-V2-Lite, TP=1, BF16, Graph OFF
- placement: Layer 17 CPU experts `{6,8,25,36}`, all other routed experts NPU
- placement SHA256: `f6d4e9c6a2e5e8060e846dbc7c628d069c9aa6150aaa1e2690ea28a51ba286a3`
- P1 corpus SHA256: `9ae547d3fef84f097b71eb944952e708298168be2083d1b1ce4faff76d03268e`
- P1 gate: 45/45 exact tokens, finite outputs, prefix determinism, max absolute logprob delta <= 0.20

## A3 isolation

All builds and runs occurred in disposable container `kt-r4a2-gemm`, restricted
to CPU set `0-15` and Ascend NPU 0. Image:

`quay.io/ascend/verl:v0.8.0-cann9.0.0-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-vllm`

The source and model host mounts were read-only; a writable source copy lived
inside the container layer. No host package, system library, business
container, or NPU1+ was modified.

## Ordered stop result

P1 remained 42/45 after the only candidate that passed the isolated gate was
integrated. Consequently P2 and P3 were not run.
