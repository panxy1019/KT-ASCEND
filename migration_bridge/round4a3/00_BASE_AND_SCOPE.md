# Round 4A.3 Base and Scope

## Repository freeze

- Round 4A.2 parent final / Round 4A.3 parent base: `c2a456aec16846d353bd3075361d2cda6a3e085c`
- parent branch: `feature/kt-round4a3-numerical-acceptance`
- SGLang Round 4A.2 final / Round 4A.3 child base: `8a4bb4c850860377d9880b8c0fa570ca2d0d19e1`
- child branch: `feature/kt-ep-round4a3-numerical-acceptance`
- production-code change: **none**

Round 4A.3 retains the verified Round 4A.1 BF16-visible expert boundaries and
FP32 Hybrid partial accumulation. It does not search for, build, or integrate a
new GEMM backend.

## Frozen P1 scope

- model: DeepSeek-V2-Lite, TP=1, BF16, Graph OFF
- placement: Layer 17 CPU experts `{6,8,25,36}`; the other 60 routed experts are NPU-owned
- placement file SHA256: `c12d2954e0188ab9d7d1567085d342fe4b3ed5b769fdb976f743c2bbb50fc509`
- backend: LLAMAFILE / CPUInfer, 16 CPU workers, one threadpool, NUMA node 0
- placement strategy: `frequency` (explicitly frozen; the current parser default is `uniform`)
- deferred experts: 0
- dynamic placement: OFF

## A3 isolation

All A3 work runs in disposable container `kt-r4a3-acceptance`, restricted to
CPU set `0-15` and Ascend NPU0. The image is
`quay.io/ascend/verl:v0.8.0-cann9.0.0-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-vllm`.
The source and model mounts are read-only; `/workspace/kt-src` is the writable
container copy. No host library, business container, or NPU1+ is modified.

The container-only `libhwloc15` runtime dependency was installed after the
fresh container exposed the missing `libhwloc.so.15` required by the already
built, frozen `kt_kernel`. The package transaction log is preserved as
`libhwloc-apt.log`; no kernel/backend was rebuilt.

## Ordered gates

Q derives the candidate envelope. The resulting contract is frozen before H
is observed. H cannot tune epsilon or C. Free generation, quality, ADR, P1,
P2, and P3 are forbidden until their preceding gate passes.

Evidence states used in this round are `CODE_INSPECTED`, `ANALYTIC_DERIVED`,
`A3_VERIFIED`, and `HELDOUT_VERIFIED`.
