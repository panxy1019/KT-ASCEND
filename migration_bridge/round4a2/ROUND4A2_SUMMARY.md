# B Round 4A.2 Summary

## Repository

Parent Round4A1 final / Round4A2 base: `ba508e4f920e99cd8cf1c0127d9aa6e5e0ac2559`
Round4A2 final: branch HEAD after documentation commit
Branch: `feature/kt-round4a2-cpu-gemm-numerical-closure`

SGLang base/final: `8a4bb4c850860377d9880b8c0fa570ca2d0d19e1`
Branch: `feature/kt-ep-round4a2-cpu-gemm-numerical-closure`
SGLang production diff: none

## A3

Container: `kt-r4a2-gemm` (disposable)
CPU: Kunpeng/aarch64, container CPU set `0-15`
NPU: Ascend NPU0 only
CANN image: 9.0.0
torch: 2.9.0
torch_npu: 2.9.0.post2
transformers-kt: 5.6.0.post2 frozen package
host modified: **NO**

## Frozen P1

Layer: 17
CPU experts: `{6,8,25,36}`
placement SHA256: `f6d4e9c6a2e5e8060e846dbc7c628d069c9aa6150aaa1e2690ea28a51ba286a3`
GGUF SHA256: `a16a50827ec81b54195bf246c7f9d05f7c1d5f3601ee33426c732f65892e180f`

## Current Backend Reproduction

Backend: ARM LLAMAFILE SGEMM
P1 exact: 42/45
Failure: one `v_struct_03` trajectory repeated at 16/32/64
First divergence: token index 9, 8828 vs 1273
Matched-history margin `(8828-1273)`: All-NPU `+0.125`, LLAMAFILE `0`
Prefix/finite: pass
Status: reproduced

## Backend Inventory and Provenance

| Backend | Version | Library SHA256 | Result |
|---|---|---|---|
| LLAMAFILE | frozen source | source base SHA | 42/45 baseline |
| OpenBLAS | 0.3.20+ds-1 | `3ea6a3c2...c8cdde6` | isolated winner, P1 failed |
| BLIS | 0.8.1-2 | `05034f0e...7c3f54` | numerical gate fail; 16T exit139 |
| ATLAS | 3.10.3 | `469a315b...87a59af` | numerical gate fail |
| ACL NEGEMM | 20.08 | `b6fb6fe3...75d3665` | numerical gate fail |
| current-tree KML | missing source path | n/a | not buildable |

Compiler: GCC/G++ 11.4.0. Exact paths, hashes and runtime settings are under
`backend_manifests/`.

## GEMM / Expert Numerical Matrix

33 real matched-history samples, E6=15/E8=4/E25=9/E36=5, ten repeats.

| Backend | Gate max rel-L2 | Up max rel-L2 | Down median | Down max | Deterministic |
|---|---:|---:|---:|---:|---:|
| LLAMAFILE | baseline | baseline | 0 | 5.048414e-4 | pass |
| OpenBLAS | 4.976403e-5 | 1.504919e-6 | 0 | 5.048410e-4 | pass |
| ATLAS | 4.976403e-5 | 1.780107e-4 | 7.311742e-8 | 1.381836e-3 | pass |
| ACL | 4.976403e-5 | 1.780107e-4 | 7.311742e-8 | 1.381836e-3 | pass |

BLIS 1T on 12 representative samples had down max `4.071028e-4`, worse than
the paired LLAMAFILE maximum `1.875118e-4`.

## First-Divergence Margins

| Case | All-NPU | LLAMAFILE | OpenBLAS |
|---|---:|---:|---:|
| v_struct_03 `(8828-1273)`, token 9 | +0.125 | 0 | +0.125 |
| v_en_01 `(30-279)`, token 10 | 0 | 0 | -0.125 |

OpenBLAS did not provide a corpus-wide improvement; it exchanged one near-tie
failure for another.

## Candidate Decision and Integration

Selected for experiment: OpenBLAS, because it alone passed the original margin
gate. The minimal selector altered only LLAMAFILE F32 SGEMM dispatch and was
fail-fast for unsupported contracts. It passed isolated/integrated equality,
C0–C4, sequential/overlap equality, a 1000-forward single hash, and the
registered SGLang Ascend KT EP routing test.

Full P1 failed, so the adapter was rejected and removed. Final production diff:
**zero**. Probe code and evidence remain; SGLang is unchanged.

## P1 Controlled

C0: pass
C1: pass
C2: pass
C3: pass
C4: pass
sequential vs overlap: exact
1000-forward: one hash
shared/scaling ownership: exact

## P1 Full Model

Requests: 45
Exact: **42/45 (FAIL)**
Mismatches: `v_en_01` at 16/32/64
Prefix: pass
Finite: pass
Post-divergence max absolute selected-token logprob delta: 2.0071879169
Matched-history failure margin: `-0.125` vs All-NPU `0`
Placement/CPU coverage contract: unchanged
Status: `BLOCKED`

## P2 / P3 / Stability

P2: `NOT_RUN_P1_BLOCKED`
P3: `NOT_RUN_P1_BLOCKED`
P3 campaign A/B: not run
Production memory claim: none

## Performance

LLAMAFILE P1 total: 109.38 s
OpenBLAS P1 total: 106.12 s
No performance claim: **YES**

## Regression

Round 4A P1 local: pass
SGLang KT EP registered routing test: pass
backend-specific captured input: pass
Round2A/Round2B/Round2C/Round3 campaigns: not run after ordered P1 stop
Final production code: unchanged

## Final

```text
ROUND4A2_CPU_BACKEND_NUMERICAL_INVESTIGATION = COMPLETE
ROUND4A2_CPU_BACKEND_NUMERICAL_CLOSURE = BLOCKED

BACKEND_OPTIONS_EXHAUSTED = TRUE
REQUIRES_ACCEPTANCE_DECISION = TRUE

MULTI_EXPERT_SINGLE_LAYER = BLOCKED
MULTI_LAYER_P2 = NOT_RUN
MULTI_LAYER_P3 = NOT_RUN

DEEPSEEK_V2_LITE_TP1_MULTI_PLACEMENT = BLOCKED
```
