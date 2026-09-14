# B Round 4A.3 Summary

## Repository

- Parent Round4A2 final / Round4A3 base: `c2a456aec16846d353bd3075361d2cda6a3e085c`
- Round4A3 final: documentation commit containing this summary (full SHA in handoff)
- Branch: `feature/kt-round4a3-numerical-acceptance`
- SGLang base/final: `8a4bb4c850860377d9880b8c0fa570ca2d0d19e1`
- SGLang branch: `feature/kt-ep-round4a3-numerical-acceptance`
- production code change: **NO**

## Scope

- new GEMM backend search: NO
- TP2 before approval: NO
- Graph/Deferred/Dynamic placement: OFF/OFF/OFF
- backend: frozen LLAMAFILE / CPUInfer

## Frozen P1

- layer: 17
- CPU experts: `{6,8,25,36}`
- placement file SHA256: `c12d2954e0188ab9d7d1567085d342fe4b3ed5b769fdb976f743c2bbb50fc509`
- placement strategy: explicit `frequency`

## Corpora

- Q: 9 prompts, 576 positions, SHA `7617682ccd1d004f6221d35b9e484465ebb6c2a40359481f047f3fbb298b439a`
- H: 12 prompts, 768 positions, SHA `dc137b4c9531029f58eb2daa4746760840a301e6930ee6b98da12f5edc06c2cc`
- D: GSM8K-32 plus C-Eval-32, manifest SHA `7dfb4badd385e26ad43a9bb7ebd03ca20b75198f82b4b2ef4750c7245d9f0499`

## A3

- disposable container: `kt-r4a3-acceptance`
- CPU: Kunpeng/aarch64, CPU set `0-15`
- NPU: Ascend NPU0 only
- CANN image: 9.0.0
- torch/torch_npu: 2.9.0 / 2.9.0.post2
- host modified: **NO**

## System / Instrumentation

- routing/placement frozen and inspected: pass
- only Layer 17 `{6,8,25,36}` CPU-owned: pass
- sampling-pass response/logit consistency: 2688/2688 accepted captures
- all finite: pass
- strict aggregate CPU-not-hit Q control: unavailable (0 positions)
- same-path 10-repeat Round4A3 rerun: not run before held-out; historical
  Round4A.2 evidence only

Two preliminary Q captures were invalidated due concurrent dump interleaving
and wrong forward-pass selection. Exclusive locking and serving-token/logit
invariants closed both instrumentation defects before the final Q derivation.

## Q Expert / Logit Envelope

- expert relative-L2 gate: `<=1e-2` retained from verified Round 4A evidence
- full-logit max abs p50/p95/p99/max: `0.25 / 0.50 / 0.75 / 6.9375`
- mean abs p50/p95/p99/max: `0.024826 / 0.079393 / 0.115251 / 0.797817`
- relative L2 p50/p95/p99/max: `0.005821 / 0.017505 / 0.026017 / 0.115144`
- candidate error p50/p95/p99/max: `0.125 / 0.25 / 0.390625 / 0.75`
- margin distortion p50/p95/p99/max: `0 / 0.125 / 0.25 / 0.25`

## Q Candidate Margin Contract

- epsilon: `0.390625`
- C: `1`
- stable: 482; exact 482/482
- near tie: 94 (16.32%); tie-set pass 94/94
- max tie-set size: 6
- positions with size >5: 1/576
- candidate contract content SHA: `37be7e2b3548a335ea2287d6b06b67835924f14961d47646de6ce3595c832f56`

## Held-out H

- stable: 689; exact 689/689
- near tie: 79 (10.29%); membership pass 79/79
- candidate error max: 0.6875 (no overflow)
- full-logit max abs: 1.5 (no overflow)
- relative L2 max: 0.0387082 (no overflow)
- all finite: pass
- tie-set max: **16 (FAIL; frozen maximum 6)**
- failing locations: `h_math_01` indices 11, 36, 62; baseline margin 0
- held-out status: `HELDOUT_CONTRACT_FAILURE`

## Free Generation / Quality / ADR

- free generation: `NOT_RUN_HELDOUT_BLOCKED`
- GSM8K/C-Eval A/B: `NOT_RUN_HELDOUT_BLOCKED`
- ADR decision: **Option B — REJECT CONTRACT**
- H2: not proposed; no independent mechanism hypothesis

## P1 / P2 / P3

- P1 requalification: `NOT_RUN_HELDOUT_BLOCKED`
- P2: `NOT_RUN_HELDOUT_BLOCKED`
- P3: `NOT_RUN_HELDOUT_BLOCKED`
- stability campaigns: `NOT_RUN_HELDOUT_BLOCKED`

## Final

```text
NUMERICAL_ACCEPTANCE_CONTRACT = REJECTED
HETEROGENEOUS_NUMERICAL_ACCEPTANCE = REJECTED

P1_REQUALIFIED = BLOCKED
MULTI_EXPERT_SINGLE_LAYER = BLOCKED
MULTI_LAYER_P2 = NOT_RUN
MULTI_LAYER_P3 = NOT_RUN

DEEPSEEK_V2_LITE_TP1_MULTI_PLACEMENT = BLOCKED
```
