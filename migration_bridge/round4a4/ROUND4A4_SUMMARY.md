# B Round 4A.4 Summary

## Repository

Parent Round4A3 final: `7b9de4a8249dacfd442f9fb465e2dac6c611f986`  
Round4A4 base: `7b9de4a8249dacfd442f9fb465e2dac6c611f986`  
Branch: `feature/kt-round4a4-pairwise-margin-qualification`

SGLang base: `8a4bb4c850860377d9880b8c0fa570ca2d0d19e1`  
SGLang instrumentation: `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2`

## Scope

New GEMM backend search: NO  
Production arithmetic changes: NO  
Debug/test instrumentation: YES, default OFF  
TP2 before approval: NO  
Graph/Deferred/Dynamic/MTP/Speculative: OFF

## Frozen P1

Layer: 17  
CPU experts: `{6,8,25,36}`  
Physical NPU experts: 60  
Placement SHA: `9548bf6e06014e034c6a6650af3a891a546f70d8c8d79b059174ba571c44471f`

## Corpora

Q2: 18 prompts, 1,152 positions, SHA
`551fc4fdf75053eda18d511ab2b479907576bd31989460281ef2424689dd3e07`  
H2: 16 prompts, 1,024 positions, SHA
`b1874527eca8c01ab3df16215679eb7365a938b2c89dcb910334ad70bba7b787`  
F: 6 prompts  
Quality manifest SHA:
`c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce`

## Exact and expert gates

System routing/mapping/ownership/shared/scaling/buffer/stream: PASS  
P1 same-process 10-repeat: PASS  
P1 clean restart and prefixes: PASS  
CPU-not-hit control: 32/32 exact  
Expert rel-L2 max: `0.0035398305 <= 0.01`

## Pairwise Q2 and B_pair

Q2 pairs: 36,104  
Distortion p50/p90/p95/p99/p99.5/p99.9/max:
`0 / 0.125 / 0.125 / 0.25 / 0.25 / 0.75 / 1.75`  
Selected source: Q2 maximum with predeclared `1.25x` reserve  
`B_pair = 2.1875`  
Candidate K: 32  
Candidate contract SHA:
`accbbc15baf49d869b062dc2b71b4fe6f71a7fa8bd74f27120757d1c1f0e6627`

Q2 stable: 639, exact 639/639  
Q2 ambiguous: 513, membership 513/513  
Ambiguity cardinality: diagnostic only

## H2 and free generation

H2 stable: 654, exact 654/654  
H2 ambiguous: 370, membership 370/370  
H2 pairwise overflow: 0  
H2 maximum distortion: 0.6875  
All finite: PASS

H2 free requests/divergences: 16/3  
F free requests/divergences: 6/2  
Stable divergences: 0  
All five divergent Hybrid tokens were in the frozen ambiguity set.

## Quality and ADR

C-Eval: All-NPU 57/128, Hybrid P1 57/128, delta 0  
GSM8K MC: All-NPU 45/128, Hybrid P1 47/128, delta +2/128  
Invalid outputs: 0  
Statistical result: `NO_STATISTICALLY_MEANINGFUL_REGRESSION`

ADR: Option A, `APPROVED`  
Final contract version: `round4a4-v1`  
Canonical contract SHA:
`223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717`

## P1 Requalification

C0-C4: PASS  
Sequential equals overlap: PASS  
1,000-forward: one unique hash  
System/numerical/quality: PASS  
Status: `A3_VERIFIED_READY`

## P2

Placement: layers `{1,9,17,26}`, 16 CPU experts  
Numerical subsets: PASS, no pairwise overflow  
Coverage: 4/4 layers, 16/16 experts, 74,784 CPU hits  
Same-path 10-repeat: **FAIL**  
Sequential control: **FAIL**  
Failure: `SAME_PATH_NONDETERMINISM`  
Status: `BLOCKED`

## P3, stability, and final quality

P3: `NOT_RUN_P2_BLOCKED`  
P3 stability campaign: NOT RUN  
P3 final quality: NOT RUN

## Regression

Current SGLang KT EP tests: 5 passed  
Round 2A retained suite: 17 passed, 4 stale FP32/BF16 dtype-assert failures  
Round 2B retained suite: 22 passed, 1 skipped, 1 unresolved pipeline failure  
Round 2C retained suite: 44 passed

## Final

`PAIRWISE_NUMERICAL_ACCEPTANCE = HELDOUT_VERIFIED`  
`P1_REQUALIFIED = A3_VERIFIED_READY`  
`MULTI_EXPERT_SINGLE_LAYER = A3_VERIFIED_READY`  
`MULTI_LAYER_P2 = BLOCKED`  
`MULTI_LAYER_P3 = NOT_RUN`  
`DEEPSEEK_V2_LITE_TP1_MULTI_PLACEMENT = BLOCKED`

Round 4A.4 established and held-out-verified the pairwise numerical contract and
requalified P1. It did not complete multi-layer placement: the exact P2
same-path determinism gate failed, so execution stopped before P2 quality and
all P3 work as required.
