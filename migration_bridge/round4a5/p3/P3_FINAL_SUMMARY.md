# P3 Final Summary

## Final decision

All P3 gates passed on A3 under the unchanged production-overlap runtime,
pairwise contract, placement, GGUF, and quality corpus.

```text
P3_SAME_PROCESS_DETERMINISM = A3_VERIFIED_READY
P3_CLEAN_RESTART            = A3_VERIFIED_READY
P3_PAIRWISE_NUMERICAL       = A3_VERIFIED_READY
P3_CPU_COVERAGE             = A3_VERIFIED_READY
P3_CPU_NOT_HIT_CONTROL      = A3_VERIFIED_READY
P3_FREE_GENERATION          = A3_VERIFIED_READY
P3_DOWNSTREAM_QUALITY       = A3_VERIFIED_READY
P3_STABILITY                = A3_VERIFIED_READY
P3_RETAINED_REGRESSION      = PASS
MULTI_LAYER_P3              = A3_VERIFIED_READY
```

## Repository and model

| Item | Value |
|---|---|
| P2 immutable tag / target | `round4a5-p2-a3-verified` / `d4830525c6806d5d587b3096634614aad7fef181` |
| P3 branch | `feature/kt-round4a5-p3` |
| P3 execution documentation base | `25821cd22d6a1c63f43467a9f0e07bf5958182b3` |
| P3 final | immutable final tag target |
| SGLang child | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |
| model / revision | DeepSeek-V2-Lite / `604d5664` |
| placement layers | `{1,5,8,12,17,19,22,26}` |
| selected CPU experts | 32 |
| placement SHA | `77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d` |
| GGUF SHA / size | `992d32e0de59e135464f77fc4869e810f3d71b3ec01f6ed0996cbeaa58e49df4` / 17,716,742,368 |
| pairwise contract SHA | `223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717` |
| quality manifest SHA | `c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce` |

## Gate matrix

| Gate | Result | Principal evidence |
|---|---|---|
| Same-process determinism | PASS | 117 requests; 9/9 repeated trajectories and prefixes exact |
| Clean restart | PASS | 9/9 trajectories exact across processes |
| Pairwise numerical | PASS | Q2/H2 512 positions, stable exact, ambiguity membership, zero overflow |
| Coverage | PASS | 8/8 layers, 32/32 experts, 161,925 CPU routes |
| CPU-not-hit | PASS | 32/32 real positions exact-zero CPU partial and exact merge |
| Free generation | PASS | 22 requests, 9 ambiguous divergences, zero stable divergences, 9/9 in set |
| Quality | PASS | C-Eval 27/128; GSM8K 31/128; deltas and 95% CIs zero |
| Stability | PASS | 2,304 tokens, two campaigns, deterministic, bounded memory plateaus |
| Retained regression | PASS | R2A/R2B/R2C/R3/SGLang/P1/P2 all closed |

## A3

Environment `a3-r4a5-p3-0825d8eb90af`: Kunpeng 920, cpuset 0-15, NPU0
only, CANN 9.0.0, Python 3.11.15, torch 2.9.0+cpu, torch_npu
2.9.0.post2, Transformers 5.3.0.dev0. Host and business containers were not
modified.

## Scope boundary

This closes static TP1 multi-placement correctness for one Ascend NPU and
LLAMAFILE CPU experts. It does not qualify TP2/HCCL, graph mode, deferred or
dynamic placement, MTP/speculation, quantization, KML, multi-node operation,
or production performance.

`MULTI_LAYER_P3 = A3_VERIFIED_READY`.
