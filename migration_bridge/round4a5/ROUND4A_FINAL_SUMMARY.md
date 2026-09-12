# Round 4A Final Summary

## Outcome

Round 4A is closed for DeepSeek-V2-Lite TP1 static multi-placement on one
Ascend NPU with LLAMAFILE/CPUInfer experts. P1 established the heterogeneous
numerical contract, P2 closed the framework transfer-lifetime defect, and P3
proved that the same semantics scale from four to eight CPU-enabled MoE layers.

```text
ROUND4A_STATUS                                = PASS
PAIRWISE_NUMERICAL_ACCEPTANCE                 = HELDOUT_VERIFIED
P1_REQUALIFIED                                = A3_VERIFIED_READY
MULTI_LAYER_P2                                = A3_VERIFIED_READY
MULTI_LAYER_P3                                = A3_VERIFIED_READY
DEEPSEEK_V2_LITE_TP1_MULTI_PLACEMENT          = A3_VERIFIED_READY
```

## Frozen lineage

| Item | Value |
|---|---|
| P2 immutable tag | `round4a5-p2-a3-verified` |
| P2 tag target | `d4830525c6806d5d587b3096634614aad7fef181` |
| P3 branch | `feature/kt-round4a5-p3` |
| SGLang child | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |
| Round 4A final tag | `round4a-tp1-multiplacement-a3-verified` |

The final tag resolves the exact evidence/documentation commit. No upstream
alignment, rebase, or official Ascend implementation merge occurred in this
lineage.

## Frozen contracts and artifacts

| Item | Value |
|---|---|
| model revision | `DeepSeek-V2-Lite-604d5664` |
| tokenizer JSON SHA | `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669` |
| pairwise contract | `B_pair=2.1875`, K=32 |
| pairwise contract canonical SHA | `223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717` |
| quality manifest SHA | `c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce` |
| P3 placement SHA | `77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d` |
| P3 GGUF SHA | `992d32e0de59e135464f77fc4869e810f3d71b3ec01f6ed0996cbeaa58e49df4` |

## P3 closure

| Gate | Result |
|---|---|
| production-overlap same-process determinism | 117/117 requests; 9/9 prompt trajectories and prefixes exact |
| clean restart | 9/9 exact |
| pairwise numerical | Q2/H2 512 positions; zero overflow; stable exact and ambiguity membership 100% |
| route coverage | 8/8 layers, 32/32 CPU experts, 161,925 CPU routes |
| CPU-not-hit | 32/32 exact controls |
| free generation | 9 divergences; all ambiguous and in-set; zero stable divergence |
| downstream quality | C-Eval 27/128 and GSM8K 31/128, identical to All-NPU/P2 |
| stability | 2,304 generated tokens over two independent campaigns; bounded plateaus |
| retained regression | R2A/R2B/R2C/R3/SGLang/P1/P2 all PASS |

## What was proven

The frozen routing, mapping, ownership, shared-expert, scaling, and
framework-managed transfer semantics remain correct and deterministic when the
static placement expands from one CPU-enabled layer (P1), to four (P2), to
eight (P3). Cross-backend deviations remain inside the held-out-verified
pairwise envelope and do not produce a statistically meaningful quality
regression on the frozen 256-example quality set.

## Scope boundary

This status does not qualify TP2/HCCL, multiple NPUs, graph execution,
deferred/dynamic expert placement, MTP/speculative decoding, W8A8/MXFP4/KML,
multi-node serving, concurrency scaling, or production performance. These are
new risk surfaces and require new rounds.

The next permitted activity is a separate `Round 5 — Upstream Ascend
Alignment`, starting from the immutable Round 4A tag.
