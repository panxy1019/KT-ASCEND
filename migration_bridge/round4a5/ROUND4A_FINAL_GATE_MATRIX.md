# Round 4A Final Gate Matrix

Evidence states are profile-specific. `N/A` means that profile did not define
an independent gate; it is not converted into a synthetic PASS.

| Gate | P1 | P2 | P3 |
|---|---|---|---|
| System semantics | PASS | PASS | PASS |
| Same-path determinism | PASS | PASS | PASS |
| Clean restart | PASS | PASS | PASS |
| Frozen numerical contract | PASS | PASS | PASS |
| Selected route coverage | PASS | PASS | PASS |
| CPU-not-hit exact control | PASS | PASS | PASS |
| Free-generation contract | PASS | N/A by P2 closure profile | PASS |
| Downstream quality | PASS | PASS | PASS |
| Stability/lifecycle | PASS (1,000 forwards) | PASS (5,760 generated tokens and transfer loops) | PASS (2,304 generated tokens) |
| Retained regression | PASS | PASS | PASS |

## Profile summary

| Profile | Placement | Numerical result | Coverage | Final state |
|---|---|---|---|---|
| P1 | Layer 17, 4 CPU experts | held-out pairwise contract established | 1/1 layer, 4/4 experts | `A3_VERIFIED_READY` |
| P2 | Layers 1/9/17/26, 16 CPU experts | Q2/H2 subsets, zero overflow | 4/4 layers, 16/16 experts | `A3_VERIFIED_READY` |
| P3 | Layers 1/5/8/12/17/19/22/26, 32 CPU experts | Q2/H2 subsets, zero overflow | 8/8 layers, 32/32 experts | `A3_VERIFIED_READY` |

## Final promotion rule

Every required P3 hard gate passed without changing placement, selected
experts, `B_pair`, K, quality corpus, worker count, or production-overlap
execution. Therefore:

`DEEPSEEK_V2_LITE_TP1_MULTI_PLACEMENT = A3_VERIFIED_READY`.
