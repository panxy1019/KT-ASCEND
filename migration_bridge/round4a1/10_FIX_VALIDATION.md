# Fix Validation

## Targeted progression

| Check | Before | After F1 | After F1+F2 |
|---|---|---|---|
| `v_en_01`, first 2 tokens | `[185,549]` | `[185,185]` | `[185,185]` |
| `v_en_01`, 64 tokens | divergent | not final gate | exact all-NPU sequence |
| `v_en_02`, first 2 tokens | `[185,51249]` after F1 | `[185,51249]` | `[185,549]` |
| A3 CPU+NPU routed fixture | pass | pass | pass |
| P1 45 requests | 30 exact | 26 exact in an intermediate run | 42 exact |

The temporary 26/45 result after F1 alone is expected: correcting expert
boundaries exposed the independent premature-partial-rounding error. F2 fixed
the known `v_en_02` tie and raised the final count to 42/45.

## A3 regression

The registered Ascend CPU+NPU routing test passed after the final minimal diff:

```text
1 passed, 7 warnings in 16.80s
```

Warnings are pre-existing pytest/Numpy/unsupported-quantization notices. No
crash, deadlock or non-finite output occurred.

## Evidence

- `evidence/round4a1-p1-fixed2-v-en01-64.json`
- `evidence/round4a1-p1-fixed2-v-en02.json`
- `evidence/round4a1-p1-fixed2-45.json`
- `evidence/round4a1-p1-fixed2-compare.json`
