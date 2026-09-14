# P1 Requalification

## Final minimal-fix matrix

| Gate | Required | Observed | Status |
|---|---:|---:|---|
| requests complete | 45 | 45 | PASS |
| all finite | yes | yes | PASS |
| prefix deterministic | yes | yes | PASS |
| exact token requests | 45 | 42 | **FAIL** |
| max abs logprob delta | <=0.20 | 2.4375150725245476 after divergent history | **FAIL** |

The three mismatches are `v_struct_03` at lengths 16, 32 and 64. All share one
first divergence at token index 9; shorter lengths 1 and 8 are exact.

```text
P1_NUMERICAL_CLOSURE = BLOCKED
P2_RESUME_AUTHORIZED = NO
P3_RESUME_AUTHORIZED = NO
```

The gate is not relaxed. See `ADR_R4A_NUMERICAL_ACCEPTANCE.md` for the residual
backend decision.
