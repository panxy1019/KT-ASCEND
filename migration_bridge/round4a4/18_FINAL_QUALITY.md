# Final Quality

Status: `P1_QUALITY_VERIFIED_P2_P3_NOT_RUN`

The frozen 128-example C-Eval and 128-example GSM8K multiple-choice A/B passed
for All-NPU versus Hybrid P1:

| Benchmark | All-NPU | Hybrid P1 | Delta | Paired bootstrap 95% CI | Result |
|---|---:|---:|---:|---|---|
| C-Eval | 57/128 | 57/128 | 0 | `[0, 0]` | no regression |
| GSM8K MC | 45/128 | 47/128 | +2/128 | `[0, 5/128]` | no regression |

Both sides had zero invalid outputs. The formal protocol used frozen conditional
log-likelihood scoring over four choices because this base-style checkpoint did
not reliably emit parseable free-form benchmark answers. GSM8K correct answers
and deterministic distractors were frozen before the formal A/B.

P2 quality smoke was not run after the exact N1 gate failed. P3 was not entered,
so no All-NPU/P1/P3 scale-out quality claim is made.
