# Downstream Quality A/B

Evidence state: `QUALITY_VERIFIED`.

The final quality manifest was frozen before formal mode comparison. It contains
128 GSM8K and 128 C-Eval examples and has SHA256
`c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce`.

Because the served DeepSeek-V2-Lite checkpoint is a base-style completion
model, both benchmarks use deterministic conditional log-likelihood over four
frozen options. GSM8K distractors are mechanically derived from each reference
number and shuffled with a per-row frozen seed; C-Eval retains its source
options. This prevents parser behavior from masquerading as model quality and
produces zero invalid outputs in both modes.

| Benchmark | All-NPU | Hybrid P1 | Delta | Paired bootstrap 95% CI | Invalid NPU/Hybrid | Status |
|---|---:|---:|---:|---:|---:|---|
| C-Eval, n=128 | 57/128 (0.4453125) | 57/128 (0.4453125) | 0 | [0, 0] | 0/0 | PASS |
| GSM8K MC, n=128 | 45/128 (0.3515625) | 47/128 (0.3671875) | +0.015625 | [0, 0.0390625] | 0/0 | PASS |

The bootstrap used 20,000 paired resamples and seed 0. There were no C-Eval
discordant correctness outcomes. On GSM8K, two rows were All-NPU-wrong and
Hybrid-correct; no row was All-NPU-correct and Hybrid-wrong. The conclusion is
`NO_STATISTICALLY_MEANINGFUL_REGRESSION`, not a performance or capability
claim.

Summary evidence SHA256:
`58af72d287e3a5b9bffcc5dff884fa8e0601db1e70ae0c4a0118e0d16e4525ee`.

