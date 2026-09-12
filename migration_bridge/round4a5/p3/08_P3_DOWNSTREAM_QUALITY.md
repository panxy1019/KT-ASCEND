# P3 Downstream Quality

Evidence state: `QUALITY_VERIFIED` on A3

The recovered frozen quality manifest was used without changing examples,
prompts, evaluator, tokenizer, or decoding protocol:

`c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce`.

| Benchmark | All-NPU | P2 | P3 | P3 delta vs NPU | P3 delta vs P2 |
|---|---:|---:|---:|---:|---:|
| C-Eval | 27/128 | 27/128 | 27/128 | 0 | 0 |
| GSM8K MC | 31/128 | 31/128 | 31/128 | 0 | 0 |

For both comparisons and both benchmarks, discordant correctness pairs were
zero, invalid outputs were zero, and the 20,000-sample paired bootstrap 95%
confidence interval was `[0,0]`.

Evidence: `evidence/g7-quality-p3.json`,
`evidence/g7-quality-vs-allnpu.json`, and
`evidence/g7-quality-vs-p2.json`.

`P3_DOWNSTREAM_QUALITY = A3_VERIFIED_READY` with
`NO_STATISTICALLY_MEANINGFUL_REGRESSION`.
