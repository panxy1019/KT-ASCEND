# Q Numerical Error Distribution

Evidence state: **ANALYTIC_DERIVED** from the accepted A3 sweep.

| Metric | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| full-logit max abs | 0.25 | 0.50 | 0.50 | 0.75 | 6.9375 |
| mean abs | 0.024826 | 0.062288 | 0.079393 | 0.115251 | 0.797817 |
| relative L2 | 0.005821 | 0.014110 | 0.017505 | 0.026017 | 0.115144 |
| baseline top-16 max error | 0.125 | 0.125 | 0.25 | 0.390625 | 0.75 |
| absolute margin distortion | 0 | 0.125 | 0.125 | 0.25 | 0.25 |

The contract epsilon is derived from the top-16 candidate-set p99, not from a
known failure token. Full data and per-row top-16 evidence are in
`evidence/q-final-metrics-v2.json`.

## CPU-hit control limitation

Counting the entire matched prefix plus sampling pass leaves zero aggregate
CPU-not-hit positions: every Q prefix exercises at least one of the four CPU
experts. The strict CPU-hit=0 control is therefore **UNAVAILABLE**, not passed.
Sampling-token-only hit counts are retained for diagnostics but are not a valid
exact control because earlier Hybrid KV states can carry CPU-path differences.

The previously verified Round 4A CPU-not-hit local control remains historical
evidence; this sweep does not relabel it as current held-out evidence.
