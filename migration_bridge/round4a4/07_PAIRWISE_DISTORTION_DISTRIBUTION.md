# Pairwise Distortion Distribution

Evidence state: `ANALYTIC_DERIVED` from `A3_VERIFIED` logits.

Pairwise absolute distortion is

`|(zH_a - zH_i) - (zN_a - zN_i)|`,

where `a` is the All-NPU top-1 and `i` belongs to the top-32 union universe.

| Metric | p50 | p90 | p95 | p99 | p99.5 | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q2 pairwise abs distortion | 0 | 0.125 | 0.125 | 0.25 | 0.25 | 0.75 | 1.75 |
| Q2 per-position maximum | 0.125 | 0.1875 | 0.25 | 0.375 | 0.522969 | 1.308938 | 1.75 |
| H2 pairwise abs distortion | 0 | 0.125 | 0.125 | 0.25 | 0.25 | 0.375 | 0.6875 |
| F pairwise abs distortion | 0 | 0.125 | 0.125 | 0.1875 | 0.25 | 0.436969 | 0.75 |

The full JSON also stratifies Q2 by CPU hit count, CPU expert, prompt family,
and token-position bucket. The observed maximum is not claimed to prove a
specific reduction tree; it is a measured bound for this frozen model,
placement, backend, and runtime.

