# Minimal formal performance — no measurable speedup

Evidence state: A3_VERIFIED measurement, ANALYTIC_DERIVED paired inference.
Frozen single shape192/256, batch1; three ABBA blocks; 12 independent serving
processes, two warmups and one timed request per phase. No resampling or
post-result workload selection. Capture/route recorder OFF, cache OFF.

| Block | Paired OFF-time / ON-time |
|---|---:|
| 0 | 0.980560 |
| 1 | 1.025396 |
| 2 | 0.962814 |

Geometric mean **0.989243×**, Student-t log-ratio95% CI(df2)
**[0.912743,1.072153]**. Frozen lower-bound>1 criterion is not met.
`NO_MEASURABLE_SPEEDUP` — not proof that Deferred is always slower or faster.

| Client-observed metric | OFF | ON |
|---|---:|---:|
| Requests timed | 6 | 6 |
| Latency p50 seconds | 29.6062 | 29.7977 |
| Latency p95 seconds | 30.2297 | 30.9474 |
| TTFT p50 seconds | 0.3613 | 0.3546 |
| TTFT p95 seconds | 1.6534 | 1.5568 |
| Mean end-to-end tokens/s | 8.6806 | 8.5887 |

Same-workload preflight observed16 deferred routes. This sparse workload is
not evidence for arbitrary deferral density. Shared-host load varied; zero
other NPU0 HBM process was observed in phase samples, not proof of exclusive
compute or no transient contention. Three blocks provide limited precision.

Raw reports and requests: `evidence/r64-minimal-performance-v1/`.
Analysis: its `performance-summary.json`; resource review/description and
raw monitor JSONL in `evidence/`. Local analysis root `/tmp/r64-performance-complete`
is a byte-preserving copy of A3 `/workspace/results/r64-minimal-performance-v1`.
Source equality across all phases was enforced by the campaign runner.

Do not publish an acceleration READY tag. Quality screen PASS remains valid;
remaining descriptive integrity and final ADR must close separately.
