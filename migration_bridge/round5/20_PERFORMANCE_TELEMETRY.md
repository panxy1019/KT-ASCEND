# Round 5 — Performance Telemetry

Performance was observed but was not an acceptance gate and no optimization
claim is made.

| Profile | CPU layers | Selected CPU experts | Observed decode telemetry |
|---|---:|---:|---|
| Round3 anchor | 1 | 1 | 16-token request completed in 2.86 s (request-level timing) |
| Aligned P1 | 1 | 4 | correctness workload completed; no normalized tok/s artifact retained |
| Aligned P2 | 4 | 16 | correctness workload completed; no normalized tok/s artifact retained |
| Aligned P3 | 8 | 32 | approximately 2.9-3.6 token/s during the determinism campaign |

The ranges are not directly comparable because the short Round3 request and
the P3 campaign use different request matrices.  Prefill latency and CPU/NPU
utilization were not captured with a sufficiently controlled protocol for a
performance conclusion.

P3 exercised all 32 CPU experts and accumulated 161,913 CPU route hits; its
throughput therefore does not come from reducing CPU coverage.  No raw ACL
copy, worker-topology redesign, NUMA tuning, KML, quantization, or sequential
control was introduced to improve the number.

```text
NO_PERFORMANCE_CLAIM = YES
PERFORMANCE_REVIEW_REQUIRED = YES (future dedicated round)
```
