# P3 Performance Telemetry

Evidence state: `A3_VERIFIED`, diagnostic only

No performance gate or optimization claim is made. Production-overlap serving
logs reported the following request-level decode telemetry:

| Workload | Samples | min | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| G1 determinism | 156 | 0.51 | 2.99 | 3.40 | 3.69 token/s |
| quality | 54 | 0.32 | 2.71 | 3.08 | 3.16 token/s |
| stability B | 28 | 0.54 | 2.83 | 3.09 | 3.41 token/s |

The low minima include first decode/warm-up observations. Steady serving was
approximately 2.9–3.6 token/s, consistent with the live observation during
G1. Prefill telemetry varies strongly with cached and short inputs, so it is
retained in server logs rather than reduced to a production claim.

`NO PERFORMANCE CLAIM = YES`. Any future raw-copy, NUMA, worker-topology, KML,
or quantization optimization belongs after this correctness freeze.
