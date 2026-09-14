# Performance Telemetry

This round makes no performance claim.

| Measurement | LLAMAFILE | OpenBLAS candidate |
|---|---:|---:|
| 33-row median expert reconstruction | 12.01 ms | 6.91 ms (1T isolated) |
| P1 total request time | 109.38 s | 106.12 s |
| median P1 request | 1.641 s | 1.604 s |
| median 64-token request | 6.071 s | 5.945 s |

ATLAS median expert time was 14.59 ms; ACL including probe layout/allocation was
32.73 ms. None exceeded the 5x catastrophic threshold. Timing includes Python
harness overhead for isolated candidates and must not be interpreted as a
production throughput benchmark.
