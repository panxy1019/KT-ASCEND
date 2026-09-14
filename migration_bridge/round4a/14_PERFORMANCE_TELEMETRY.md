# Performance Telemetry

Only correctness-run wall times are recorded; no performance claim is made.
P0/P1 use the frozen correctness path with exact ACL stream synchronization,
synchronous D2H/H2D and `threadpool_count=1`. P2/P3 telemetry was not collected
after the P1 stop gate.

Corpus V correctness matrix averages:

| Profile | 1 token | 8 tokens | 16 tokens | 32 tokens | 64 tokens | 45-request wall sum |
|---|---:|---:|---:|---:|---:|---:|
| all-NPU | 0.1031 s | 0.4248 s | 0.7988 s | 1.5632 s | 3.0903 s | 53.8209 s |
| P1 | 0.2287 s | 0.8925 s | 1.6468 s | 3.1711 s | 6.2795 s | 109.9675 s |

```text
No performance claim: YES
```
