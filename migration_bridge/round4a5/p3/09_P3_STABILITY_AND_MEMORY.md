# P3 Stability, Lifecycle, and Memory

Evidence state: `A3_VERIFIED_READY`

Two independent serving campaigns completed:

```text
6 prompts × 64 generated tokens × 3 cycles × 2 campaigns
= 2,304 generated tokens
```

Both campaigns produced one exact trajectory per prompt. The six Campaign B
trajectories also exactly matched Campaign A. No crash, deadlock, traceback,
NaN, or Inf was observed.

| Metric | Campaign A | Campaign B |
|---|---:|---:|
| RSS start | 99,439,976,448 | 99,414,159,360 |
| RSS peak | 99,548,364,800 | 99,554,304,000 |
| RSS end | 99,457,380,352 | 99,440,816,128 |
| RSS final-window range | 262,144 | 262,144 |
| HBM start | 33,850 MiB | 33,445 MiB |
| HBM peak | 33,851 MiB | 33,467 MiB |
| HBM end | 33,850 MiB | 33,467 MiB |
| HBM final-window range | 1 MiB | 2 MiB |

Campaign A returned to its HBM starting point. Campaign B acquired an early
22 MiB allocator plateau, then oscillated within 2 MiB rather than increasing
per request. RSS in both final windows varied by only 256 KiB. This evidence
supports `no monotonic unbounded growth`; it does not claim a zero-byte
allocator cache delta.

Evidence: `evidence/g8-campaign-{a,b}.json` and corresponding memory CSVs.

`P3_STABILITY = A3_VERIFIED_READY`.
