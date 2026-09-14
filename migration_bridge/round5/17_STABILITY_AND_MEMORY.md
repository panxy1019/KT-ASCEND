# Round 5 — Stability and Memory

Evidence state: `A3_VERIFIED_READY`

## Transfer and CPU-plane lifecycle

- Round 2A: 1,000 LLAMAFILE forwards, one deterministic output, bounded RSS.
- Round 2B: BF16/FP32/int64 framework D2H/H2D exactness, 1,000 repetitions;
  real CPUInfer pipeline completed 1,000 cycles.
- P1: 1,000 production-overlap routed forwards, one output hash.
- P2: 117-request determinism matrix, 512 matched-history positions and 1,024
  recorded full-model forwards without ordering failure.

These runs exercise the same upstream framework-managed `Tensor.copy_`,
pinned-buffer, stream/event and CPUInfer drain semantics used by P3.

## P3 mandatory campaigns

```text
6 prompts x 64 tokens x 3 cycles x 2 campaigns = 2,304 tokens
```

| Campaign | Deterministic | RSS KiB start/peak/end | HBM MiB start/peak/end | Result |
|---|---:|---|---|---:|
| A | YES | 24,680,392 / 24,680,400 / 24,680,376 | 30,438 / 30,460 / 30,460 | PASS |
| B | YES | 24,700,260 / 24,700,260 / 24,700,256 | 30,438 / 30,460 / 30,460 | PASS |

Campaign output IDs were exact across A and B.  RSS finished below its initial
sample in both campaigns.  HBM increased by 22 MiB during initialization and
then remained at 30,460 MiB; this is an allocator/runtime plateau, not a
monotonic per-request increase.

The first preliminary Campaign A monitor parsed the wrong pipe-delimited
`npu-smi` field and reported zero.  It was not used for the formal gate.  The
monitor was corrected, then both formal campaigns were run from the beginning.

## Failure checks

| Check | Result |
|---|---:|
| crash | none |
| deadlock/timeout | none |
| traceback | none |
| NaN/Inf | none |
| stale output / changing same-path hash | none |
| monotonic unbounded RSS growth | none observed |
| monotonic unbounded HBM growth | none observed |

The evidence supports bounded stability for the stated campaign; it does not
claim that all possible long-duration workloads are leak-free.

```text
R5_ALIGNED_LIFECYCLE_STABILITY = A3_VERIFIED_READY
```
