# Round16 current execution

Updated: 2026-09-29 11:31 UTC.

Terminal state: `ROUND16_PARTIAL_WITH_GAPS`.

The default remains S0: SGLang + KT, Round14a promoted GDN, QSA Vectorized
Prefill / DIRECT_BF16_ROW Decode, PLE ASYNC, C128/MXFP4, Uniform Q32
BF16/NZ29, chunk32, H129, TP1 and serial single-request execution.

## A — current S0 critical path

Two fresh S0 profile processes completed. Both contain complete serialized W1
and W2 measurement batches and direct NPU event samples. The diagnostic
medians were:

| Block | QSA indexer | GDN | resident expert |
|---|---:|---:|---:|
| A | 2.868 ms | 1.057 ms | 0.416 ms |
| B | 2.986 ms | 1.088 ms | 0.434 ms |

These are direct provider spans, not additive critical-path percentages. The
native CPU worker ABI exposes no worker start/end callback. The trace therefore
does not contain a direct isolated `HOST_SYNC_EXPOSED_WAIT` interval or a
direct `HOST_LAUNCH_METADATA_CRITICAL` interval. Parent/child subtraction and
cross-clock addition remain prohibited.

The A disposition is `A1_INSUFFICIENT_ATTRIBUTION`. A2 and A3 are both
`NOT_SELECTED_EVIDENCE_GAP`; no A component or request candidate was created.
This is not a measured no-gain result. A16 is identical to S0.

## B — vLLM-Ascend + KT Expert

B completed the whole staged integration chain on the exact image:

1. `REAL_IMAGE_Q32_CONSTRUCTOR_VERIFIED`: one real layer allocated 512 logical
   experts as 32 physical NPU slots, with a 300 MiB nominal payload and no
   512-expert device payload.
2. `REAL_Q32_LOADER_AND_LAYOUT_VERIFIED`: resident rows were selected before
   NPU transfer, all gate/up/down projections were covered, and Ascend
   post-load layout processing stayed within Q32 storage.
3. `REAL_HYBRID_MOE_VERIFIED`: all-NPU, all-CPU and mixed Top-10 routes passed
   an independent mixed-precision reference, repeat hashes, 20 lifecycle
   rounds and cancellation drain.
4. `ALL_LAYER_RESIDENCY_VERIFIED`: 48 distinct owners and 96 storages stayed
   resident together; allocator peak was 15,099,666,432 bytes and cleanup
   returned to the component baseline.
5. `PASS_ROUND16_B_V5_FULL_MODEL_SMOKE`: five exact canaries, a 32-token
   Decode, cancel with no unfinished request, and exact same-process reuse all
   passed with finite logprobs.

The final B engine was TP1, max-running=1, max-model-len=256 and eager
(`CUDAGraphMode.NONE`). vLLM retained request, KV, QSA/GDN/PLE state, batch,
cancel and token ownership. KT owned the physical Q32 expert storage, the C128
MXFP4 work and its short-lived completion/buffer lease. Graph replay was not
retained; this result does not inherit a FULL_DECODE_ONLY label.

## KV qualification boundary

The single-page `unsqueeze(0)` path is recorded only as local layout
compatibility. It is not the KV correctness proof.

The real NPU component probe used cache shape `[3, 3072, 2, 256]`, non-identity
block table `[[2, 0, 1]]`, twelve positions crossing all three pages, two KV
heads and 24 query heads. K/V writes were exact. Sparse GQA attention matched
an independent reference with relative L2 `0.00184652` and max absolute error
`0.015625`.

The full model is qualified only for the frozen Round16 range: input and output
each at most 128 tokens, cumulative context at most 256. A long-context full
model request crossing multiple KV pages was not run and broad KV correctness
is not claimed.

## Request comparison

Four fresh independent processes ran in order `S0-A -> B-A -> B-B -> S0-B`.
Each process used profiler OFF and ten generations: W1, W2 and W4 each had one
warmup and two measurements, followed by one exact arithmetic canary. All
performance requests generated their complete output budget and finite
logprobs.

For W2, B/S0 was:

| Pair | E2E ratio | TPOT ratio | serial output throughput ratio |
|---|---:|---:|---:|
| S0-A / B-A | 1.20836 | 1.22502 | 0.82757 |
| S0-B / B-B | 1.17604 | 1.19442 | 0.85031 |

Both the Decode gate and W1/W4 protection gate failed. B reached
`B16_MEASURED_NO_PROMOTION`; this is a measured system result, not a pure
framework causal comparison, because the carriers and graph modes differ.

## Resources and retirement

Each formal request process passed a fresh two-by-ten-second admission. All
four model processes exited and were removed. After evidence archival and
hashing, the registered 131 model plus 48 bank files had no active FD/mmap;
targeted `POSIX_FADV_DONTNEED` released 296.55 GB, 348.99 GB, 209.02 GB and
293.70 GB respectively. `fincore` then reported zero resident bytes for the
registered set. No global cache drop or external workload change occurred.

Round16 has no active controller, model or guard. Historical timers remain
paused. No push, merge, tag, reset or clean was performed.

## Terminal interpretation

B is fully integrated and measured but not selected. A did not reach a safe
candidate decision because direct critical-path evidence remains incomplete.
The round therefore ends as `ROUND16_PARTIAL_WITH_GAPS`, with S0 unchanged.
The exact remaining A gap is a reviewed direct measurement boundary for CPU
result exposed wait and Host launch/metadata; it must not be filled by adding
nested spans or by treating QSA's sampled event as the whole request bottleneck.
