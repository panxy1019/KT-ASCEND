# Round 5 — Aligned P3 Requalification

Evidence state: `A3_VERIFIED_READY`

## Frozen profile

```text
model: deepseek-ai/DeepSeek-V2-Lite
revision: 604d5664dddd88a0433dbae533b7fe9472482de0
TP: 1
device: NPU0 only
CPU layers: {1, 5, 8, 12, 17, 19, 22, 26}
CPU experts per selected layer: 4
selected CPU experts: 32
execution: production overlap
CPU backend: LLAMAFILE / CPUInfer, F32 GGUF
NPU backend: Ascend BF16/unquantized
Graph/Deferred/Dynamic/MTP/Speculative: OFF
```

Frozen identities:

| Item | SHA256 |
|---|---|
| canonical placement | `77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d` |
| P3 GGUF | `992d32e0de59e135464f77fc4869e810f3d71b3ec01f6ed0996cbeaa58e49df4` |
| pairwise contract | `223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717` |
| quality manifest | `c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce` |

No threshold, placement, expert, corpus, or production execution mode was
changed after observing P3 output.

## Same-process and clean-restart determinism

The production-overlap server completed the full matrix:

```text
9 prompts x 10 repeats x 64 tokens = 90 full requests
9 prompts x {8,16,32}             = 27 prefix requests
```

All nine prompts had exactly one 64-token trajectory across ten repetitions.
Every prefix request exactly matched its frozen 64-token prefix, and all 5,760
recorded logprobs were finite.

A separately started serving process reproduced all nine 64-token trajectories
exactly; all 576 restart logprobs were finite.

```text
P3_SAME_PROCESS_DETERMINISM = A3_VERIFIED_READY
P3_CLEAN_RESTART = A3_VERIFIED_READY
```

## Frozen pairwise numerical contract

The Round 4A bound remained `B_pair=2.1875`, with candidate `K=32`.

| Corpus | Positions | Pairs | p50 | p95 | p99 | Max | Stable exact | Ambiguous membership | Overflow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2 frozen subset | 256 | 8,117 | 0.0625 | 0.25 | 0.50 | 1.9375 | 137/137 | 119/119 | 0 |
| H2 frozen subset | 256 | 8,098 | 0.0625 | 0.1875 | 0.3125 | 0.5625 | 155/155 | 101/101 | 0 |

All outputs were finite.  There was no stable flip, ambiguity escape, or
pairwise envelope overflow.  Ambiguity cardinality was diagnostic only.

The observed Q2 maximum remained below the bound, while H2 did not show
placement-scale numerical growth relative to aligned P2.

```text
P3_PAIRWISE_NUMERICAL = A3_VERIFIED_READY
```

## Coverage and system controls

Coverage came from upstream `ExpertDistributionRecorder` logical counts and
the production KT GPU ownership masks over 1,024 recorded forward passes.

| Layer | Frozen CPU experts | Total CPU route hits |
|---:|---|---:|
| 1 | `{31,43,50,57}` | 17,879 |
| 5 | `{14,38,39,41}` | 18,370 |
| 8 | `{1,17,33,45}` | 22,689 |
| 12 | `{35,41,46,58}` | 22,108 |
| 17 | `{6,8,25,36}` | 20,997 |
| 19 | `{14,32,47,62}` | 19,851 |
| 22 | `{21,60,61,62}` | 24,101 |
| 26 | `{10,26,30,56}` | 15,918 |

Totals:

```text
selected layers exercised: 8/8
selected experts exercised: 32/32
total CPU route hits: 161,913
routing counts nonnegative: YES
ownership mask invariant: YES
```

All eight selected layers were also run through direct C0-C4 controls using
their real weights.  At every layer:

- KTEP output equalled the explicit CPU-partial plus NPU-partial reference;
- routing, shared expert and routed scaling were each applied exactly once;
- physical NPU expert count was 60;
- C0 CPU partial was exact zero;
- C0 Hybrid merge equalled the NPU partial exactly.

```text
P3_CPU_COVERAGE = A3_VERIFIED_READY
P3_CPU_NOT_HIT_CONTROL = A3_VERIFIED_READY
```

## Free generation

The frozen H2 set (16 requests) and frozen F set (6 requests) were generated
for 64 greedy tokens.  Every aligned P3 trajectory exactly reproduced the
immutable Round 4A P3 trajectory for the same request:

| Corpus | Requests | Exact versus Round 4A | New first divergences |
|---|---:|---:|---:|
| H2 | 16 | 16/16 | 0 |
| F | 6 | 6/6 | 0 |

Consequently, aligned P3 introduced neither a stable-region divergence nor a
new ambiguity-membership question.

```text
P3_FREE_GENERATION = A3_VERIFIED_READY
```

## Downstream quality

| Benchmark | Frozen all-NPU | Aligned P2 | Aligned P3 | P3 invalid | P3 vs all-NPU discordant | Bootstrap delta 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| C-Eval | 27/128 | 27/128 | 27/128 | 0 | 0 | [0, 0] |
| GSM8K MC | 31/128 | 31/128 | 31/128 | 0 | 0 | [0, 0] |

The P3 and P2 correctness vectors are also identical.  The paired bootstrap
used 20,000 samples with seed 0.

```text
P3_DOWNSTREAM_QUALITY = A3_VERIFIED_READY
```

## Stability and memory

Two independent campaigns each executed six prompts, 64 tokens and three
cycles.  All per-prompt trajectories were deterministic within and across
campaigns.

| Campaign | Tokens | RSS KiB start/peak/end | RSS delta | HBM MiB start/peak/end | HBM delta |
|---|---:|---|---:|---|---:|
| A | 1,152 | 24,680,392 / 24,680,400 / 24,680,376 | -16 KiB | 30,438 / 30,460 / 30,460 | +22 MiB |
| B | 1,152 | 24,700,260 / 24,700,260 / 24,700,256 | -4 KiB | 30,438 / 30,460 / 30,460 | +22 MiB |

The 22 MiB HBM change happened during initialization and then plateaued; RSS
did not grow across either campaign.  Total generation was 2,304 tokens, with
no crash, deadlock, traceback, NaN, or Inf.

```text
P3_STABILITY = A3_VERIFIED_READY
```

## Gate

```text
R5_ALIGNED_P3 = A3_VERIFIED_READY
ALIGNED_DSV2_LITE_TP1_MULTI_PLACEMENT = A3_VERIFIED_READY
```

This does not qualify TP2/HCCL, Graph, Deferred Experts, Dynamic Placement,
MTP, Speculative Decoding, W8A8, MXFP4, multi-NPU, or production concurrency.
