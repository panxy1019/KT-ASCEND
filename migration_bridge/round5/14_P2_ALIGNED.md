# Round 5 — Aligned P2 Requalification

Evidence state: `A3_VERIFIED_READY`

## Frozen profile and recovered artifact

```text
TP: 1
device: NPU0 only
CPU layers: {1, 9, 17, 26}
CPU experts per selected layer: 4
selected CPU experts: 16
execution: production overlap
B_pair: 2.1875
candidate K: 32
Graph/Deferred/Dynamic/MTP/Speculative: OFF
```

The P3 GGUF cannot represent P2 because it intentionally does not contain
Layer 9.  The first P2 launch therefore failed at weight loading with the
explicit error `blk.9.ffn_gate_exps.weight not found`; it never entered
inference.  The original Round 4A P2 artifact was then recovered from A3
persistent storage and used without conversion:

| Property | Value |
|---|---|
| file | `deepseek-v2-lite-p2-f32.gguf` |
| size | 8,858,371,392 bytes |
| SHA256 | `2c4cd307a53b761c4191cf108d1e00ec2bd2c99b48a595a28ac8fdd0e0edd1fa` |
| manifest SHA256 | `9d0812bb0a30f22f2cb2f3454999d89e3abd73964fe8e47843c929aad802b6a7` |
| manifest layers | `{1,9,17,26}` |
| dtype | F32, BF16-valued source weights |

The manifest records the same model index and four source-safetensors hashes
as the frozen DeepSeek-V2-Lite revision.  No placement or artifact was rebuilt
after observing aligned P2 output.

## Same-process determinism and clean restart

The production-overlap profile completed 9 prompts x 10 repeats x 64 greedy
tokens.  Every prompt produced one trajectory.  The 8/16/32-token requests
were exact prefixes of the corresponding 64-token output.  All 5,760 recorded
logprobs were finite.

A new serving process reproduced all 9 trajectories exactly.  Its 576
recorded logprobs were also finite.

```text
P2_SAME_PROCESS_DETERMINISM = PASS
P2_CLEAN_RESTART = PASS
```

## Frozen pairwise numerical contract

The actual Round 4A Q2/H2 baseline logits were used.  The bound and candidate
universe were not refitted.

| Corpus | Positions | Pairs | Pairwise p50 | p95 | p99 | Max | Stable exact | Ambiguous membership | Overflow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2 frozen subset | 256 | 8,117 | 0.0625 | 0.25 | 0.4375 | 2.125 | 137/137 | 119/119 | 0 |
| H2 frozen subset | 256 | 8,098 | 0.0625 | 0.25 | 0.375 | 0.9375 | 155/155 | 101/101 | 0 |

All logits were finite.  H2 maximum distortion is above the historical P2
value 0.75 but remains below frozen `B_pair=2.1875`; there is no stable flip,
ambiguity escape, or envelope overflow.  The increase is retained as scaling
telemetry and was not used to modify the contract.

```text
P2_PAIRWISE_NUMERICAL_CONTRACT = A3_VERIFIED_READY
```

## Route coverage

Coverage was measured with the upstream framework
`ExpertDistributionRecorder`, not with a local production routing fork.  Its
logical counts and KT GPU ownership masks covered 1,024 forward passes; the
ownership mask was invariant throughout.

| Layer | Frozen CPU experts | Per-expert route hits | Total |
|---:|---|---|---:|
| 1 | `{31,43,50,57}` | `4035,5113,3752,4982` | 17,882 |
| 9 | `{38,41,45,46}` | `4420,5804,4011,5841` | 20,076 |
| 17 | `{6,8,25,36}` | `7008,964,9414,3586` | 20,972 |
| 26 | `{10,26,30,56}` | `2802,3804,4328,4925` | 15,859 |

```text
layers exercised: 4/4
selected CPU experts exercised: 16/16
total CPU route hits: 74,789
count values nonnegative: YES
ownership mask invariant: YES
```

The total exactly reproduces the historical Round 4A workload count.  The
full matched-history and determinism outputs are finite; the next section adds
direct routed-weight/merge controls at every selected layer.

## Per-layer system and CPU-not-hit controls

Each selected layer was independently executed through the production KTEP
path with its four frozen CPU experts.  C0 contained no CPU-owned route; C1-C4
covered one, two, positional, and multi-token CPU routes.

| Layer | KTEP vs explicit CPU+NPU | Shared/scaling once | C0 vs all-NPU | physical NPU experts |
|---:|---:|---:|---:|---:|
| 1 | exact | exact | exact | 60 |
| 9 | exact | exact | exact | 60 |
| 17 | exact | exact | exact | 60 |
| 26 | exact | exact | exact | 60 |

Every reported max-absolute and relative-L2 error is zero.  This directly
checks disjoint ownership, logical-to-physical mapping, routing weight once,
routed scaling once, shared expert once, and CPU-not-hit non-contamination.

## Downstream quality

The frozen quality manifest SHA is
`c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce`.

| Benchmark | Frozen all-NPU | Aligned P1 | Aligned P2 | P2 invalid | P2 vs all-NPU discordant | Bootstrap delta 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| C-Eval | 27/128 | 27/128 | 27/128 | 0 | 0 | [0, 0] |
| GSM8K MC | 31/128 | 31/128 | 31/128 | 0 | 0 | [0, 0] |

P2 is also pairwise identical to aligned P1 on both benchmark correctness
vectors.  Result: `QUALITY_VERIFIED`.

## Lifecycle and retained regression boundary

The same aligned production source had already passed the Round 2A 1,000
forward lifecycle and Round 2B 1,000 framework-managed transfer cycles.
During P2, 1,024 recorded full-model forward passes plus the 117-request
determinism matrix and 512 matched-history positions completed without crash,
deadlock, traceback, non-finite output, or transfer-order nondeterminism.

The retained aligned R2A/R2B/R2C, SGLang KTEP, Round 3, and P1 results were
produced from the same production source and child SHA.  No production code
changed between those runs and P2.

## Evidence integrity

| Artifact | SHA256 |
|---|---|
| `a3_aligned_p2_repeat64.json` | `77e920c26eefb1bab01abc828e9e859f43ccab29179781b84e38af041b611bf5` |
| `a3_aligned_p2_same_path_summary.json` | `9fe718e544c180f2f065d6a02a8a0391815166aa800b4f70d23a8c5da358abce` |
| `a3_aligned_p2_clean_restart.json` | `254fd8a6f90e76659711e4967a33a23f311f7aa88ddfec24e0ca0d62a02c11b2` |
| `a3_aligned_p2_clean_restart_summary.json` | `6ba037e68b7fa4c917a426c131fbd1975a4b99ad56c2af8de0485572b842fa7d` |
| `a3_aligned_p2_q2_contract.json` | `7692ece00b336bfd1b716bd3f1744dbf55c9b122957d7d3d00f796eb6231ec80` |
| `a3_aligned_p2_h2_contract.json` | `4fcd0765055aba0ae5c450c1c090c64f3dfb322950bfc5febc663b23c24c5b4e` |
| `a3_aligned_p2_route_coverage.json` | `99cc85e3a9095ffc972ba503ba48fe97b176b0fc20c497d8df6c460eaaaac7a3` |
| `a3_aligned_p2_quality.json` | `15620ff1ba93c1d04c4bbc90f77c1ce3ee4bff315525c63d2cd80dffb0c0ef1e` |
| `a3_aligned_p2_quality_vs_allnpu.json` | `2fc8ffb447737b716b4112302d237cea92a193b58693b471f2c19eb68974cc13` |
| `a3_aligned_p2_expert_distribution.pt` | `b93980dc924dd53683afe92645a8d061fa7d142bb1e4cfb9e07f51e96ed244b8` |
| `a3_aligned_p2_gpu_expert_distribution.pt` | `a04151e97d9b30b308bdc05fa7c443a4f7333ed9f64ecb4cb340efa24de3d6af` |

## Gate

```text
R5_ALIGNED_P2 = A3_VERIFIED_READY
```

This promotion qualifies only the frozen P2 profile.  P3 remained unrun until
this gate and document were complete.
