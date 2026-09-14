# Round 5 Final Summary

## Final outcome

```text
ROUND5_UPSTREAM_ALIGNMENT = A3_VERIFIED_READY
R5_ALIGNED_P1 = A3_VERIFIED_READY
R5_ALIGNED_P2 = A3_VERIFIED_READY
R5_ALIGNED_P3 = A3_VERIFIED_READY
ALIGNED_DSV2_LITE_TP1_MULTI_PLACEMENT = A3_VERIFIED_READY
```

Round 5 migrated the qualified DeepSeek-V2-Lite TP1 profile onto the pinned
upstream KTransformers architecture without rebasing or merging Round 4A's
historical patch stack.  Raw upstream was treated as a falsifiable candidate;
only the minimal missing DSV2-Lite, mapping and numerical semantics were
ported after the raw probes.

## Local reference

| Item | Value |
|---|---|
| Round 4A tag | `round4a-tp1-multiplacement-a3-verified` |
| tag object | `9e2c9e220337f57b55f1fe9b50da0862ccc0cf64` |
| tag target | `096a6ef860a01dc0a3c8f2cc5618a396425ee09b` |
| local SGLang child | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |

## Upstream freeze and aligned source

| Item | Value |
|---|---|
| pinned KTransformers | `440df948027128318bc30a11a545433eb0c7d383` |
| pinned SGLang | `3424f35d0e60f03b9e75d9b90941f8f4cb944524` |
| official Ascend PR | `kvcache-ai/ktransformers#2157`, merged |
| strategy | `UPSTREAM_FIRST` |
| aligned parent | `be5b48621db32fb894402e3e886060dea263cdc9` |
| aligned SGLang | `412219b4b05a888808f0b88d2963a8b6164a0fda` |
| final tag | `round5-upstream-aligned-tp1-a3-verified` |
| tag object | `94baf6272dd86eae7e465a04f4ead630032400d4` |
| audit qualification commit | `38b9ab1b5` |
| compact evidence commit | `5b3eaf2b4` |

## Model and frozen contracts

| Item | Value |
|---|---|
| model | `deepseek-ai/DeepSeek-V2-Lite` |
| full revision | `604d5664dddd88a0433dbae533b7fe9472482de0` |
| tokenizer SHA256 | `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669` |
| P3 placement SHA256 | `77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d` |
| P3 GGUF SHA256 | `992d32e0de59e135464f77fc4869e810f3d71b3ec01f6ed0996cbeaa58e49df4` |
| pairwise contract SHA256 | `223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717` |
| pairwise bound / K | `2.1875 / 32` |
| quality manifest SHA256 | `c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce` |

## A3 environment

| Item | Value |
|---|---|
| host | `a3-server-00` |
| aligned container | `kt-r5-upstream-aligned` |
| CPU affinity | `0-15` |
| NPU visibility | NPU0 only |
| CANN | `9.0.0` |
| Python | `3.11.15` |
| torch | `2.9.0+cpu` |
| torch_npu | `2.9.0.post2` |
| host modified | NO |
| business containers modified | NO |

## Architecture result

Upstream now owns ACL callback processing, stream subscription, pinned host
buffers, framework-managed transfers, immediate host submission on NPU, CPU
drain-before-H2D, buffer slots and KTEP coordination.  Round 5 did not restore
the local standalone Hybrid coordinator or raw ACL pointer copies.

Minimal retained ports cover:

- DSV2-Lite 1408 LLAMAFILE dimensions and standalone CPU plane;
- active NPU rather than CUDA-only capability, stream and synchronization;
- logical/physical expert mapping and 60-expert NPU boundary;
- BF16-visible expert/route boundaries;
- FP32 CPU/NPU routed merge with one final BF16 cast;
- callback state lifetime and focused behavioral regressions.

## Foundation requalification

| Gate | Result |
|---|---:|
| raw upstream build | PASS |
| upstream runtime primitives | PASS |
| raw DSV2-Lite capability | BLOCKED as recorded; not retroactively changed |
| Alignment ADR | `UPSTREAM_FIRST` |
| R2A current public contracts | 18/18 PASS |
| R2B framework transfer | 5/5 PASS |
| R2C/SGLang KTEP contracts | 4/4 PASS |
| Round3 Layer17/Expert8 | PASS |

## P1 / P2 / P3 gate matrix

| Gate | P1 | P2 | P3 |
|---|---:|---:|---:|
| system semantics | PASS | PASS | PASS |
| same-path determinism | PASS | PASS | PASS |
| clean restart | PASS | PASS | PASS |
| frozen pairwise contract | PASS | PASS | PASS |
| CPU layer/expert coverage | 1/1, 4/4 | 4/4, 16/16 | 8/8, 32/32 |
| CPU-not-hit control | PASS | PASS | PASS |
| free generation | qualified | qualified | H2 16/16, F 6/6 exact vs Round 4A |
| downstream quality | PASS | PASS | PASS |
| lifecycle/stability | 1000-forward PASS | PASS | 2304-token PASS |
| retained regression | PASS | PASS | PASS |

## Pairwise scaling

| Profile | Q2 max | Q2 overflow | H2 max | H2 overflow | Stable exact | Ambiguous membership |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 1.75 | 0 | 0.5625 | 0 | 292/292 | 220/220 |
| P2 | 2.125 | 0 | 0.9375 | 0 | 292/292 | 220/220 |
| P3 | 1.9375 | 0 | 0.5625 | 0 | 292/292 | 220/220 |

No profile exceeded the immutable bound.  There was no stable token flip or
ambiguity escape.

## Quality

| Profile | C-Eval | GSM8K MC | Invalid |
|---|---:|---:|---:|
| all-NPU | 27/128 | 31/128 | 0 |
| aligned P1 | 27/128 | 31/128 | 0 |
| aligned P2 | 27/128 | 31/128 | 0 |
| aligned P3 | 27/128 | 31/128 | 0 |

All Hybrid correctness vectors exactly match all-NPU.  Discordant pairs are
zero and each 20,000-sample paired-bootstrap 95% CI is `[0,0]`.

## P3 stability

Campaign A and B each generated 1,152 tokens.  Outputs matched across
campaigns.  RSS ended 16 KiB and 4 KiB below the respective starts.  HBM moved
from 30,438 to 30,460 MiB during initialization and plateaued.  There was no
crash, deadlock, non-finite output, or unbounded memory growth.

## Final acceptance

```text
ROUND5_UPSTREAM_ALIGNMENT = A3_VERIFIED_READY
UPSTREAM_PARENT = 440df948027128318bc30a11a545433eb0c7d383
UPSTREAM_SGLANG = 3424f35d0e60f03b9e75d9b90941f8f4cb944524
ALIGNED_PARENT = be5b48621db32fb894402e3e886060dea263cdc9
ALIGNED_SGLANG = 412219b4b05a888808f0b88d2963a8b6164a0fda
ALIGNED_P1 = A3_VERIFIED_READY
ALIGNED_P2 = A3_VERIFIED_READY
ALIGNED_P3 = A3_VERIFIED_READY
ALIGNED_DSV2_LITE_TP1_MULTI_PLACEMENT = A3_VERIFIED_READY
```

Round 5 does not qualify TP2/HCCL, multi-NPU, Graph, Deferred Experts,
Dynamic Placement, MTP, Speculative Decoding, W8A8, MXFP4, multi-node, or
production concurrency.  Those remain separate risk surfaces.
