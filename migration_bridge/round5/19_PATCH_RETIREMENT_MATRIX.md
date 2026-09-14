# Round 5 — Patch Retirement Matrix

Evidence state: `CODE_INSPECTED` plus the referenced A3 regressions.

| Local Round 4A fix | Upstream counterpart | Final action | Reason | Regression protecting it |
|---|---|---|---|---|
| ARM architecture and feature detection | upstream multi-variant ARM detection | `DROP_AS_UPSTREAMED` | upstream is the architecture owner; minimal native-identity correction was applied | R2A detector tests |
| LLAMAFILE F32 GGUF, BF16 I/O | upstream LLAMAFILE/CPUInfer | `KEEP_MINIMAL` | raw support exists; DSV2-Lite compatibility remains required | R2A GGUF/routed tests, Round3 |
| intermediate size 1408 tail | raw upstream rejected the tail | `PORT_TO_UPSTREAM` | required by real DSV2-Lite expert dimensions | Round3, P1-P3 real model |
| standalone single-threadpool forward | upstream wrapper assumed accelerator context in one path | `PORT_TO_UPSTREAM` | CPU-only expert plane remains a required foundation | R2A lifecycle |
| local callback/vendor abstraction | upstream ACL report worker and subscription | `DROP_AS_UPSTREAMED` | upstream is a strict runtime superset | R2B callback probes |
| raw pointer ACL transfer experiments | upstream framework `Tensor.copy_` | `OBSOLETE` | unsafe lifetime/ordering path must not return | R2B transfer and P2/P3 determinism |
| local framework-managed transfer fix | upstream pinned buffers and `copy_` | `DROP_AS_UPSTREAMED` | equivalent principle is upstream-owned | R2B 5-test suite |
| local stream aliases | upstream CUDA-shaped vendor ABI plus active-device resolver | `DROP_AS_UPSTREAMED` | aliases were private compatibility surfaces | active-stream SGLang contract |
| callback-state lifetime closure | upstream worker with incomplete temporary ownership in raw snapshot | `KEEP_MINIMAL` | raw primitive probe exposed lifetime risk | callback lifecycle regression |
| global logical expert IDs | upstream KTEP/EPLB metadata | `KEEP_MINIMAL` | DSV2 mapping boundary required adaptation | C0-C4, per-layer controls |
| logical-to-physical mapping int32 | upstream metadata was int64 at model boundary | `PORT_TO_UPSTREAM` | compiled LLAMAFILE boundary requires contiguous int32 | KTEP mapping contract |
| physical NPU expert count from weight dim0 | upstream grouped MoE integration | `PORT_TO_UPSTREAM` | sparse expert load must expose 60 physical experts | P1/P2/P3 controls |
| placeholder ID 0 plus zero weight | upstream Ascend route sanitizer | `KEEP_MINIMAL` | prevents invalid negative expert IDs reaching GMM | route sanitizer test |
| routing/scaling/shared exact-once | upstream KTEP structure | `KEEP_MINIMAL` | DSV2 model boundary needed exact preservation | C0-C4 and per-layer controls |
| CUDA capability guards on NPU | upstream paths still probed CUDA-only APIs | `PORT_TO_UPSTREAM` | raw DSV2 import failed without guards | KTEP import and serving |
| router FP32 fallback | upstream router JIT path was CUDA-oriented | `PORT_TO_UPSTREAM` | preserves qualified NPU routing behavior | pairwise Q2/H2 |
| BF16-visible gate/up/mul/down | absent in raw upstream LLAMAFILE behavior | `PORT_TO_UPSTREAM` | Round 4A.1 proved these boundaries necessary | expert and pairwise regression |
| BF16-visible route weight | absent in raw upstream profile | `PORT_TO_UPSTREAM` | prevents previously observed margin drift | pairwise Q2/H2 |
| FP32 CPU/NPU partial merge | raw SGLang path was dtype-dependent | `PORT_TO_UPSTREAM` | prevents independent BF16 rounding before merge | C0-C4 exact reference |
| final one-time BF16 cast | conditional hook existed but did not cover profile | `PORT_TO_UPSTREAM` | part of frozen numerical semantics | pairwise and quality gates |
| local standalone Hybrid coordinator | upstream KTEP is the sole coordinator | `DROP_AS_UPSTREAMED` | duplicate architecture would violate source-of-truth invariant | SGLang KTEP contracts |
| bisection/debug environment knobs | no production need | `TEST_ONLY` | retain evidence, remove architectural dependence | none; diagnostic history only |
| Graph/Deferred/Dynamic/MTP/W8A8/MXFP4 | present in upstream | `DEFER_TO_FUTURE` | code presence is not qualification | explicitly OFF in every profile |

## Result

The aligned implementation retains only DeepSeek-V2-Lite compatibility,
mapping and frozen numerical semantics that raw upstream did not satisfy.  It
retires the local callback coordinator, raw-transfer variants, standalone
Hybrid coordinator, and private diagnostic compatibility APIs.

```text
ROUND5_PATCH_RETIREMENT = COMPLETE
```
