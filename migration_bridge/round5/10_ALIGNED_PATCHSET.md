# Round 5 — Minimal Upstream-Aligned Patch Set

Status: `IMPLEMENTED; A3 REQUALIFICATION IN PROGRESS`

The aligned candidate starts at pinned upstream parent
`440df948027128318bc30a11a545433eb0c7d383` and child
`3424f35d0e60f03b9e75d9b90941f8f4cb944524`.  No Round 4A history was
merged or rebased.  Each retained change corresponds to a raw A3 failure or a
previously qualified invariant.

## Parent commits

| Commit | Semantic reason | State |
|---|---|---|
| `dfcebfb` | Guard the CUDA-only SGLang capability probe on NPU | A3 import verified |
| `bbf8b36` | Permit DeepSeek-V2-Lite's 1408 LLAMAFILE tail | A3 real model verified |
| `255cf14` | Preserve BF16-visible expert stages and FP32 routed merge | A3 real Layer17 verified |
| `6368cde` | Close CPUInfer callback state lifetimes | A3 runtime exercised |
| `0d8fb1d` | Preserve DSV2 logical/physical expert mapping | A3 strict route verified |
| `c84537b` | Synchronize the active accelerator rather than CUDA only | A3 NPU verified |
| `bf1628b` | Preserve NPU router FP32 fallback | A3 serving verified |
| `e07051f` | Resolve active CUDA/NPU stream handles for CPUInfer | A3 serving verified |
| `e98916c` | Report and select native ARM kernels on aarch64 | A3 R2A verified |
| `99b3906` | Preserve standalone CPU expert execution | A3 R2A verified |
| `9743248` | Consume the physical-to-logical LLAMAFILE mapping | A3 mapping verified |
| `01c99b9` | Expose the pinned-host transfer contract publicly | A3 R2B verified |
| `958226e`, `be5b486` | Add transfer and callback-lifecycle regressions | A3 5/5 verified |

## SGLang child commits

| Commit | Semantic reason | State |
|---|---|---|
| `89288d20e` | Guard CUDA-only MXFP8 capability inspection | A3 verified |
| `44a72c677` | Keep DSV2 physical expert count and placeholder semantics | A3 verified |
| `ef0b59a92` | Return FP32 unquantized routed contributions | A3 verified |
| `b4f9cb959` | Apply the one-time final BF16 model boundary | A3 verified |
| `c90482bcc` | Preserve router/route-weight boundary behavior | A3 verified |
| `412219b4b` | Use active accelerator Stream/Event/context APIs in KTEP | A3 verified |

## Explicit upstream ownership retained

- ACL report worker and stream subscription;
- framework-managed pinned `Tensor.copy_` transfers;
- host-visible immediate NPU CPUInfer submission and drain-before-H2D;
- upstream KTEP as the sole Hybrid coordinator;
- upstream loader cache and worker-pool configuration;
- upstream Graph/Deferred/W8A8/MXFP4 code present but unqualified and disabled.

## Deliberately not ported

- the local standalone `HybridMoECoordinator`;
- raw pointer ACL transfer shortcuts;
- diagnostic-only task bindings and method aliases;
- Round 4A bisection environment knobs;
- TP2/HCCL, Graph, Deferred, Dynamic placement, MTP or quantization enablement.

The production delta is therefore a semantic compatibility layer on the
pinned upstream architecture, not a replay of the historical patch stack.

