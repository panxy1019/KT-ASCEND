# Round 5 — Runtime Semantic A/B

Evidence state: local column combines `CODE_INSPECTED` with immutable Round 4A
evidence; raw upstream combines `CODE_INSPECTED` and the raw A3 probes; aligned
final is `A3_VERIFIED_READY` for the frozen TP1 profile.

| Semantics | Local Round 4A | Upstream raw `440df948` | Aligned final |
|---|---|---|---|
| D2H API | framework `Tensor.copy_` into retained pinned tensor | framework async `copy_` into pinned buffer | upstream framework `copy_`; 1000-cycle transfer and P3 serving verified |
| D2H visibility | explicit local NPU synchronization before CPU consume | `_wait_device` before synchronous host submit | active accelerator stream handle plus device wait before host consumption |
| expert IDs / weights D2H | framework copy, dtype boundary enforced | framework copy into pinned buffers | framework copy; int64 routes and FP32 weights verified exact in transfer tests |
| H2D API | framework `Tensor.copy_` | framework non-blocking `copy_` after CPU drain | framework `copy_` after CPUInfer drain |
| H2D visibility | blocking local hand-off; child protects shared-buffer consumption | stream dependency plus NPU synchronize/clone in child | active NPU stream/event dependency; P2/P3 repeated serving deterministic |
| pinned lifetime | tensor retained through consumer | persistent buffer pool and tensor ownership | upstream persistent pinned buffers retained; two 1152-token campaigns verified |
| immediate CPU submit | host `cpu_infer.submit` on NPU path | host `cpu_infer.submit` on NPU path | retained upstream immediate NPU submit |
| deferred submit | off in profile | callback implementation present; off in profile | off / not qualified |
| callback worker | no first-class ACL report worker | `aclrtProcessReport` worker | upstream worker retained; init/subscription/shutdown verified |
| stream subscription | local vendor/callback abstraction | `aclrtSubscribeReport`, idempotent | upstream idempotent subscription verified |
| CPU sync | host CPUInfer drain | wait device then host CPUInfer drain | upstream drain retained; 1000-cycle and full-model verification passed |
| buffer reuse | qualified layer/slot lifetime | buffer-depth/layer slot plus explicit drain | upstream slots plus explicit drain; P3 8-layer lifecycle verified |
| CPU expert stage rounding | BF16-visible compatibility mode | FP32 stages; compatibility switch absent | minimal BF16-visible gate/up/mul/down port retained |
| route weight | BF16-visible before accumulation | raw FP32 weight | BF16-visible route boundary retained |
| CPU partial dtype | FP32 | FP32 C++ output, Python path-dependent | FP32 routed contribution verified |
| NPU partial dtype | forced FP32 contribution for unquantized KTEP | no setter found for conditional FP32 return | minimal SGLang FP32 contribution port retained |
| merge dtype | FP32 CPU + FP32 NPU | path-dependent in raw DSV2 | FP32 CPU+NPU merge verified exactly in C0-C4 controls |
| final cast | once after FP32 merge | conditional hook exists | one final BF16 cast retained and pairwise qualified |
| shared expert | once at model boundary | once at model boundary by design | exact-once semantics verified at every P2/P3 selected layer |

## A/B invariants

The aligned column may only be filled from A3 evidence. It must retain:

```text
same model/tokenizer/GGUF/placement
same routing, ownership, scaling, and shared-expert semantics
framework-visible transfer dependencies
production CPU/NPU overlap
BF16-visible expert stages
FP32 partial merge with one final cast
```

Raw upstream failures remain raw failures.  The aligned PASS does not
retroactively convert the raw DeepSeek-V2-Lite capability probe into a raw
upstream PASS.
