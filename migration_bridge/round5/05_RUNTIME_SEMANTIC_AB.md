# Round 5 — Runtime Semantic A/B

Evidence state: local column combines `CODE_INSPECTED` with immutable Round 4A
evidence; upstream and aligned runtime columns remain provisional until A3.

| Semantics | Local Round 4A | Upstream raw `440df948` | Aligned final |
|---|---|---|---|
| D2H API | framework `Tensor.copy_` into retained pinned tensor | framework async `copy_` into pinned buffer | pending |
| D2H visibility | explicit local NPU synchronization before CPU consume | `_wait_device` before synchronous host submit | pending |
| expert IDs / weights D2H | framework copy, dtype boundary enforced | framework copy into pinned buffers | pending |
| H2D API | framework `Tensor.copy_` | framework non-blocking `copy_` after CPU drain | pending |
| H2D visibility | blocking local hand-off; child protects shared-buffer consumption | stream dependency plus NPU synchronize/clone in child | pending |
| pinned lifetime | tensor retained through consumer | persistent buffer pool and tensor ownership | pending |
| immediate CPU submit | host `cpu_infer.submit` on NPU path | host `cpu_infer.submit` on NPU path | pending |
| deferred submit | off in profile | callback implementation present; off in profile | off / not qualified |
| callback worker | no first-class ACL report worker | `aclrtProcessReport` worker | pending U1 |
| stream subscription | local vendor/callback abstraction | `aclrtSubscribeReport`, idempotent | pending U1 |
| CPU sync | host CPUInfer drain | wait device then host CPUInfer drain | pending U3/U4 |
| buffer reuse | qualified layer/slot lifetime | buffer-depth/layer slot plus explicit drain | pending multi-layer |
| CPU expert stage rounding | BF16-visible compatibility mode | FP32 stages; compatibility switch absent | expected minimal port |
| route weight | BF16-visible before accumulation | raw FP32 weight | expected minimal port |
| CPU partial dtype | FP32 | FP32 C++ output, Python path-dependent | pending probe |
| NPU partial dtype | forced FP32 contribution for unquantized KTEP | no setter found for conditional FP32 return | expected minimal child port |
| merge dtype | FP32 CPU + FP32 NPU | likely hidden dtype in raw DSV2 | pending probe |
| final cast | once after FP32 merge | conditional hook exists | pending probe |
| shared expert | once at model boundary | once at model boundary by design | pending Round2C/R3 |

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

Raw upstream failures remain raw failures. A later minimal patch may change the
aligned result but cannot retroactively change the upstream-raw column.

