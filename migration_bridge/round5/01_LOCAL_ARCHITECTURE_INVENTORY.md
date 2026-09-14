# Round 5 — Local Round 4A Architecture Inventory

Evidence state: `CODE_INSPECTED`. Runtime conclusions remain backed by the
immutable Round 4A A3 evidence referenced from the final tag.

## Reference boundary

This inventory describes parent `096a6ef860a01dc0a3c8f2cc5618a396425ee09b`
with SGLang child `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2`.
It is a read-only migration oracle, not the Round 5 implementation base.

## Build and CPU plane

- `kt-kernel` builds the LLAMAFILE/CPUInfer expert plane for aarch64.
- DeepSeek-V2-Lite uses F32 GGUF expert weights containing BF16-valued source
  values, BF16 hidden-state I/O, and `moe_intermediate_size=1408`.
- The Round 4A build avoids unsupported Kunpeng 920 SVE/BF16/I8MM assumptions.
- `LlamafileMoEWrapper` explicitly enables `bf16_numerical_compat` for the
  validated DeepSeek expert path.
- CPUInfer owns its host worker queue. It does not own or arbitrarily
  synchronize NPU work.

## Stream, task, and transfer lifecycle

The production NPU path in `kt-kernel/python/experts_base.py` has the following
ordering:

1. hidden state, route weights, and expert IDs are copied into pinned CPU
   tensors with framework `Tensor.copy_`;
2. the D2H visibility point is completed before CPUInfer consumes those
   buffers;
3. CPUInfer work is submitted to the host worker pool while independent NPU
   expert work proceeds;
4. CPUInfer is drained before H2D;
5. the CPU partial is returned with framework `Tensor.copy_` and consumed only
   after its dependency is visible.

Round 4A.5 removed raw pointer-based `acl.rt.memcpy` from this production
transfer boundary. This is the fix that closed P2 same-path nondeterminism.
The local vendor abstraction and CPUInfer callback state use RAII and preserve
exceptions across the callback ABI. The local path does not implement the
upstream ACL report-processing worker/subscription lifecycle as a first-class
runtime component.

## Buffer ownership

- CPU tensors are pinned and retained through CPU consumption.
- Per-layer/slot ownership and the shared staging buffer have explicit reuse
  boundaries.
- The H2D tensor remains alive until the framework dependency is established.
- Diagnostic private scratch/output controls remain default-off and are not
  part of the qualified production profile.

## Expert identity and routing

- Global logical expert IDs are canonical at the router boundary.
- `logical_to_gpu_index` is an `int32` device mapping for compact physical NPU
  weights.
- CPU and NPU ownership are disjoint.
- NPU placeholder expert ID is valid (`0`) and paired with zero route weight.
- Physical NPU expert count comes from the actual compact weight dimension.
- The routed scaling factor is removed from the device MoE method and applied
  exactly once at the model boundary.
- Shared expert output is applied exactly once outside the routed CPU/NPU
  split.

## Numerical semantics

The local LLAMAFILE expert has an explicit compatibility mode that exposes the
same BF16-valued boundaries as the qualified Ascend path:

```text
gate output          -> BF16 value
up output            -> BF16 value
SwiGLU multiply      -> BF16 value
down output          -> BF16 value
route weight         -> BF16 value
```

The CPU and NPU routed partials are accumulated in FP32 and cast once at the
combined output boundary:

```text
FP32(CPU partial) + FP32(NPU partial) -> one final BF16 cast
```

These are correctness semantics, not optional performance tuning. They are
protected by the frozen pairwise contract (`B_pair=2.1875`, `K=32`).

## Serving integration

- `KTEPWrapperMethod` remains the single Hybrid coordinator.
- The wrapper masks/remaps only the NPU-owned routed experts; CPUInfer receives
  logical IDs and performs its own skip decision.
- SGLang remains the model/serving source of truth.
- The qualified profile is TP1, batch 1, greedy, one NPU, static placement,
  Graph/Deferred/Dynamic/MTP/Speculative all off.

## Local strengths and migration liabilities

| Area | Assessment |
|---|---|
| DSV2-Lite semantics | qualified through P1/P2/P3 |
| numerical boundary | explicit and held-out verified |
| framework transfer | qualified through multi-layer determinism |
| ACL report worker | less complete than pinned upstream |
| generic callback lifecycle | safe local abstraction, but duplicated architecture |
| future Graph/Deferred behavior | out of scope and not qualified |
| architecture maintenance | local production patches should be retired where upstream is equivalent |

