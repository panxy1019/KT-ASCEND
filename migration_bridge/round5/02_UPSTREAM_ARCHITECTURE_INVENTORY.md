# Round 5 — Pinned Upstream Architecture Inventory

Evidence state: `CODE_INSPECTED`. Nothing in this document is a raw-upstream
`A3_VERIFIED` claim.

## Frozen source

| Component | Commit |
|---|---|
| KTransformers | `440df948027128318bc30a11a545433eb0c7d383` |
| SGLang gitlink | `3424f35d0e60f03b9e75d9b90941f8f4cb944524` |
| Ascend integration | parent commit `040a498` / PR `#2157` |

The inventory is from these commits, not from moving `main` tips.

## Build and packaging

- `CPUINFER_USE_ASCEND_NPU=ON` selects the Ascend vendor build and links CANN.
- The setup layer forwards detected aarch64 features into CMake switches.
- ARM target construction begins at `armv8.2-a`; DOTPROD and FP16 are enabled
  when detected, while SVE/BF16/I8MM are independently gated.
- KML and BLIS are disabled by default for the Ascend CPUInfer build.
- The source includes CUDA/ROCm/MUSA/MACA and quantized paths that remain
  present but out of scope for the Round 5 BF16/F32 qualification profile.

Raw buildability on the exact A3 environment is still a required gate.

## Callback and stream lifecycle

`ascend_callback_worker.cpp` implements an ACL report-processing lifecycle:

- a dedicated host worker calls `aclrtProcessReport`;
- initialization captures/sets the ACL context;
- a readiness barrier precedes stream subscription;
- streams are subscribed with `aclrtSubscribeReport`;
- shutdown drains, unsubscribes, and joins the worker;
- start/subscription are idempotent.

The Ascend vendor shim maps the CUDA-shaped host callback interface to
`aclrtLaunchCallback`. This is an architectural superset of the local
Round 4A report-processing model, subject to A3 primitive verification.

## CPU submission and synchronization

For NPU immediate work, `experts_base.py` intentionally does not depend on
host-observability of an ACL `NO_BLOCK` callback. It:

1. queues framework D2H copies into pinned buffers;
2. explicitly waits for device completion before immediate CPU consumption;
3. submits the CPU task synchronously to CPUInfer's WorkerPool;
4. runs NPU experts while that worker task executes;
5. waits for NPU work and drains CPUInfer on the host;
6. copies the completed pinned CPU output back with framework `copy_`.

Deferred mode retains callback submission and Graph-aware safeguards. Deferred
and Graph are `PRESENT_IN_CODE / NOT_QUALIFIED` in Round 5.

## Transfer and buffer ownership

- Hidden state, route weights, and IDs use pinned host buffers and framework
  `copy_(..., non_blocking=True)`.
- CPU output H2D uses framework `copy_` after CPUInfer drain.
- Buffer slots are selected by layer index and buffer depth.
- SGLang owns a persistent shared staging buffer and dual-stream hand-off.
- The pinned tensor objects carry lifetime through the transfer boundary.

The implementation directly addresses the failure mechanism found in
Round 4A.5, but must still pass the 1000-repeat primitive and P2/P3 serving
gates on A3.

## Expert identity, mapping, and scaling

- `KTEPWrapperMethod` keeps a logical mask and builds `int32`
  `logical_to_gpu_index` / `gpu_index_to_logical` tables.
- The device method receives compact physical expert weights.
- CPU route ownership is the complement of the NPU mask.
- `routed_scaling_factor` is disabled inside the wrapped device MoE and left
  to the model boundary, avoiding double application.
- EPLB-aware physical-to-logical metadata is supported, with identity fallback.

The upstream child is substantially newer and contains dynamic placement,
layerwise loading, LoRA, GLM and V4-specific machinery. Those features are
disabled for the Round 5 MVP and cannot inherit a READY state from source
presence.

## Raw numerical semantics and identified gaps

The raw upstream LLAMAFILE kernel produces FP32 gate/up/down buffers and applies
SwiGLU and route weighting in FP32. Unlike the immutable Round 4A implementation,
the pinned code does not expose a `bf16_numerical_compat` switch and does not
round gate, up, multiply, down, and route weight to BF16-visible values.

The upstream SGLang Hybrid merge contains a conditional final cast based on
`_kt_return_fp32_contribution`, but static inspection found no setter for this
flag in the pinned upstream unquantized DSV2 path. Therefore raw upstream must
not be assumed to reproduce the Round 4A FP32 CPU/NPU partial accumulation
contract.

These are `UNKNOWN_NEEDS_PROBE` until captured-input and A3 runtime evidence is
collected. They are likely minimal-port candidates, not grounds to alter the
frozen numerical acceptance contract.

## Initial risk assessment

| Area | Static assessment |
|---|---|
| ACL report worker | `UPSTREAM_SUPERSET` candidate |
| framework D2H/H2D | `UPSTREAM_EQUIVALENT` or superset candidate |
| immediate NPU submit ordering | explicitly designed for host observability |
| DSV2 BF16-visible boundaries | absent in raw upstream |
| FP32 Hybrid accumulation | not enabled for raw unquantized DSV2 by inspected code |
| logical/physical mapping | broadly equivalent, child behavior must be probed |
| DSV2 F32 GGUF loader | `UNKNOWN_NEEDS_PROBE` |
| Graph/Deferred/W8A8/MXFP4 | present, out of qualification scope |

