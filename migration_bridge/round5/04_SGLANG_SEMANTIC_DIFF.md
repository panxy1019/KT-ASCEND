# Round 5 — SGLang Semantic Diff

Evidence state: `CODE_INSPECTED`.

| Side | Commit |
|---|---|
| Round 4A child | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |
| pinned upstream child | `3424f35d0e60f03b9e75d9b90941f8f4cb944524` |

The comparison is semantic. The upstream file is much larger because it also
supports V4, GLM, layerwise prefill, dynamic expert update, LoRA, and several
quantized layouts that are out of the Round 5 qualification profile.

## KTEP coordinator

Both children use `KTEPWrapperMethod` as the main coordinator and follow the
same high-level submit → device experts → sync → merge flow. The upstream child
adds device-module abstractions and a richer report/subscription-aware backend.
No second Hybrid coordinator is needed.

## Expert identity and ownership

Both retain a CPU logical mask and compact device experts. Both create `int32`
logical-to-physical mapping tables. The upstream child also consumes EPLB
physical-to-logical metadata and falls back to identity mapping.

The exact placeholder behavior differs: the immutable local Ascend path has a
validated valid-ID-zero/zero-weight contract, while generic upstream masking
can express CPU routes as `-1`. This is a kernel-interface risk and remains
`UNKNOWN_NEEDS_PROBE`; no conclusion is inferred from the Python representation
alone.

## Routed scaling and shared expert

Both disable `routed_scaling_factor` inside the wrapped device MoE so CPU and
NPU routed contributions are not scaled differently. DeepSeek applies the
factor at the common model boundary. Shared expert execution remains outside
the routed ownership split. Round2C and Round3 must re-establish exactly-once
behavior.

## Transfer and stream semantics

The local child contains the Round4A.5 safety synchronization/clone around the
CPU output hand-off. The pinned upstream parent/child combination adds:

- an ACL report worker and stream subscription;
- host-synchronous immediate NPU CPUInfer submission;
- explicit device wait before CPU input consumption;
- host CPUInfer drain before H2D;
- framework-managed pinned copies;
- an NPU synchronization/clone before a shared output buffer is merged.

This is a likely upstream superset, but only the P2-like repeated serving test
can close the lifetime invariant.

## Numerical merge

The immutable child marks the unquantized device contribution for FP32 return,
adds CPU and NPU partials in FP32, and performs one final cast to the hidden
dtype. That behavior is the production half of the Round4A.1 Fix B.

The pinned upstream KTEP source has a conditional final cast if
`_kt_return_fp32_contribution` is set. Static search found no assignment in the
pinned upstream unquantized method. Therefore raw upstream DSV2 is currently
classified `UNKNOWN_NEEDS_PROBE`, with a minimal port expected if the A3 probe
confirms BF16-before-merge behavior.

## Weight loading

The upstream child has a newer compact-expert loader plus many layout-specific
loaders. The only relevant Round 5 question is whether it can load the frozen
DeepSeek-V2-Lite BF16/unquantized NPU weights and the exact F32 GGUF CPU experts
without changing model identity. W8A8/MXFP4 success cannot substitute for this
probe.

## Qualification status

| Feature | Source presence | Round 5 qualification |
|---|---:|---:|
| static KTEP | yes | P1/P2/P3 A3 verified |
| DSV2 BF16/unquantized | code path candidate | P1/P2/P3 A3 verified |
| F32 GGUF LLAMAFILE | code path candidate | P1/P2/P3 A3 verified |
| Graph | yes | off / not qualified |
| Deferred experts | yes | off / not qualified |
| Dynamic placement | yes | off / not qualified |
| W8A8/MXFP4 | yes | out of scope |
