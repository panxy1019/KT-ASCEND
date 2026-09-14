# ADR — Round 5 Upstream Alignment Strategy

Status: `ACCEPTED_AND_A3_QUALIFIED`

Evidence: static semantic audit plus raw upstream A3 build/runtime/capability
probes.

## Context

The immutable Round 4A local line is correctness-qualified but carries a local
Ascend runtime architecture. Pinned upstream independently converged on the
same core transfer lesson: pinned framework copies, explicit visibility, host
CPUInfer drain, and an ACL report-processing worker. Its runtime primitives
build and execute correctly on A3.

Raw upstream is not a DeepSeek-V2-Lite replacement. It fails KTEP import on an
unguarded CUDA capability query and rejects the frozen 1408 intermediate
dimension. Static audit also proves that it lacks the qualified BF16-visible
expert stages and enabled FP32 unquantized Hybrid merge.

## Decision

```text
Option A — UPSTREAM_FIRST
```

The Round 5 implementation base remains parent `440df948...` with SGLang
`3424f35...`. Only small, separately reviewable semantics already proven
necessary by Round 4A are ported. The Round 4A history is not merged or rebased
onto upstream.

## Minimal patch set

1. Guard CUDA-only MXFP8 capability inspection on non-CUDA platforms.
2. Port remainder-safe LLAMAFILE support for `moe_intermediate_size=1408`.
3. Port the BF16-visible gate/up/SwiGLU/down/route-weight compatibility mode.
4. Preserve FP32 CPU/NPU routed partial accumulation and one final BF16 cast
   for the unquantized DSV2 path.
5. Port only the DSV2 logical/physical mapping boundaries still required by
   actual A3 failures (int32 boundary, physical expert dimension, valid
   placeholder/zero weight).
6. Close CPUInfer direct-sync state lifetime and callback exception safety
   without replacing upstream's ACL report worker.
7. Port regression tests, not historical diagnostic production knobs.

Every item is a separate semantic commit or a tightly coupled parent/child
pair. A patch is retained only if a raw failure or regression demonstrates its
need.

## Explicitly adopted upstream semantics

- ACL callback report worker and stream subscription;
- framework pinned D2H/H2D;
- explicit device wait before immediate CPU consumption;
- host-synchronous immediate CPUInfer submission on NPU;
- host drain before H2D;
- upstream ARM feature detection;
- upstream KTEP as the only coordinator.

## Explicitly rejected alternatives

### LOCAL_FIRST_SELECTIVE_PORT

Rejected because the upstream callback/stream architecture already subsumes
the local runtime design and passed raw primitives. Continuing from local would
retain a larger maintenance delta without stronger evidence.

### HYBRID_MINIMAL_PORT

Not needed as a distinct strategy. The selected upstream-first patch set keeps
validated model/numerical boundaries but does not retain a second runtime
coordinator.

### Merge all local history

Rejected because it would mix obsolete diagnostics, prior architecture, and
current upstream changes, destroying attribution.

## Consequences and gates

- Raw upstream remains documented as DSV2-incompatible.
- The aligned candidate must pass R2A/R2B/R2C before Round3 or P1.
- `B_pair=2.1875`, `K=32`, placements, GGUF, tokenizer and quality corpus stay
  frozen.
- Graph, Deferred, Dynamic, W8A8 and MXFP4 remain present but not qualified.
- Any stable flip, envelope overflow, same-path nondeterminism, ownership error
  or quality regression blocks alignment; thresholds are not refit.

```text
R5_G4_ALIGNMENT_ADR = APPROVED_UPSTREAM_FIRST
ROUND5_UPSTREAM_ALIGNMENT = A3_VERIFIED_READY
```
