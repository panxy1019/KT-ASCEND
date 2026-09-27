# Round12a execution contract

Round12a starts from the frozen Round12 result. It does not revise the Round12
QSA promotion or overwrite the immutable v167 failure.

The execution order is fixed:

1. Prove tied-parameter semantics offline in the frozen image.
2. Run one Ascend component process with tied parameters and tied buffers.
3. Run one full-model non-KT request with the promoted QSA source.
4. Freeze one identical request queue and run KT/non-KT/non-KT/KT in four
   independent model processes.
5. Compare quality, TTFT, E2E, startup, host memory, HBM, cleanup and cache
   retirement, then state whether the KT-style heterogeneous profile is worth
   using in the tested scope.

## Single source change

The base OffloaderV1 builds a device dictionary from every `state_dict()` key.
Qwen3.8 registers the same `A_log` parameter through both the GDN module and
its attention wrapper, so PyTorch rejects the two keys under its default tied
semantics.

The candidate calls `state_dict(keep_vars=True)`, keeps the first name for each
exact tensor object, copies that object once, and leaves
`functional_call(..., tie_weights=True)` at its default. It does not use
`tie_weights=False`: the frozen-image proof shows that option produces the same
numbers in a small read-only example but breaks parameter and buffer identity
inside `forward`.

No other OffloaderV1 scheduling, transfer, precision, stream or allocator
behavior changes.

## Component gate

The Ascend component must show:

- the exact candidate source hash;
- tied parameter and persistent-buffer state aliases;
- identity preserved inside every `functional_call`;
- bit-exact output and repeat hash against the pre-offload NPU reference;
- unique byte accounting, CPU parameter immutability and NPU buffer residency;
- allocator baseline, guard ACK, container removal and physical12 cleanup.

Failure keeps the model stage closed. One new component revision is available;
there is no unchanged automatic retry.

## First non-KT request

The first model process uses the same promoted QSA source and consumer as the
KT arm. It must load and bind all 48 layers, publish the explicit full-BF16
OffloaderV1 identity, reach server readiness, complete one 64-input/4-decode
request, produce the frozen token IDs with finite logprobs, and clean up.

Only after that request passes may the ABBA snapshots be submitted.

## ABBA workload and decision rule

The always-required row is 64 input tokens and 4 decode tokens, with one warmup
and three measurements per process. A 128-input/16-decode row is added to both
arms only if the first non-KT smoke finishes within 300 seconds and its measured
rate conservatively projects the larger row below the 1,800-second request
limit. The queue is frozen before the first ABBA process and cannot change
between arms.

The primary metric is profiler-off request E2E; client first-output TTFT is
secondary. Model startup is recorded separately. A result is
`WORTH_IT_FOR_QUALIFIED_SCOPE` only if both independent paired blocks have
KT/non-KT E2E at or below 0.80, outputs and termination contracts pass, and KT
does not regress the frozen resource contract. Ratios between 0.80 and 0.95 or
opposite paired directions yield `CONDITIONAL_OR_SAMPLE_UNCERTAIN`. Ratios at
or above 0.95, or a stable KT regression, yield `NOT_WORTH_IT_FOR_TESTED_SCOPE`.

This is a system-profile comparison. The KT arm uses C128/MXFP4 plus Static Q32
BF16/NZ29, while the non-KT arm uses full-BF16 layer offload. The report must
not describe the ratio as isolated framework overhead.

Every device process requires fresh 2×60-second admission. Evidence is archived
and hashed before the targeted project cache-retirement hook runs. No global
cache drop, other-device use, external-task modification, timer activation,
push, merge or tag is allowed.
