# Fix Design

## Accepted minimal changes

### F1: explicit LLAMAFILE BF16 numerical compatibility

`GeneralMOEConfig.bf16_numerical_compat` defaults false and is enabled only by
the LLAMAFILE wrapper. Decode and prefill round at the production Ascend-visible
gate/up, fused multiply, down and route-weight boundaries while retaining F32
SGEMM accumulation.

### F2: one final BF16 round across Hybrid partials

The LLAMAFILE CPU contribution is emitted as FP32. The Ascend method returns an
FP32 contribution only when the KTEP coordinator explicitly marks it. KTEP
adds CPU and NPU contributions in FP32, then casts once to the original hidden
dtype. Other quantization methods keep their existing contract.

### F3: default-off numerical capture

Arm-file gated diagnostics capture exact real inputs/stages. They do nothing
unless the corresponding environment variables are set and are not an
acceptance-path dependency.

## Rejected experiment

An experimental per-TopK contribution ABI and six-finalize reconstruction was
tested to reproduce route order. It improved some routed-output tensors but did
not change the remaining token divergence, added material runtime/ABI
complexity, and was removed before the final diff.

## Non-actions

- no threshold relaxation;
- no token/logit special case;
- no host or driver modification;
- no copied Ascend kernel;
- no restored historical KML backend;
- no P2/P3 execution while P1 is red.
