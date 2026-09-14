# Round 4A1 Summary

## Outcome

Round 4A1 found and fixed two real numerical compatibility defects, improving
P1 exact-token requests from 30/45 to 42/45. It also isolated the residual to
input-dependent GEMM reduction differences between ARM LLAMAFILE and Ascend
GMM. The acceptance gate is still red, so P2 and P3 were correctly not resumed.

## Verified fixes

1. Match Ascend-visible BF16 boundaries after gate/up, fused SwiGLU multiply,
   down output and route weight in both LLAMAFILE decode and prefill paths.
2. Keep CPU/NPU route partials in FP32 at the Hybrid boundary and perform one
   final BF16 cast after their addition.

## Evidence highlights

- exact production input/routes captured for real Layer 17 experts;
- source expert tensors are BF16; F32 GGUF contains BF16-valued weights;
- R2 reduces down error by orders of magnitude and can be bitwise exact;
- known `v_en_01` 64-token and `v_en_02` two-token cases became exact;
- final P1 result: 42/45, finite and prefix deterministic;
- remaining first divergence: `v_struct_03` token index 9;
- all-NPU margin 8828 over 1273: +0.125; Hybrid margin: 0;
- all 15 expert subsets completed: E6/E25 can flip independently, E8 is neutral,
  and E36 can cancel E25, so no single expert is a stable dominant cause;
- real captured route samples: E6=6, E8=4, E25=5 and E36=4;
- registered A3 CPU+NPU routing test passes after the final minimal diff.

## Gate statement

```text
P1 = BLOCKED (42/45, max post-divergence |delta logprob| 2.4375150725245476)
P2 = NOT RUN
P3 = NOT RUN
HOST MODIFICATION = NONE
```

WP1-WP8 numerical investigation and minimal-fix validation are complete. The
round ends at the mandatory Case C stop decision because the frozen P1 gate is
not satisfied; this is a completed investigation with a blocked product gate,
not permission to continue into P2/P3.
