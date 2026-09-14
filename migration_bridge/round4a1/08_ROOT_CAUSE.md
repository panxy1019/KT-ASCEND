# Root Cause

## 1. First observed failure

Frozen P1 produced 30/45 exact requests. The first inspected divergence was
`v_en_01` at decode step 1 (`185` all-NPU versus `549` Hybrid after the common
first token 185).

## 2. Reproducibility

The failure was deterministic, prefix-stable and finite. Exact staged inputs
and routes were reproducible on A3.

## 3. System-correctness evidence

Routing IDs/weights and Layer 17 inputs matched before divergence. Existing
0/1/2/multi-route, sequential/overlap and lifecycle tests excluded mapping,
buffer lifetime and stale-result faults.

## 4. First numerical stage where error grows

Original R0 diverged at gate/up outputs and accumulated approximately
0.003-0.005 relative L2 by down output. R2 BF16 boundaries reduced this by
orders of magnitude.

## 5. Exact dtype/accumulation differences

Two fixable differences were found:

1. LLAMAFILE exposed an all-FP32 gate/up-SwiGLU-down trajectory while Ascend
   exposes BF16 values after gate/up, fused SwiGLU and down GMM.
2. Hybrid independently rounded CPU and NPU partial sums to BF16 before adding,
   while all-NPU finalize accumulates BF16-valued route terms in FP32 and rounds
   once at the routed-output boundary.

Both were fixed minimally.

## 6. Expert contribution

In the remaining `v_struct_03` history, pass8 E36 contributes the largest
sparse term mismatch (95 BF16 elements versus one for E25). However, the
15-subset campaign proves that no expert is globally dominant: E6 or E25 alone
can flip token 9, E36 alone cannot, and `{25,36}` is exact because their margin
effects partially cancel. Other inputs for the same experts are bitwise exact,
so the residual is value- and direction-sensitive rather than a bad weight or
expert-ID mapping.

## 7. Why the token flips

Sparse Layer 17 differences are stored into later-layer KV history. At the
first token divergence, all-NPU favors token 8828 over 1273 by 0.125 reported
logprob, while Hybrid makes 1273, 8828 and 17570 tie. Greedy tie-breaking then
selects 1273 and divergent autoregressive history amplifies later deltas.

## Classification

- fixed portion: **Case A/B**, missing boundary and premature partial rounding;
- residual portion: **Case C**, ARM LLAMAFILE SGEMM and Ascend GMM
  use different reduction trajectories after all source-visible dtype
  semantics have been aligned.

The residual must not be hidden by relaxing the 45/45 gate. A new CPU backend,
copied accelerator kernel or token-specific tie-break is outside the authorized
minimal-fix scope.
