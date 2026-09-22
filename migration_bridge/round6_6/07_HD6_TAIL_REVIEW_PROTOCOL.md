# HD6 tail review protocol — no refit

HD6 model capture has completed once. Before quality, review the observed tails under the pre-frozen diagnostic-only model-distribution contract. The original B_pair2.1875 comparison is **not** passed:3HD6 positions overflow, max2.65625. No historical stable flip or ambiguity escape. This does not change the old blocked result or create a new bound.

Select all three historical-overflow positions plus the maximum full-logit maxabs position, without deleting any:

| Prompt |Token index|Global layer call|Reason|
|---|---:|---:|---|
|hd6-03|37|233|max pairwise2.65625|
|hd6-04|45|306|pairwise2.5625|
|hd6-07|41|497|max full-logit abs7.25|
|hd6-08|6|527|pairwise2.25|

Replay every CPU-owned expert actually selected at these calls in layers1/9/17/26:52expert/input combinations. Use unchanged installed LLAMAFILE, actual GMM oracle and canonical CPU FP32;10repeats and original0.01local safety. This is a post-heldout diagnostic, **not** a replacement Phase-1 result, new fit, altered placement, or second model heldout run. Preserve its separate identity. New `--natural-only` only filters diagnostic pair selection; it does not alter production execution.

If a local gate fails, investigate and retain the failure before any model promotion. If it passes, that strengthens the bounded operator-safety interpretation but still does not prove downstream quality. The52count is derived from recorded route membership, not a selection of numerically favorable experts.
