# Upstream and Legacy Numerical Philosophy

Evidence state: **CODE_INSPECTED** and inherited Round4A1-Round4A3 evidence.

- Identical bytes, kernel and reduction arithmetic may use bitwise equality.
- Different blocking/reduction implementations require bounded numerical
  equivalence while retaining exact system semantics.
- CPU/NPU Hybrid implements the same model semantics through different
  numerical backends; it is not guaranteed to be a bitwise-identical
  implementation of All-NPU.
- Same-path Hybrid determinism remains exact.
- Numerical tolerance never covers routing, mapping, ownership, shared expert,
  scaling, buffer lifetime, stream or callback defects.

Round4A1 proved the value of the original exact-token gate by finding BF16
visible-boundary and partial-accumulation defects. Round4A2 showed that swapping
LLAMAFILE for an in-scope BLAS backend moved near-tie divergences rather than
closing them monotonically. Round4A3 then correctly rejected a global-epsilon
contract whose cardinality hard gate did not generalize. Round4A4 retains all
exact semantic gates and tests pairwise order stability as an independent
mechanism.
