# ADR: Round 4A Numerical Acceptance

## Status

`FINAL — BACKEND_OPTIONS_EXHAUSTED — REQUIRES_ACCEPTANCE_DECISION`. P1 remains
blocked and P2/P3 are not allowed under the frozen acceptance contract.

## Context

Round 4A1 requires exact P1 tokens and <=0.20 maximum absolute logprob delta
before P2/P3. Two correctable numerical mismatches were demonstrated and fixed:
missing BF16 expert boundaries and premature BF16 rounding of Hybrid partials.
P1 improved from 30/45 to 42/45 exact.

The remaining failure is one first-divergence history shared by three requested
lengths. Source weights, hidden inputs, routes and visible BF16 boundaries are
correct. Real expert comparisons and all 15 non-empty CPU-expert subsets show
sparse, input-dependent BF16 differences after the down GEMM. Some inputs for
the same expert are bitwise exact and others differ in only a few elements.
E6 or E25 can individually trigger the remaining near-tie token flip, while
E36 can either add error or cancel E25 at the decision boundary. This is
consistent with different ARM LLAMAFILE SGEMM and Ascend GMM reduction trees,
not a fixed bad expert or routing error.

## Decision

1. Keep the two minimal, independently verified Case A/B fixes.
2. Do not relax 45/45 or the 0.20 logprob gate.
3. Do not ship the rejected six-finalize/per-route experimental ABI.
4. Do not add a token-specific logit/tie-break patch.
5. Do not copy an Ascend kernel or restore a historical KML backend inside this
   numerical-closure round.
6. Stop before P2/P3 while the residual remains.

## Consequences

- P1 is operational and much closer numerically, but not accepted.
- P2/P3 remain untested rather than being contaminated by a known red base.
- A future explicitly authorized backend investigation may evaluate a pinned
  KML/BLAS or another CPU GEMM implementation in the disposable container. It
  must demonstrate the full 45/45 gate and preserve performance/ownership
  constraints; low stagewise rel-L2 alone is insufficient.

## Round 4A.2 backend investigation result

Round 4A.2 evaluated the current LLAMAFILE path, the non-buildable repository
KML references, OpenBLAS, BLIS, ATLAS, and Arm Compute Library NEGEMM using
identical captured operands and ten-repeat deterministic probes. BLIS, ATLAS,
and ACL failed the stagewise numerical/runtime gates. OpenBLAS alone restored
the original `v_struct_03` margin from `0` to the All-NPU value `+0.125` and
passed controlled local integration, but it changed a second exact All-NPU tie
for `v_en_01` from margin `0` to `-0.125`. Full P1 therefore remained 42/45,
with the repeated failure trajectory moving from `v_struct_03` to `v_en_01`.

The OpenBLAS adapter was removed and the final production diff is zero. This
closes the authorized maintainable backend search without relaxing any gate:

```text
ROUND4A2_CPU_BACKEND_NUMERICAL_INVESTIGATION = COMPLETE
BACKEND_OPTIONS_EXHAUSTED = TRUE
P1 = BLOCKED
P2 = NOT_RUN
P3 = NOT_RUN
```

## Rejected alternatives

- acceptance threshold relaxation: violates the task;
- per-token correction: non-general and hides root cause;
- six NPU finalize calls plus per-route CPU ABI: material complexity and no
  observed token benefit;
- executing CPU-owned experts on NPU: invalidates Hybrid placement;
- host KML installation: outside scope and violates environment isolation.
