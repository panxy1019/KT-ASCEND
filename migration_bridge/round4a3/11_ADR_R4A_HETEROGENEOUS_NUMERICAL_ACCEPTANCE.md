# ADR: Round 4A Heterogeneous Numerical Acceptance

## Decision

`HETEROGENEOUS_NUMERICAL_ACCEPTANCE = REJECTED`

ADR option: **B — REJECT CONTRACT**.

## Context

The original 45/45 cross-backend token gate was valuable: it exposed incorrect
BF16-visible boundaries and early BF16 rounding of CPU/NPU partials. Round 4A.1
fixed those real numerical-semantic defects and improved P1 from 30/45 to
42/45. Round 4A.2 then showed that replacing LLAMAFILE with OpenBLAS merely
moved a near-tie failure from `v_struct_03` to `v_en_01`; it did not create a
stable corpus-wide solution.

Different reduction/blocking orders can be deterministic and numerically close
without being bitwise equal. That principle does not relax system semantics:
routing, mapping, ownership, scaling, shared expert, buffer lifetime, streams,
and same-path determinism remain exact gates.

## Candidate derived from Q

Q produced epsilon `0.390625` and C=1. Stable top-1 was exact for 482/482
positions; all 94 near ties selected a token in the baseline tie set. The Q
candidate bounded the tie set at six and was frozen before H.

## Held-out result

H generalized the numerical error and token-membership portions:

- stable exact 689/689;
- near-tie membership 79/79;
- no expert/logit envelope overflow in the recorded metrics;
- all finite.

It did not generalize the bounded tie-set requirement. Three `h_math_01`
positions had zero margin and at least 16 candidates inside the frozen window.
This exceeds the Q-frozen maximum six and makes the near-tie rule insufficiently
discriminating.

## Consequences

- `NUMERICAL_ACCEPTANCE_CONTRACT = REJECTED`
- no approved replacement for the original strict cross-backend token gate
- P1 is not requalified
- P2/P3 remain forbidden and unrun
- free generation and downstream quality qualification are not used to override
  the held-out numerical blocker
- production code remains unchanged

## Reopen conditions

Reopen only with a mechanism-level hypothesis defined independently of H, a
new Q derivation protocol, and a newly frozen H2. It is not sufficient to widen
epsilon/C, permit arbitrary ties, remove `h_math_01`, or replace held-out
samples because they failed.
