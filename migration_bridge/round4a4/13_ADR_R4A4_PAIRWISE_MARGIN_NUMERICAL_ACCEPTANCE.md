# ADR: Pairwise-Margin Heterogeneous Numerical Acceptance

Status: `APPROVED`  
Decision evidence: `HELDOUT_VERIFIED`, `QUALITY_VERIFIED`

## Context

The original Round 3/4A `45/45` cross-backend exact-token gate was intentionally
strict. It exposed real implementation defects: mismatched BF16-visible expert
boundaries and premature BF16 rounding of CPU/NPU partial sums. Round 4A.1
fixed both and improved P1 from 30/45 to 42/45 while preserving exact routing,
mapping, ownership, shared-expert and scaling semantics.

Round 4A.2 exhausted reasonable in-scope CPU GEMM candidates. OpenBLAS repaired
one LLAMAFILE near-tie trajectory but introduced a different near-tie trajectory;
both backends ended at 42/45. Backend replacement therefore was not a monotonic
cross-backend trajectory solution.

Round 4A.3 correctly remains `REJECTED`. Its held-out H passed stable-token,
tie-membership and numerical-envelope gates, but failed a frozen maximum
tie-set-cardinality rule. Retrofactively deleting or enlarging that hard gate
would invalidate the held-out protocol.

## Independent mechanism

Round 4A.4 asks whether the baseline top-1 ordering against relevant candidates
is stable under the measured backend-induced pairwise perturbation. For
All-NPU top-1 `a` and candidate `i`, it measures

`e_i = |(zH_a-zH_i) - (zN_a-zN_i)|`.

Cardinality of a global-epsilon window is not a floating-point correctness
invariant: many low-probability candidates can occupy a narrow BF16 logit band
without changing the serving token. Cardinality is therefore diagnostic only.

## Derivation and held-out result

Q2 comprised 18 prompts and 1,152 matched-history positions. Across 36,104
pairs, p99 distortion was 0.25 and the maximum was 1.75. The predeclared Q2-only
rule selected `B_pair = 1.25 * max = 2.1875`, retaining 639 stable and 513
ambiguous positions. Stable exactness and ambiguity membership were both 100%.

The candidate was frozen with SHA256
`accbbc15baf49d869b062dc2b71b4fe6f71a7fa8bd74f27120757d1c1f0e6627`
before H2 was inspected. H2 contained 16 new prompts and 1,024 positions. It
passed 654/654 stable exactness, 370/370 ambiguity membership, zero pairwise
overflow, and all-finite gates. H2 maximum distortion was 0.6875.

Free generation over H2 and six additional F prompts produced five first
divergences. All five were pairwise-ambiguous, all five Hybrid tokens were in
their frozen baseline ambiguity sets, and no stable-region divergence occurred.

Downstream quality used 128 frozen C-Eval and 128 frozen GSM8K multiple-choice
examples. C-Eval was identical at 57/128; GSM8K was 45/128 All-NPU and 47/128
Hybrid. Paired bootstrap intervals showed no statistically meaningful
regression, and invalid output counts were zero.

## Decision

Approve `PAIRWISE_HETEROGENEOUS_NUMERICAL_ACCEPTANCE` for this frozen scope:

- N0 system routing, mapping, ownership, shared expert, scaling, buffer, stream
  and callback semantics are exact;
- N1 same-path determinism and prefix consistency are exact;
- N2 CPU-not-hit control is exact;
- N3 CPU-vs-NPU expert relative-L2 is at most `1e-2`, with all values finite;
- N4 pairwise absolute distortion is at most `B_pair=2.1875` over the frozen
  top-32 union candidate universe;
- N5 every pairwise-stable top-1 is exact;
- N6 every ambiguous Hybrid top-1 belongs to the frozen baseline ambiguity set;
- N7 downstream quality has no statistically meaningful regression;
- ambiguity-set cardinality is diagnostic only.

The decision approves P1 requalification; it does not pre-approve P1, P2, or
P3. Each must pass its ordered system, numerical, coverage, stability, and
quality gates with this exact contract. P2/P3 may not refit `B_pair`, K, or the
classification rule.

## Boundaries and rollback

This contract is scoped to DeepSeek-V2-Lite revision 604d5664, TP1, BF16,
LLAMAFILE/CPUInfer, Ascend NPU0, Graph/Deferred/Dynamic/MTP/Speculative OFF, and
the frozen placements. It must be reopened for a model, weight, tokenizer,
backend, dtype, Top-K universe, or execution-mode change.

Immediately reject or roll back on any exact semantic failure, same-path
nondeterminism, CPU-not-hit mismatch, expert-envelope overflow, pairwise stable
flip, ambiguity escape, pairwise overflow, non-finite value, or statistically
meaningful quality regression. Placement-scale overflow in P2/P3 blocks that
placement; it does not authorize a wider contract.

