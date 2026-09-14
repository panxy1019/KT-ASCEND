# Independent Mechanism Hypothesis

Evidence state: **ANALYTIC_DERIVED** (hypothesis pending Q2/H2 evidence).

Round4A3 was correctly rejected because its frozen cardinality hard gate did
not generalize to held-out H. The observed H positions did not contain a stable
token flip, tie-set escape, expert-envelope overflow, non-finite value or
serving-token disagreement. This result remains rejected and is not relabeled.

Round4A4 tests a distinct mechanism. For All-NPU baseline top-1 token `a` and
candidate `i`, define:

```text
mN_i = zN_a - zN_i
mH_i = zH_a - zH_i
d_i  = mH_i - mN_i
e_i  = |d_i|
```

The new hypothesis is that heterogeneous greedy correctness depends on whether
the sign/order of each relevant pairwise margin is stable under a Q2-derived
bound `B_pair`. The number of candidates within that bound is diagnostic only;
it is not a floating-point correctness invariant.

The relevant candidate universe is frozen to:

```text
U_t = baseline serving top-32 union Hybrid serving top-32
```

This union prevents a Hybrid-emergent candidate from being hidden by a
baseline-only top-K truncation. Stable positions require exact top-1 identity.
Ambiguous positions require the Hybrid top-1 to remain in the baseline
ambiguity set. H2 is completely new and cannot modify the bound, K, or rule.
