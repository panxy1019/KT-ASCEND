# Q2 Contract Candidate

Evidence state: `ANALYTIC_DERIVED` and candidate frozen before held-out use.

The candidate retains exact system semantics, exact same-path determinism,
exact CPU-not-hit control, finite outputs, and the unchanged expert
`relative-L2 <= 1e-2` safety gate. Its numerical mechanism is:

- candidate universe: baseline top-32 union Hybrid top-32;
- pairwise distortion bound: `B_pair = 2.1875`;
- stable ordering: Hybrid top-1 must equal All-NPU top-1, 100%;
- ambiguous ordering: Hybrid top-1 must be in the baseline ambiguity set, 100%;
- ambiguity-set cardinality: diagnostic only.

Q2 result: stable `639/639`, ambiguity membership `513/513`, overflow `0`,
all finite. Candidate contract SHA256:
`accbbc15baf49d869b062dc2b71b4fe6f71a7fa8bd74f27120757d1c1f0e6627`.

