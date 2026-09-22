# WP1 — resident-layout numerical probe

Evidence: `evidence/r65-5a-subset-v1/probe/npu_subset_equivalence.json`.
State: **A3_VERIFIED**, bounded diagnostic scope.

The deployed `UnquantizedFusedMoEMethod.forward_npu` was invoked directly with
real checkpoint BF16 weights, its actual format-cast routine and 64/32 physical
expert tensors. Inputs are the captured failing-token Hybrid MLP inputs.
Four layers × eight resident experts × two layouts × ten repeats were tested.
All 32 expert comparisons were exact; every within-layout repeat was exact and finite.
No new numerical tolerance was introduced.

Expert selection used four hot resident experts, two unused resident experts
nearest the rank-6/7 boundary, and two seeded random resident controls. Actual
rank-6/7 IDs are recorded separately: CPU-owned boundary experts cannot be tested
as resident experts in the unchanged R32 layout.

## Evidence boundary

This isolates grouped-kernel layout behavior using reconstructed checkpoint
weight tensors. It does not by itself prove the serving loader populated every
physical weight slice correctly, nor test every multi-token/group occupancy.
Existing mapping/shape evidence remains supporting evidence, not a substitute
for those untested cases. Mixed routing is examined in WP2.

The fresh runtime manifest matches all 91 prior manifest file hashes and all
recorded repository states. The additionally inspected deployed unquant.py SHA is
`56b46c170c8150afb437cb1bc0ea4453a456ca14f849170e7d0f27358cabcafb`.

Guard exited zero and reaped owned descendants. Its recorded cleanup signals
are retained verbatim; no unrelated process was targeted.

Artifact archive SHA256:
`a25356d35aee00a2cade13901dd360515f975a0903836a772a929d3155c6ca99`.

Historical cosine values slightly exceeding 1 are float32 metric-rounding
artifacts, not output differences; exact comparisons use tensors/hashes.
