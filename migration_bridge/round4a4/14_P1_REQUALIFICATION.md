# P1 Formal Requalification

Status: `A3_VERIFIED_READY`

Frozen placement: Layer 17 CPU experts `{6,8,25,36}`, physical NPU expert
count `60`, placement SHA256
`9548bf6e06014e034c6a6650af3a891a546f70d8c8d79b059174ba571c44471f`.

## System and lifecycle

- C0 NPU-only: exact against All-NPU and explicit reference;
- C1 one CPU route: exact against explicit `ΣCPU+ΣNPU`;
- C2 two CPU routes: exact against explicit reference;
- C3 CPU routes in non-leading Top-K positions: exact;
- C4 four CPU routes in a multi-token batch: exact;
- logical-to-physical mapping dtype: `torch.int32`;
- shared expert and routed scaling: exactly once;
- sequential routed/final SHA equals overlap routed/final SHA;
- overlap 1,000-forward lifecycle: one unique routed hash;
- CPU-not-hit control: exact.

The shared routed SHA is
`878e638f32c3311f816f3dc01f6e83dcd822c4f332083a591675d3ccaaafaba2`;
the final shared/scaled SHA is
`2f25017046177a407220bae1cf240e8451e57bda34a4461347b4615329b4db17`.

## Numerical and quality contract

The maximum controlled-case Hybrid-vs-All-NPU expert relative-L2 was
`0.0035398305`, below `1e-2`. Q2 and H2 passed the frozen pairwise contract,
H2/F free-generation had no stable-region divergence, and the 128+128 quality
A/B showed no statistically meaningful regression. Same-path 10-repeat and
clean-restart prefix evidence is recorded in WP0.

All N0–N7 contract gates pass. Therefore:

`P1_REQUALIFIED = A3_VERIFIED_READY`  
`MULTI_EXPERT_SINGLE_LAYER = A3_VERIFIED_READY`

