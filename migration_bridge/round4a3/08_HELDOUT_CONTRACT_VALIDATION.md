# Held-Out Contract Validation

Evidence state: **HELDOUT_VERIFIED FAILURE**.

## Gates that passed

- stable positions: 689; exact Hybrid/All-NPU top-1: 689/689
- near-tie positions: 79; Hybrid top-1 in frozen tie set: 79/79
- near-tie ratio: 10.29%
- candidate error max: 0.6875 <= 0.75
- full-logit max abs: 1.5 <= 6.9375
- relative L2 max: 0.0387082 <= 0.1151437
- all finite: yes

## Blocking gate

The frozen Q contract permits a maximum tie-set size of six. H contains three
positions with a top-16-truncated tie set of size 16:

- `h_math_01`, token indices 11, 36, and 62
- baseline margin: 0 at all three
- baseline and Hybrid serving token: 64 at all three

The token choices agree, but the contract has lost the required bounded tie-set
discrimination at these positions. Classification:
`TIE_SET_TOO_LARGE / HELDOUT_CONTRACT_FAILURE`.

This is not repaired with H-derived epsilon/C changes and the H samples are not
removed. No H2 is proposed because there is no independent new mechanism
hypothesis; replacing a failing held-out trajectory would be result-driven
corpus selection.
