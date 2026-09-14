# H2 Held-out Pairwise Validation

Evidence state: `HELDOUT_VERIFIED`.

H2 was frozen as 16 completely new prompts before the Q2 candidate was applied.
It did not participate in `B_pair`, safety-factor, top-K, or rule selection.

| Corpus | Positions | Stable | Stable exact | Ambiguous | Membership | Overflow | Contract |
|---|---:|---:|---:|---:|---:|---:|---|
| Q2 | 1,152 | 639 | 639/639 | 513 | 513/513 | 0 | QUALIFIED |
| H2 | 1,024 | 654 | 654/654 | 370 | 370/370 | 0 | HELDOUT_VERIFIED |

H2 maximum pairwise distortion was `0.6875`, below frozen
`B_pair=2.1875`; all values were finite. Ambiguity diagnostics were size p95
`13`, size max `32`, probability-mass p95 `0.984297`, and mass max `0.999716`.
Those values are deliberately diagnostic: Round 4A.4 does not restore the
Round 4A.3 cardinality hard gate.

Validation evidence SHA256:
`5e384653f347e7ac765458610ca0df167f82c2cdd2b17279e87b66e92b071a89`.

