# B_pair Derivation

Evidence state: `ANALYTIC_DERIVED`.

`B_pair` was selected entirely inside Q2 before H2 was inspected. The
predeclared selection rule is the Q2 maximum pairwise distortion with a 1.25
safety reserve:

`B_pair = 1.75 * 1.25 = 2.1875`.

| Bound source | B_pair | Stable | Ambiguous | Stable exact | Membership | Overflow |
|---|---:|---:|---:|---:|---:|---:|
| p99 | 0.25 | 1,042 | 110 | 100% | 100% | 25 |
| p99.5 | 0.25 | 1,042 | 110 | 100% | 100% | 25 |
| p99.9 | 0.75 | 877 | 275 | 100% | 100% | 4 |
| max | 1.75 | 701 | 451 | 100% | 100% | 0 |
| max x 1.25, selected | 2.1875 | 639 | 513 | 100% | 100% | 0 |

The selected contract retains 55.47% of Q2 positions as provably stable.
Ambiguity cardinality is reported but is not a correctness gate. H2 did not
participate in the source, factor, candidate-K, or rule selection.

The frozen candidate contract SHA256 is
`accbbc15baf49d869b062dc2b71b4fe6f71a7fa8bd74f27120757d1c1f0e6627`.

