# P3 Frozen Pairwise Numerical Qualification

Evidence state: `A3_VERIFIED_READY`

The Round 4A.4 contract was applied unchanged:

```text
B_pair = 2.1875
candidate K = 32
formal contract canonical SHA = 223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717
validator payload SHA = accbbc15baf49d869b062dc2b71b4fe6f71a7fa8bd74f27120757d1c1f0e6627
```

No threshold, K, ambiguity rule, Q2 subset, or H2 subset was refitted.

| Corpus | Positions | Pairs | Stable exact | Ambiguous membership | Overflow | Max distortion | Finite |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q2 subset | 256 | 8,052 | 137/137 | 119/119 | 0 | 1.9375 | yes |
| H2 subset | 256 | 8,040 | 155/155 | 101/101 | 0 | 0.5625 | yes |

Pairwise absolute-distortion distributions:

| Corpus | p50 | p90 | p95 | p99 | p99.5 | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q2 P3 | 0.0625 | 0.1875 | 0.25 | 0.5 | 0.625 | 0.9968 | 1.9375 |
| H2 P3 | 0.0625 | 0.125 | 0.1875 | 0.3125 | 0.375 | 0.5 | 0.5625 |

Scale telemetry shows a wider Q2 tail than P1 and a Q2 maximum below P2's
2.125 maximum; H2 P3 is narrower than the frozen P2 H2 result (1.875 maximum).
There is therefore no systematic bound overflow at eight layers.

Evidence: `g3-q2-contract.json` and `g3-h2-contract.json`. Raw metrics remain
in the frozen A3 artifact root with file SHAs `86f3314b...` (Q2) and
`dab784077...` (H2).

`P3_PAIRWISE_NUMERICAL = A3_VERIFIED_READY`.
