# P3 Free-Generation Qualification

Evidence state: `A3_VERIFIED_READY`

Free greedy generation ran only after teacher-forced qualification passed and
used the frozen H2 and F corpora.

| Corpus | Requests | First divergences | Stable divergences | Ambiguous divergences | In-set |
|---|---:|---:|---:|---:|---:|
| H2 | 16 | 7 | 0 | 7 | 7/7 |
| F | 6 | 2 | 0 | 2 | 2/2 |

Every first divergence occurred in a `PAIRWISE_AMBIGUOUS` position and the P3
token belonged to the frozen baseline ambiguity set. No stable-region token
flip occurred. Ambiguity-set cardinality remained diagnostic only, as required
by the approved contract.

Evidence: `evidence/g6-h2-free-analysis.json` and
`evidence/g6-f-free-analysis.json`.

`P3_FREE_GENERATION = A3_VERIFIED_READY`.
