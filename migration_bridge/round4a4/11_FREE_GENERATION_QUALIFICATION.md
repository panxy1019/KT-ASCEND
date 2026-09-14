# Free-Generation Qualification

Evidence state: `A3_VERIFIED` after held-out pairwise qualification.

All requests used greedy decoding for 64 tokens. First-divergence positions
were classified using the same baseline history and frozen contract.

| Corpus | Requests | Divergences | Stable divergences | Ambiguous divergences | Membership |
|---|---:|---:|---:|---:|---:|
| H2 | 16 | 3 | 0 | 3 | 3/3 |
| F | 6 | 2 | 0 | 2 | 2/2 |

No first divergence occurred in the pairwise-stable region. Every divergent
Hybrid token belonged to its frozen baseline ambiguity set. Exact requests also
passed without special handling.

H2 analysis SHA256:
`a5db53b64d440e92c81656f9eefbe7ccb8ed07ac4128dfbe95eeaaa1b67d6cca`.
F analysis SHA256:
`5fd2580c3d5aaafb91f36cf2ed7ac2cb064ce151e37d55c32ae6ded3a0f53588`.

