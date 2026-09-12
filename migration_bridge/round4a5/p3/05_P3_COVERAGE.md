# P3 Route Coverage

Evidence state: `A3_VERIFIED_READY`

The successful Q2/H2 production capture exercised every selected layer and
every selected CPU expert. All routing weights were finite.

| Layer | Selected / hit | CPU route hits |
|---:|---:|---:|
| 1 | 4/4 | 17,883 |
| 5 | 4/4 | 18,370 |
| 8 | 4/4 | 22,690 |
| 12 | 4/4 | 22,108 |
| 17 | 4/4 | 20,998 |
| 19 | 4/4 | 19,851 |
| 22 | 4/4 | 24,103 |
| 26 | 4/4 | 15,922 |

Totals: 8/8 layers, 32/32 experts (100%, exceeding the 75% gate), and 161,925
CPU route hits. Artifact `evidence/g4-route-coverage.json`, file SHA
`cde846e54958a3bc663f4a8d38d88d9ab0ec41fc7e0d30cef80a191fcc68828c`.

`P3_CPU_COVERAGE = A3_VERIFIED_READY`.
