# Candidate Selection

| Backend | Gate rel-L2 | Up rel-L2 | Down rel-L2 | First-divergence margin | Deterministic | Full P1 |
|---|---:|---:|---:|---:|---:|---:|
| LLAMAFILE | baseline | baseline | median 0, max 5.048414e-4 | 0.000 | pass | 42/45 |
| OpenBLAS | median 0, max 4.976403e-5 | median 0, max 1.504919e-6 | median 0, max 5.048410e-4 | +0.125 | pass | 42/45 |
| BLIS 1T | median 0, max 2.407882e-5 | median 0, max 8.529755e-5 | median 2.059320e-8, max 4.071028e-4 (12 rows) | not run: stage gate fail | pass | not run |
| ATLAS | median 0, max 4.976403e-5 | median 0, max 1.780107e-4 | median 7.311742e-8, max 1.381836e-3 | not run: stage gate fail | pass | not run |
| ACL NEGEMM | median 0, max 4.976403e-5 | median 0, max 1.780107e-4 | median 7.311742e-8, max 1.381836e-3 | not run: stage gate fail | pass | not run |

OpenBLAS was selected for the one allowed production-path experiment because it
alone passed the original critical margin gate. It was rejected after full P1.
No candidate is selected for shipping.

Final classification:

`ROUND4A2_CPU_BACKEND_NUMERICAL_INVESTIGATION = COMPLETE`

`BACKEND_OPTIONS_EXHAUSTED = TRUE`
