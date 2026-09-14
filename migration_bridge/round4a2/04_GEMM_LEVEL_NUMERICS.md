# GEMM-Level Numerics

All real-input results use 33 matched-history captured samples:
E6=15, E8=4, E25=9, E36=5. Each backend was repeated ten times.

| Backend | Gate max rel-L2 | Up max rel-L2 | Down median rel-L2 | Down max rel-L2 | Deterministic |
|---|---:|---:|---:|---:|---:|
| LLAMAFILE | baseline | baseline | 0 | 5.048414e-4 | yes |
| OpenBLAS 1T | 4.976403e-5 | 1.504919e-6 | 0 | 5.048410e-4 | yes |
| OpenBLAS 16T isolated | same envelope | same envelope | 0 | 5.048410e-4 | yes |
| BLIS 1T (12 samples) | 2.407882e-5 | 8.529755e-5 | 2.059320e-8 | 4.071028e-4 | yes |
| ATLAS 1T | 4.976403e-5 | 1.780107e-4 | 7.311742e-8 | 1.381836e-3 | yes |
| ACL NEGEMM 1T | 4.976403e-5 | 1.780107e-4 | 7.311742e-8 | 1.381836e-3 | yes |

BLIS 16T crashed with exit 139 on the first real sample and was rejected as
`BACKEND_RUNTIME`. BLIS 1T, ATLAS, and ACL did not beat the current backend on
the required representative/critical envelope and were rejected before
production integration.

OpenBLAS showed input-dependent behavior. It made the critical E25/pass8 down
output exact, but made E25/pass9 worse; E36/pass8 remained the dominant error.
Thus median rel-L2 alone was insufficient, and the full margin/P1 gates were
necessary.

BF16 bucket counts and elementwise max/mean/cosine/exact metrics are stored per
stage and sample in the JSON evidence. Synthetic diagnostics use the same
DeepSeek shapes and five deterministic patterns; real captures remain the
source of truth.

The synthetic matrix covered gate/up/down with random, small, large,
near-cancellation and sparse BF16-valued operands. All four candidates were
byte-deterministic across ten repeats. Maximum rel-L2 versus an FP64 dot product
rounded to BF16 was: OpenBLAS `8.572375e-9`, BLIS `7.022489e-5`, ATLAS
`4.503921e-6`, and ACL `4.503921e-6`. This diagnostic supports candidate
characterization but does not override the captured-input or P1 result.

Evidence: `evidence/round4a2-synthetic.json`.
