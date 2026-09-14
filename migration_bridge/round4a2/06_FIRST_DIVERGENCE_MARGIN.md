# First-Divergence Margin

## Frozen residual: v_struct_03, token 9

Identical token history was used.

| Backend | margin `(8828 - 1273)` | greedy token |
|---|---:|---:|
| All-NPU | +0.125 | 8828 |
| LLAMAFILE Hybrid | 0.000 | 1273 |
| OpenBLAS Hybrid | +0.125 | 8828 |

OpenBLAS exactly restored the known margin and the 16-token All-NPU sequence.

## New P1 residual: v_en_01, token 10

The first ten generated tokens are identical across runs.

| Backend | margin `(30 - 279)` | greedy token |
|---|---:|---:|
| All-NPU | 0.000 | 30 |
| LLAMAFILE Hybrid | 0.000 | 30 |
| OpenBLAS Hybrid | -0.125 | 279 |

This establishes the failure mechanism without using post-divergence values:
OpenBLAS closes one near-tie but perturbs a different exact tie by one 0.125
logit bucket. There is no globally compatible improvement over the corpus.

Evidence: `evidence/round4a2-openblas-margin-response.json`,
`evidence/round4a2-allnpu-ven01-top20.json`,
`evidence/round4a2-llamafile-ven01-top20.json`, and
`evidence/round4a2-openblas-ven01-top20.json`.
