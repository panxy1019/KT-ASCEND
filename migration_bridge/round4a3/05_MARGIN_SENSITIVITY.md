# Margin Sensitivity on Q

`epsilon_logit = 0.390625` (Q top-16 candidate max-error p99).

| C | Stable | Near tie | Near-tie ratio | Stable exact | Tie-set pass | Max tie set | size >5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 482 | 94 | 16.32% | 482/482 | 94/94 | 6 | 1 |
| 2 | 421 | 155 | 26.91% | 421/421 | 155/155 | 8 | 9 |
| 3 | 370 | 206 | 35.76% | 370/370 | 206/206 | 12 | 23 |
| 4 | 327 | 249 | 43.23% | 327/327 | 249/249 | 16 | 41 |

C=1 is selected because it is the smallest candidate with 100% stable exact
and 100% near-tie membership and materially limits tie-set expansion. Its
near-tie ratio exceeds the 10% review threshold and is explicitly carried as a
risk. Only one Q position (0.17%) has tie-set size greater than five, with size
six; this is treated as rare rather than frequent expansion.
