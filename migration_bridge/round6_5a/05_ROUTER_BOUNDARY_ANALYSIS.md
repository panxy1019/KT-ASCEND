# WP4 — same-operator router replay

State: A3_VERIFIED. Target report:
`/workspace/results/r65-5a-causal-v1/router_boundary.json`.

The deployed `linear_bf16_fp32` dispatch and Ascend gating Top-K operator were
replayed on the baseline and Hybrid captured hidden states with the same gate
weight. Both layers' logits and Top-K IDs exactly reproduce their captures.
No path-dependent router implementation was needed to reproduce the flip.

| Layer17 quantity | All-NPU hidden | Hybrid hidden |
|---|---:|---:|
| E62 logit | 0.6453612447 | 0.5582630634 |
| E29 logit | 0.6031817198 | 0.6205210686 |
| E62−E29 | +0.0421795249 | −0.0622580051 |
| E62 rank | 6 | 8 |
| E29 rank | 8 | 6 |
| Actual rank6/7 | E62/E27 | E29/E27 |
| Actual rank6/7 margin | 0.0251885056 | 0.0290796757 |

Pair-score perturbation is −0.1044375300; Top-K set Jaccard is 5/7.
Thus it would be incorrect to call E29 the baseline rank7 expert.

Layer12 baseline rank6/7 margin is 0.0022378862, increasing to 0.0098144710;
its Top-K set remains unchanged. A small baseline margin alone is insufficient
to predict a flip without the actual perturbation direction.

This establishes that the observed hidden-state change explains the router
change. It does not yet establish the upstream cause of that hidden-state
change or grant acceptance to the resulting final-logit distortion.
