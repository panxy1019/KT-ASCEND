# WP5/WP6 — causal interventions, not production fixes

Both experiments keep the original four-R32 mask, ownership, TP1, batch1,
32 workers and token history. Intervention code lives only under diagnostic
tools and is explicitly injected into its own serving process.

## B — baseline routes on evolving Hybrid hidden states

The debug hook replaces IDs and route weights with the captured all-NPU values
at each of 26 layers ×65 forwards (1690 interventions). It does not replace
hidden states. This intervenes on both expert choice and weights, not solely
on one Top-K discontinuity.

| token54 metric | Original Hybrid | Baseline-route counterfactual |
|---|---:|---:|
| Layer17 MLP output relative-L2 | 0.2177792 | 0.03291165 |
| Final-logit relative-L2 | 0.5690376 | 0.13488084 |
| Maximum pairwise distortion | 9.59375 | 16.65625 |
| Overflow candidate pairs | 27 | 5 |

Routing differences contribute substantially to the layerwise and overall L2
amplification, but are not the only residual source. Worst pairwise distortion
does not improve. This experiment **fails the frozen envelope** and cannot be
used as a fix or promotion result.

## NPU-for-CPU diagnostic

The original CPU task still submits and finishes. Immediately before the
original framework-managed H2D, the debug hook substitutes the CPU-owned
partial's numerical value with a same-input Ascend grouped-MoE reference.
The original R32 resident tensors, ownership, remapping and merge remain.
No production file is changed; extra diagnostic NPU weights/operations are
not an approved CPU execution strategy or performance measurement.

- 4 layers ×65 forwards =260 interventions, exact coverage.
- All 26 layer structures still match the frozen profile.
- All 64 full 102400-vocabulary logits vectors are exact versus the original
  natural all-NPU oracle; max absolute error 0.
- Pairwise distortion 0; overflow 0; no router-set divergence at token54.
- Across those interventions, the original CPU partial versus same-input GMM
  partial has maximum relative-L2 approximately 0.00109680. This aggregate
  partial metric is distinct from the per-expert safety metric.

This removes the observed failure while preserving its R32 ownership and
merge framework. It identifies the CPU/NPU arithmetic change as the effective
perturbation source for this fixture, without proving an instruction-level
reduction-tree cause or universal absence of bugs.

The first offline summary attempt had an unnecessary serving-module import.
Its failure log remains preserved. A CPU-only metrics helper and the completed
`counterfactual_replay_v2.json` replace that summary, not the model runs.
