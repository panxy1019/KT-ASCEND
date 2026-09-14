# Expert Sensitivity Decomposition

## Method

The frozen P1 Layer 17 CPU set `{6,8,25,36}` was decomposed into all 15
non-empty subsets: four singles, six pairs, four triples and the full set. Each
subset was run after a cold restart of the A3 disposable container, with the
same `v_struct_03` prompt, greedy decoding and 16 generated tokens. Only the
selected subset was CPU-owned; all other Layer 17 experts remained on NPU.

The comparison stops at the first autoregressive token mismatch. Routed-output
metrics therefore include only passes with bitwise-identical Layer 17 inputs,
router IDs and router weights. `CPU hit count` counts selected-expert route hits
over all captured tokens/passes, not unique experts.

## Results

| CPU experts | CPU hits | Routed rel-L2 | Routed max abs | Exact routed passes | Max selected-token logprob delta | First divergent token |
|---|---:|---:|---:|---:|---:|---:|
| `{6}` | 13 | 1.3191e-05 | 4.8828e-04 | 8/10 | 0.06962 | 9 (`8828 -> 1273`) |
| `{8}` | 2 | 0 | 0 | 11/11 | 0 | none |
| `{25}` | 8 | 1.7224e-05 | 7.3242e-04 | 8/10 | 0.08269 | 9 (`8828 -> 1273`) |
| `{36}` | 5 | 2.5521e-05 | 9.7656e-04 | 10/11 | 0.05905 | none |
| `{6,8}` | 15 | 1.3191e-05 | 4.8828e-04 | 8/10 | 0.06962 | 9 (`8828 -> 1273`) |
| `{6,25}` | 21 | 2.1695e-05 | 7.3242e-04 | 6/10 | 0.08433 | 9 (`8828 -> 1273`) |
| `{6,36}` | 18 | 2.9235e-05 | 9.7656e-04 | 7/10 | 0.08333 | 9 (`8828 -> 1273`) |
| `{8,25}` | 10 | 1.7224e-05 | 7.3242e-04 | 8/10 | 0.08269 | 9 (`8828 -> 1273`) |
| `{8,36}` | 7 | 2.5521e-05 | 9.7656e-04 | 10/11 | 0.05905 | none |
| `{25,36}` | 13 | 3.0581e-05 | 9.7656e-04 | 9/11 | 0.03290 | none |
| `{6,8,25}` | 23 | 2.1695e-05 | 7.3242e-04 | 6/10 | 0.08433 | 9 (`8828 -> 1273`) |
| `{6,8,36}` | 20 | 2.9235e-05 | 9.7656e-04 | 7/10 | 0.08333 | 9 (`8828 -> 1273`) |
| `{6,25,36}` | 26 | 3.3932e-05 | 9.7656e-04 | 6/10 | 0.03413 | 9 (`8828 -> 1273`) |
| `{8,25,36}` | 15 | 3.0581e-05 | 9.7656e-04 | 9/11 | 0.03290 | none |
| `{6,8,25,36}` | 28 | 3.3932e-05 | 9.7656e-04 | 6/10 | 0.03413 | 9 (`8828 -> 1273`) |

Machine-readable evidence is
`evidence/round4a1-expert-subset-sensitivity.json`.

## Interpretation

The routed-output error generally grows as more non-exact CPU routes are
included, but token sensitivity is not monotonic:

- E8 is exact for every observed same-history route and is neutral in every
  combination.
- E6 and E25 can each independently trigger the index-9 token flip.
- E36 produces the largest single-expert routed max-abs error but does not
  independently flip the token.
- `{25}` diverges while `{25,36}` does not. The E36 contribution partially
  cancels the E25-induced candidate-margin shift.
- Adding E6 to `{25,36}` reintroduces the same token flip. Thus no single expert
  is a stable dominant cause across combinations.

The tensor-level behavior is compatible with additive routed contributions,
while the final greedy decision is nonlinear around a near tie. CPU hit count
and routed rel-L2 alone cannot predict token equivalence; sign, direction and
downstream propagation of each sparse error matter.

## Scale and input dependence

The same expert can be bitwise exact for one hidden state and non-exact for
another. Across independent real captures the observed route sample counts are
E6=6, E8=4, E25=5 and E36=4, satisfying the requested minimum of three samples
per expert. The residual therefore tracks hidden-state values and backend GEMM
reduction order, not a permanently bad expert ID or weight tensor.

## Measurement limitation

The SGLang generation API exposes selected-token and top-k logprobs, not full
layer-output or final-logits tensors. Consequently this campaign reports the
available selected-token maximum logprob delta and does not invent layer-output
or full-logits rel-L2 values. Layer 17 routed outputs were captured directly;
first-divergence candidate margins and pass-level expert terms are documented
in `07_FIRST_DIVERGENCE_ATTRIBUTION.md`.
