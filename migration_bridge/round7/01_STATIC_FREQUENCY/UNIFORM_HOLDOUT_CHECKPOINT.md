# Uniform R8-C64 held-out routing checkpoint

Evidence: **A3_VERIFIED** routing/integrity, not formal performance or completed
Static Frequency selection. Model, masks, CPU64 definition and software profile
are the frozen release-v6 configuration.

All 128 independent HOLDOUT prompts completed with 16 generated tokens.
Recorder attribution: raw2176/attributed2176, included2048, proven unused
final-token forwards128, unknown0. Raw tails are retained, not fitted or included
in the route denominator. Every included forward has all26 MoE layers and six
routed assignments per token/layer.

| Metric | Result |
|---|---:|
| Included model token rows (prefill + decode inputs) | 128,032 |
| Routed assignments | 19,972,992 |
| NPU-hit assignments | 2,484,750 |
| CPU-hit assignments | 17,488,242 |
| NPU assignment hit rate | 12.4405497183% |
| CPU assignment route rate | 87.5594502817% |
| CPU routed-weight mass fraction | 87.0509984615% |
| Whole-model token rows touching CPU | 100% |

These are static Hybrid routing metrics, including actual static prefill. Later
Dynamic full-device prefill must not be conflated with Hybrid decode exposure.
The whole-model token fraction is not a summed per-layer row fraction.

The observer-ON request median was12.734819650650024 seconds. It is preserved
only as diagnostic context: **not formal or observer-OFF performance**.

Raw request SHA256:
`908e32552297df165137f5b8a8e9c3cfc5c958b4bec1ba3474239df1d59477f8`.
Lossless archive `../raw/uniform-holdout-evidence-v1.tar.gz` preserves summary,
original recorder, request/forward/routing logs, resolved command, preflight,
process stdout and completed lifecycle/reaping. Archive SHA256:
`835a2dcc59d8fc93b997cce49d6a62905f4e83fb67616c09819a24215deca78e`.

The healthy serial driver has advanced to Frequency HOLDOUT automatically.
No Frequency quality/performance or Dynamic/Deferred gate is closed by these
Uniform results. Selection remains pending independent Frequency measurements.
