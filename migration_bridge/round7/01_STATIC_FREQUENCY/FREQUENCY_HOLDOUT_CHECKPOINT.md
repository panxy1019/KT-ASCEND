# Static Frequency held-out routing checkpoint

Evidence state: **A3_VERIFIED** routing, attribution and owned-process reaping.
Not a completed product/performance qualification or scheduler selection.
The unchanged release-v6 serial driver has progressed to Frequency CMMLU.

Both profiles use the frozen HOLDOUT128 corpus,16 generated tokens/request,
26 real MoE layers, global resident budget208, CPU64=2x32 subpools, TP1/batch1,
Dynamic/Deferred/Graph OFF. Frequency was derived only from DEV and committed
before HOLDOUT at `391b332ff0dfe648cec0bb4113b9daeabb8571e9`.
Frequency per-layer budgets range4..12; they were not refitted on these results.

| Routing metric | Uniform | Frequency |
|---|---:|---:|
| Included model-token rows | 128,032 | 128,032 |
| Routed assignments | 19,972,992 | 19,972,992 |
| NPU-hit assignments | 2,484,750 | 5,245,941 |
| CPU-hit assignments | 17,488,242 | 14,727,051 |
| NPU assignment hit rate | 12.4405497183% | 26.2651734903% |
| CPU assignment route rate | 87.5594502817% | 73.7348265097% |
| CPU routed-weight mass | 87.0509984615% | 74.3861508677% |
| Whole-model token rows touching CPU | 100% | 100% |

Derived comparison: NPU assignment hit rate +13.8246237719pp;
CPU assignment count -15.7888425835% relative;
CPU routed-weight mass fraction -14.5487677542% relative.
These are actual independently generated static paths, not matched-history
numerical equality. Total routing-weight sums also differ slightly between
paths; each mass uses its own measured denominator. No inference about quality
or decode throughput follows from routing improvement alone.

Both captures have raw2176/attributed2176 forwards, included2048,
proven unused final-token forwards128, unknown0. Original tails remain in the
archive and are excluded from routing denominators. Every included forward has
all26 layers and six assignments per token/layer. Frequency's summed per-layer
fully-NPU rows4472 are not unique model tokens: fully-NPU whole-model rows0.
The separate Dynamic analysis must exclude full-device prefill from actual
Hybrid decode CPU exposure.

Frequency observer-ON request median12.412052154541016 seconds is diagnostic
only. Uniform observer-ON median12.734819650650024 seconds is likewise not a
formal throughput result. Observer-OFF performance is still pending.

## Lossless evidence freeze

Archive: `../raw/frequency-holdout-evidence-v1.tar.gz`.
SHA256: `9e7dddc576c8cf409c41324015dc47d9b4a275dc5ca98c270cb11e7140b1a1ef`.
Raw request SHA256:
`0e70678c2fd9f0847dc6e4b305a303dba8d400e36bfcd280a99dfa376f90c30a`.

Preserves original recorder bytes, request/forward/routing logs, structure,
resolved command, preflight, stdout and completed lifecycle/reaping. Remote tar
completed before SCP; transfer exited0 before local SHA and full archive-read
validation. Local and A3 archive SHA match. Frequency lifecycle=COMPLETE and
reaping.complete=true; this is a closed capture, not a snapshot of active files.

Uniform comparison source/archive is frozen separately in
`UNIFORM_HOLDOUT_CHECKPOINT.md`.

Next ordered gates: complete Frequency CMMLU256 under the unchanged >=162/256,
invalid0 rule; if accepted, finish observer-OFF static ABBA before deciding the
best static baseline. Dynamic migration and Deferred remain not started.
