# Frozen-threshold Dynamic held-out routing protocol

Evidence: **LOCAL_VERIFIED** tooling, not A3 results. No running release-v6
file was replaced and no additional NPU workload was started.

`dynamic_holdout.py` accepts the separately frozen threshold and static profile,
requires the migration system probe to pass, then records 128 independently
frozen HOLDOUT prompts with 16 generated tokens. This is untimed diagnostic
evidence and cannot be used to select threshold or report formal throughput.

Every recorder forward is correlated to exact token history/position metadata.
Proven unused final-token forwards remain in raw evidence but are excluded from
metrics; missing/unknown/duplicate attribution is rejected. Every included
decode forward must contain all 26 layers and six routes per input row.

Full-device Dynamic prefill and Hybrid decode are different execution paths:
prefill ownership counts must not be labelled actual CPU execution. The primary
post-update exposure metrics therefore include **decode only**. They report NPU
assignment hit rate, CPU assignment rate, routed-weight mass and whole-model
token CPU-touch fraction.

A second lane applies the initial static mask to the same observed Dynamic
routes. It is explicitly a **counterfactual exposure diagnostic**, not a
replacement for measured Static-vs-Dynamic results. Those independent runs can
have different heterogeneous numerical trajectories and routes.

Actual update events are associated to nonoverlapping serial request time
windows, rather than inferred only from prompt length. Unknown/ambiguous events
are rejected. Report actual triggered requests, per-layer update calls, copied
expert slots, changed resident logical experts and migration bytes separately:
the official updater can copy all slots even when some identities do not change.

The current local suite has 26 passing tests. New checks reject incomplete or
duplicate decode layers, prevent prefill/unused-tail contamination and verify
actual migration event association. Tools remain undeployed pending the healthy
static phase and the original-binary CPU backing eviction regression.

Quality trigger-rate telemetry and production update component timing still
need separate integration; do not substitute expected trigger counts for
instrumented CMMLU trigger rate or diagnostic readback time for timed migration.
