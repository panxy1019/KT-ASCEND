# Round7 performance decision audit

Evidence: **LOCAL_VERIFIED**, not an A3 result. Frozen before any Dynamic
threshold, held-out or performance output is used for scheduler selection.
Current healthy Static release-v6 is unchanged.

`tools/evaluate_performance.py` recomputes metrics from all36 raw ABBA cells:
short256decode, medium512decode, long1024decode; three blocks of left/right/right/left.
It refuses missing/duplicate cells, wrong order, mixed modes, changed workload,
wrong generated-token budget, nonfinite/nonpositive latency/rate, or observed
and nonserial summaries. Both sides must have every decode rate available to
compare median decode throughput; a subset of uncoalesced SSE runs is not used.

The user-proposed Dynamic default screen is >=3% median E2E latency reduction
OR >=3% median decode throughput increase on at least2/3 workloads. As a
conservative application of the no->5% stable-workload-regression requirement,
any >5% median E2E latency increase or >5% median available decode throughput
decrease disqualifies default enablement. Per-block E2E gains and positive-block
counts remain explicit for repeatability review; isolated averages do not
replace that review or independently establish PRODUCT_READY.

Every measured cell also needs a matching, complete host-isolation window.
Missing/contended/stale observation returns PENDING, not NO_BENEFIT. Resource
window evidence is independent of raw timing, and a single EXCLUSIVE sample is
insufficient. Current v6 timing can be independently post-audited against the
host observer; future releases produce windows directly.

This evaluator cannot close system semantics, I16 generation/weight identity,
quality, no-trigger/trigger/restart/workload-shift, or final scheduler selection.
It does not impose the Dynamic3% default gate on Static Frequency or Deferred:
their comparisons remain independently assessed under the task's own rules.
It does not modify acceptance thresholds based on results, nor measure internal
CPU wait, exact device prefill time or direct host bandwidth.

Nine new local negative/control tests pass; full pure-Python suite43/43.
New evaluator is not in executing v6 or staged v7. Include it only in a separately
versioned reviewed future tools release. No production or native binary change.
