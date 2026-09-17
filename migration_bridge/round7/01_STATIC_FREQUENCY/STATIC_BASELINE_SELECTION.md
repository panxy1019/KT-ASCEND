# Round7A Static baseline selection

Date: 2026-09-17 UTC. Evidence: A3_VERIFIED system/quality and routing;
ANALYTIC_DERIVED performance post-audit. This is not Round7 completion.

## Decision

`ROUND7A_STATIC_FREQUENCY = COMPLETE`

`STATIC_BASELINE = UNIFORM_R8_C64`

Frequency remains a frozen, system-safe, quality-passing candidate, but is not
selected as the Dynamic control. No expert mask was refitted from HOLDOUT or
CMMLU. Dynamic and Deferred remain unqualified and default OFF.

## Closed gates

DEV256 fitting and commit freeze, representative local operator safety,
Uniform/Frequency compact repeat/prefix/restart/observer parity, both HOLDOUT128
route attribution audits, and Frequency CMMLU256 are preserved in their existing
checkpoint documents and lossless archives. Frequency CMMLU is173/256 versus
frozen R64=174/256, invalid0, observed delta-0.390625pp: PASS under the unchanged
-5pp policy. CMMLU is a guardrail, not a mask-selection objective.

The observer-OFF3ABBA campaign completed all36 measured requests with correct
token budgets256/512/1024 and closed/reaped per-measurement lifecycles. Original
A3 file mtimes were audited before archiving.

## Complete exploratory performance

| Workload | Uniform median E2E seconds | Frequency median E2E seconds | Frequency latency improvement | Uniform decode tok/s | Frequency decode tok/s |
|---|---:|---:|---:|---:|---:|
| Short |131.059813|130.446878|0.467675%|1.975081|1.985218|
| Medium |260.034589|259.316453|0.276169%|1.988162|1.991722|
| Long |539.608696|532.871286|1.248573%|1.986325|1.986578|

These are complete **exploratory**, not formally isolated, measurements.
The independent observer audit accepts30/36 windows and conservatively rejects
six medium/long windows because their first post-completion sample reports
OUTSIDE_CPU_BUSY. All six flagged samples occur approximately9–10s after the
original response-file completion boundary. Teardown ownership attribution is
a hypothesis, not proven absence of contention. The original audit remains
CONTAMINATED_OR_UNOBSERVED / PENDING_HOST_ISOLATION; it is not rewritten PASS.
No samples or slow measurements were dropped, and no claim of statistically
significant speedup or formal no-benefit is made. The user permits exploratory
Static screening, so this limitation does not block proceeding with the retained
known-good Uniform control. It does prevent advertising this campaign as formal
isolated performance.

## Routing-based justification

The combined NPU-hit rate improves12.4405%→26.2652%, but exact-history stage
separation shows that this is **prefill-dominated**. Frequency prefill CPU
assignments decrease16.0494% relative; decode CPU assignments increase1.2582%,
CPU route-weight mass increases2.2465%, and decode NPU-hit rate decreases
1.1058pp. The long decode lane shows essentially unchanged throughput.

Frequency has not established a clear improvement in the target decode CPU
exposure/performance chain. Preserve Uniform rather than treating a combined
prefill/decode hit-rate gain as decode acceleration. This selection is not based
on Frequency's CMMLU score, a new Static3% hard threshold, or a changed quality
criterion. Proceed to Dynamic against Uniform and independently evaluate the
request-prefill-specific adaptation mechanism.

## Next ordered gate

After the completed Static driver is reaped, execute the original-binary CPU
backing eviction regression on layers1/13/26. Preserve any failure before a
minimal loader repair. A common-loader repair requires new binary provenance,
fresh R64, necessary local operator and Static compact regression before Dynamic
model qualification. Then DEV-only thresholds256/512/1024, commit candidate,
and only afterwards held-out routing/quality/performance.
