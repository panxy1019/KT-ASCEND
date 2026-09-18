# User-authorized abbreviated performance screen

2026-09-18: user requested shortening ABBA or three-container parallel timing.
Use serial one complete ABBA block per short/medium/long workload (12 cells),
preserving original inputs,256/512/1024 decode budgets, warmup, fresh processes,
threshold1024, C64, observer-OFF and complete host resource-window audit.
Do not perform concurrent bandwidth-contended performance comparisons.

Existing v2 has only two completed short cells at scope change. Preserve it as
STOPPED_BY_USER_SCOPE_CHANGE, not a completed block or performance failure.
Start a separately identified 12-cell screen; no latency-dependent selection or
mixing partial blocks. This costs a small repeat but preserves coherent evidence.

The original36-cell formal repeatability gate remains NOT_COMPLETED_BY_USER_SCOPE_CHANGE.
One block cannot establish across-block repeatability or support automatic default
ON under the old formal gate. Report observed gains/regressions with limited
evidence; retain Static/D0 defaults absent qualified benefit. System/quality
gates are unchanged. The existing formal evaluator must still reject12cells.
Deferred subsequent performance may use the same explicitly abbreviated scope;
its independent system/quality gates are not waived.

## Resource-only whole-block recovery, frozen before retry

09:12 UTC: abbreviated-v1 timed out waiting for host isolation after completing
short4 and medium2. Allshort windows were EXCLUSIVE; mediumturn1 was contended.
Preserve every original artifact. Retain the entire short block; replace the
entire medium block (including its clean turn0), then run the unstarted long
block in a new directory. Selection is based only on resource audit, not timings.
Use `--blocks 1 --buckets medium,long`; never merge individual best cells.
Final combined report must bind both source protocols, identical workload IDs,
runtime/configuration and raw hashes, and audit all12 selected resource windows.
Retry evidence cannot promote the original contaminated attempt or formal36gate.
