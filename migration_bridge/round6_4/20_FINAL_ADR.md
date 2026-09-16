# Round6.4 final decision — implemented, no measurable speedup

Status: **COMPLETE — IMPLEMENTED_NO_SPEEDUP**, 2026-09-16.
This is investigation/qualification closure, not an acceleration READY release.

## Accepted mechanism and unchanged hard gates

The user accepts upstream-style cross-layer Deferred as an approximate
acceleration: contributions computed from one layer are consumed across the
next layer boundary. ON/OFF token ordering and logits need not be exact.
This changes the acceptance target, not the historical Round6.3 rejection.

System ownership, exactly-once contribution/weight/scaling/shared handling,
same-path determinism, finite outputs, framework-managed transfers, buffer
lifetime and terminal draining remain hard requirements. Same-operands CPU
expert relative-L2 must remain <=0.01. No production implementation changed
during Round6.4; test/tool fixes are independently documented.

## Evidence already complete

- Focused retained regression and nonDeferred P1/P2/P3 controls: docs03–11.
- D1 batch1 same-process and clean restart: doc06; explicit graph oracle:07.
- Real-task lifecycle and expert/no-hit safety:04/08.
- Batch2 engineering controls passed; batch4 same-path failure remains
  unqualified with pre-L7 NPU/KV localization, not excused by approximation:09.
- User-approved GSM32/CMMLU256 observed2pp screen: both0pp delta, doc15.
  This is **not population noninferiority**, especially GSM5/32 with high truncation.
- Performance observer parity and actual Deferred exposure:16.
- Three frozen ABBA blocks:0.989243x,95% CI[0.912743,1.072153], doc17.
  **NO_MEASURABLE_SPEEDUP**. No post-result workload search or additional blocks.
- Two finite stability campaigns2304tokens exact, memory plateau observed:18.

## Final decision and platform boundary

Final descriptive integrity and finite checks passed (doc19). The terminal
classification is **IMPLEMENTED_NO_SPEEDUP**, with the independent label
**QUALITY_SCREEN_PASS** for the frozen D1 batch1 small-sample experiment.
It is not `DEFERRED_APPROX_ACCELERATION_READY`; no acceleration READY tag.
This label refers to the source-platform frozen D1 batch1 experiment. It does
not assert no possible speedup on every platform or workload. Target910B32worker
has migration/expert/compact/capture evidence, NOT new full quality/performance
qualification. No aggregate PASS combines these as one homogeneous experiment.

No general quality, performance or batch>1 production qualification follows.
Deferred remains defaultOFF. No TP2, Graph, Dynamic Placement, new model,
new GEMM backend or automatic performance optimization is included.

## Completed supplemental evidence

Oldhost `r64-descriptive-final-v2`: OFF integrityPASS, ON forced interrupted by
user-requested pause/migration; ON native not run. Originalv1 environment failure
remains preserved; importer fix changes only child PYTHONPATH. Destination910B
uses its own full OFF/ON-forced/ON-native cohort after CPU-binary compatibility
and compact determinism requalification (doc21). Never splice crosshost captures.
Source quality/performance/stability are explicitly source-platform evidence;
they do not qualify destination32worker quality or speed.
Target3modes all completed136requests/8840forwards each, unknown/incomplete0.
Final metadata auditPASS; analysis finite; HD3free divergences45/128,FD3two/eight.
These distributions are diagnostic and cannot replace quality. The two source
quality raw audits were rerun offline; regenerated summary exactly matches
SHA03cf07513d49b0cf51c8d148192e3a5e1a7e3cb18f3a971280592e50683b37cc.

Target resources were sampled only partially due to the explicit monitoring
pause. No full target memory-stability or target quality claim is made.
Batch4 historical same-path failure remains unqualified. No production-code
change was made to Deferred in this round; target runtime needed a910B operator
package and CPU-ISA-specific extension rebuild, documented independently.

## Release decision

DEFERRED_IMPLEMENTATION_READY = source frozen D1 batch1 evidence complete.
QUALITY_SCREEN_PASS = source fixed GSM32/CMMLU256 observed2pp protocol only.
DEFERRED_SPEEDUP_VERIFIED = NO.
DEFERRED_APPROX_ACCELERATION_READY = NO; default remainsOFF; no READYtag.
TARGET_910B_FULL_QUALITY_AND_PERFORMANCE = NOT_QUALIFIED.
ROUND6_4_EXECUTION = COMPLETE_WITH_EXPLICIT_SCOPE_LIMITATIONS.

## Reopen conditions

Any same-path drift, duplicate/missing expert contribution, stale-buffer
consumption, nonfinite output, mapping/lifetime regression reopens implementation
investigation. A different workload/model/batch/placement or performance redesign
needs its own frozen experiment; this round's small screen cannot qualify it.
