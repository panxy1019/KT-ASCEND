# F0/F1 shared contracts and integration notes

Date: 2026-09-21. Work in progress; this is not a framework completion report.

## Frequency allocation semantics differ from upstream

The recovered KTEP `initialize_gpu_experts_masks` frequency path passes a global
expert budget to `generate_gpu_experts_masks`. The Round9 contract instead needs
exactly64 (later32) physical resident slots in each of48 layers. Reusing the
upstream strategy would not establish the per-layer capacity invariant. Its
missing-statistics uniform fallback also conflicts with fail-closed calibration.

The new producer uses complete logical Top10 route counts, ranks each layer by
descending count with ascending logical ID as tie-break, then sorts selected IDs
for deterministic physical allocation. The full count matrix and producer hashes
remain in the manifest. Validation recomputes the result, so modifying a mask and
recalculating its hash does not bypass deterministic selection. Schema acceptance
ofQ32 is planning capability only; the F1 real runtime rejectsQ32 until F7.

Calibration and holdout are small hand-authored corpora independent of CMMLU.
Live determinism requires matching tokenizer/runtime/placement/sampling; a fixed
trace replay is deterministic without claiming all live model runs bit-identical.
Changing low-bit CPU versus BF16 NPU ownership can change numerical results, so
cross-placement token identity is not assumed. Same-placement feature-OFF versus
collector-ON can be checked directly.

## Partial requests must not contaminate statistics

The shared RouteStats keeps one bounded per-phase scratch and commits it only
after every layer covers the entire contiguous token range. Invalid IDs, duplicate
experts, stale generation, overlapping spans and nonfinite weights reject before
mutation. PrefillCollector verifies each scheduler chunk separately, and complete
prompt coverage before aggregate publication. Cancelling between chunks discards
only that request's scratch, retaining earlier completed requests.

The real F1 hook intercepts original logical IDs before KTEP slot remapping.
It accepts serial Static only, no prefix cache, no graph and no overlap; Dynamic
interaction needs a separate implementation/gate. Static placement has epoch0
for the loaded process. A maximum64 completed requests bounds calibration
metadata; duplicate completed request IDs reject instead of silently double-counting.

## Atomic hook installation and failure state

Initial review found that installing wrappers while discovering layers could
leave a partial installation if a later layer was missing or duplicated. The
implementation now verifies the entire48-layer set before changing any apply
method. MoE observation/forward failure marks the collector broken and requires
process restart. Other forward failures remain fatal under the current normal
scheduler; no recoverable model-exception lifecycle is qualified here.

Frequency identity was initially process-global; it is now bound to the KTEP
wrapper that owns the actual mask. Changed/missing masks after binding reject.
Installation must precede any physical expert allocation, and file SHA is checked
again during model construction.

## Cancellation does not release in-flight byte credits

HostWorkBudget tracks outstanding host/H2D/pinned bytes under one lock. Optional
prefetch cannot consume configured correctness reserves. Cancellation signals the
owner but does not return credits until backend completion and explicit release.
Copied or forged allocation objects are rejected even if field values match.

Review found an unbounded per-request placement dictionary. `retire_request` now
removes a cursor only after all request allocations are released; it rejects
active or in-flight ownership. Allocation epochs never reset, including request
ID reuse after retirement. This is a local ownership contract, not measured DRAM
bandwidth or a starvation-free scheduler; actual Expert/PLE integration is pending.

## Validation scope

Meaningful RED/GREEN host tests cover the contracts above. The new overlay is
installed separately from frozen candidatev8 and binds its whole-tree SHA.
Trial24 begins a fresh feature-OFF full-model smoke on exclusive physical15.
All real result claims belong in the evidence ledger after the run completes.

No CMMLU256, ABBA, long-context real generation or performance claim is introduced.

## NUMA admission after a clean model stop

Trial25's first launch correctly rejected resource admission before creating a
container: selected nodes2 and7 had less than80GiB free despite abundant total
host memory. Inspection showed large file-cache residency after trial24 exited.
Applied POSIX_FADV_DONTNEED only to the48 owned MXFP4 GGUF files, with size/mtime/
inode unchanged and before/after NUMA snapshots recorded. No global cache drop,
threshold reduction or unrelated process action was used. The identical fresh
admission then passed and trial25 started under the existing live guard.

Trial24 feature-OFF ran two DEV and two HOLDOUT requests successfully. The integer
canary returned391, translation returned `Seawater contains salt.`, and all logged
output probabilities were finite. The three local-environment-dependent test
modules plus hook tests passed16/16 inside the real container environment.
Trial25 collector-ON and subsequent Frequency integration remain in progress.
