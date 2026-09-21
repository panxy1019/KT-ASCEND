# Framework ledger — plan: 43_FRAMEWORK_COMPLETION_EXECUTION_PLAN.md

## Latest: v35–v44 integration and framework freeze preparation

F5 v35 passes the real async PLE consumer with 11 drains; v36 feature-OFF has
identical tokens, zero maximum logprob difference and no async event file. F6
v38 preserves an incorrect absolute-zero budget invariant; v39 fixes it by
requiring exact return to the immutable persistent-ring baseline, then passes
Q32 request Dynamic, PLE async, joint budgeting and active abort/reuse.

The context-tier v2 device probe passes exact all-12-layer K/V spill/reload from
physical generation `(0,0)` to `(1,1)`. It is not connected to the model
allocator. F9 v40 preserves an idle-hook failure. Final CPU snapshot v16 adds
the tested idle/inactive distinction, passes331 tests and installs a fresh
overlay. v41 then passes actual4096 pool allocation, FP8-host-reference to BF16
staging on real QSA,11 batches, two drains and HOLDOUT2.

F11 v43 passes HOLDOUT2 and demonstrates two complete contiguous 32-token chunks.
HTTP abort returns200 during the long prefill but does not promptly interrupt
later scheduler chunks; this is a recorded limitation, not a pass. v44 is the
clean-restart/two-chunk completion and ordinary abort/reuse PASS. F12 v45 closes
the missing Prefill-End + Layerwise + async PLE cell on one real request:48-layer
prefill, four async drains, generation-1 commit/final drain and integer391 all
pass with no cgroup pressure event. Final F12/F13 artifacts bind these receipts. No
CMMLU256, formal performance, real long-context or deployment claim is added.

## Latest: MXFP4 NUMA trial34 and F4 bounded full-model smoke PASS

Trial33 preserves the clean-file-cache false stop. The corrected runtime guard
requires low MemFree and low conservative reclaim headroom together, or an owned
cgroup pressure/allocation event. Trial34 completes model load, HOLDOUT2, four
real Prefill-End transactions, active abort and state reuse on exclusive15.
Per-node owned-anonymous peaks are23.086/18.063/15.072/14.997GiB; cgroup
high/OOM/failcnt are zero.23 targeted NUMA/monitor/summary tests pass. The48GiB
startup policy is qualified only for this exact MXFP4 short-smoke scope; BF16 and
Q8_0 keep80GiB. F4 now reaches FULL_MODEL_SMOKE for Q32/Layerwise/Prefill-End,
while abort-during-prefill and injected full-model migration failure remain open.
F5 async PLE full-model reference/differential work is next. Framework complete=NO.

CPU follow-up v6 passes300 tests and fresh real-baseline installation (both
exit0). It adds the source/scope/numerical/drain validator for the actual context
component receipt plus mutation negatives. Raw logs, no-NPU container inspection
and source hashes are archived in20260921-framework-cpu-v6. The capability
checkpoint references this snapshot; no new full-model qualification follows.
NUMA alternatives were raised with the user; pending a response, retain the
original topology and80GiB/node gate. No full-model process is running.

## Latest: 2026-09-21 07:19 UTC context reference component PASS

CPU v5 passes298 tests with actual Torch; real baseline overlay installation also
passes. This snapshot includes F8 post-pool measurement binding, selected-row
staging, the page-generation retention fix, and precise abort-drain predicates.
The owned CPU container maps no NPU devices. Four actual frozen/current source
hash comparisons confirmed context staging/measurement/binding/probe identity.

Context component probe v4 exits0 and passes12 actual-QSA-layer synthetic cases.
NPU BF16 KV -> immutable host BF16 records -> compact BF16 NPU staging preserves
QSA output exactly. FP8 record decode -> compact staging also exactly matches a
full decoded-page NPU QSA oracle; maximum relative-L2 versus original BF16 is
0.03765484318137169 under the fixture-specific0.10 bound. Peak extra allocation
355328bytes. Host lease invalidation drains correctly. No model hook, allocator
page release, native FP8, HiCache, long-context or performance claim follows.
Evidence: physical15/20260921-framework-cpu-v5 and20260921-ple-v4.

F8 opt-in `--context-target` supports4k/16k/64k/128k sizing and source-bound
post-initialization actual tensor measurement. It remains unexecuted in a full
model. Full-model NUMA2 admission remains below80GiB (07:17:83,263,152KiB).
F4/F5/F6 full-model gates, F9/F10 actual tier ownership, F11/F12 and F13 remain open.
ROUND9_FRAMEWORK_COMPLETE=NO.

## Latest: 2026-09-21 05:52 UTC component gates pass; full-model NUMA admission blocked

Physical15 became idle again. Fresh exclusive admission allowed PLE v2 and joint
budget v3 to execute; both containers exited0. No shared-training mode was enabled.
PLE v2 passes original BF16 row identity, cancellation, stream ordering and reuse,
with493056bytes extra device allocation. Joint v3 passes real simultaneous PLE and
two synthetic expert H2D payloads, required migration progress, exact copy digest,
and zero final credits. Peak extra allocation20154368bytes. This is component-only
transport evidence, not a model or overlap/performance claim. Current schema4 is
bound by PLE v2; joint v3 binds that exact standalone receipt.

Fresh Dynamic Q32 component v19 also passes against the current migration core.
CPU snapshot v4 passes274 tests, including actual Torch FP8 codec cases, in a
container with no NPU devices. Its installer check passes against the preserved
real SGLang source. Later F6 probe assertions and launcher edits are separately
frozen in the real v3 probe source; do not ascribe them to CPU snapshot v4.

Full-model v33 has not started: fresh admission rejects NUMA2 MemFree below80GiB
(05:50 sample82934124KiB). Project-owned unmapped checkpoint/bank file cache was
reclaimed without changing bytes or metadata. A resident-page scan found no
remaining resident cache in those275 canonical files. The guard remains unchanged;
no unrelated cache, training process, or topology was altered. Repeated startup
attempts without a resource change are not a useful gate.

F8 real TP1 geometry and F9 FP8/F10 request-local host page contracts are CPU-only;
actual QSA consumer, allocator/tier lifecycle, F11 and F12 remain unfinished.
ROUND9_FRAMEWORK_COMPLETE=NO. Raw new gates are archived under evidence/physical15
with per-directory archive-sha256.json. Historical sections below retain their
original point-in-time status.

## Historical: 2026-09-21 05:04 UTC external training resumed

F2 component gate v2 passes actual BF16 private transport. Full-model v31 failed
at the FusedMoE keyword dispatch seam; fixed by preserving `layer` and
`dispatch_output` parameter names. Fresh v32 completes both HOLDOUT requests,
all48 layers, finite outputs and integer391. Output IDs equal Q32 Hybrid v29;
maximum output logprob difference is0.02407521102577448. Different prefill
arithmetic (original BF16 private bank versus MXFP4 CPU backing) is explicitly
not byte-identical. No quality/performance qualification is inferred.

At05:04:41UTC the exclusive guard stopped only our v32 container because a new
physical15 PID3425392 appeared in `mzy-train`. This is not an owned OOM or a
numerical failure. Abort/reuse had not started and remains NOT_RUN. Do not restart
the device suite without fresh admission; a clarification about exclusive versus
new shared training peak is pending. CPU development continues.

F4 live-prefill barrier and124-component state snapshot passed host review;
fresh Dynamic math component gate and real full-model integration remain pending.
F3 bounded complete-prefill count/mass summary is implemented and locally tested.
F5 ring failure paths are fixed; default-OFF exact-input consumer/binding work is
in progress. Neither F4 nor F5 has new device qualification.

Frozen CPU snapshot `framework-cpu-v3` passes230 tests in a CPU-only container,
with no NPU devices mapped. The first v2 package omitted a historical JSON test
fixture and native header; its six errors remain recorded. v3 added only those
fixtures and passed. Later source additions need their own fresh checks.

ROUND9_FRAMEWORK_COMPLETE=NO. F2 lifecycle/fallback/restart gates, F4–F13 device
and integration work remain open. The historical sections below are append-only
progress records, not the current status of each phase.

## 2026-09-21 start

User approved reviewed architecture and execution; no further design approval needed.
Existing feature branch is the implementation workspace. Untracked historical
docs/handoffs/2026-09-20-04-PAUSED.md belongs to the user and remains untouched.

Pre-flight: F1/F3 share telemetry; F2/F4 need a live-state-preserving barrier;
F5/F6 share buffer lifetime and budgets; F8/F9/F10 share page/state identity;
F7 capacity must enter the schema before these consumers. Resolved by approved
contract rulings in43, not by removing old guards.

F0 COMPLETE: baseline manifest records six exact historical trial identities;
every referenced summary evidence hash was recomputed successfully. This freezes
history and does not grant a new overlay resource or correctness admission.

F0a/F3a/F1 IN_PROGRESS. CPU contracts now cover separate request/placement/buffer
epochs, bounded route counters, complete prefill spans and deterministic per-layer
frequency Q64/Q32 manifests. Frequency replays counts and checks exact producer
identity and payload, rather than accepting an arbitrary re-hashed mask.
F6 groundwork: HostWorkBudget has host/H2D/pinned credits and explicit completion
before cancellation reclamation. It is not wired to real traffic yet.

Local TDD: route stats5, frequency manifest3, prefill spans4, config2 and budget6
tests pass. Missing implementations produced RED before implementation; budget
also caught accepting a copied allocation lease and now requires the issued object.
The wider CPU discovery has122 importable passes;3 modules need numpy/torch that
are absent from local Python. Their remote environment checks remain pending.

F1 real integration: installed isolated framework-sglang-v1 from preserved baseline.
New flags default OFF. Real collector intercepts original logical Top10 before
KTEP remapping; serial48-layer complete prefill only; maximum64 calibration requests.
Frequency is fixed per-layer, deliberately distinct from upstream global-budget
frequency. Q32 schema acceptance still does not enable physical Q32 launching.

Starting trial24: new overlay feature-OFF, MXFP4/static/Q64/C128, exclusively
physical15 after fresh resource admission. No prior trial is re-labelled.
All later stages PENDING. ROUND9_FRAMEWORK_COMPLETE=NO.

## F1 real progress

- Trial24 feature-OFF: four short HTTP requests pass, including integer391;
  owned observed peak39848MiB. Torch/numpy-dependent prior tests plus hooks pass16/16
  in the real container environment. Controlled operator stop; no resource guard stop.
- Trial25 initial admission blocked on per-node free memory. Reclaimed only48
  owned MXFP4 bank file caches with POSIX_FADV_DONTNEED; file signatures unchanged.
  Fresh identical admission passed. No global cache drop or weakened threshold.
- Trial25 collector-ON: two DEV requests, actual47 input tokens, complete48-layer
  statistics. Output token IDs exactly match trial24; maximum output logprob
  difference is0.0. Owned observed peak39842MiB. No abort/restart claim yet.
- Q64/Q32 Frequency manifests generated from these actual routes and bound to
  corpus/tokenizer/runtime/initial placement identity. Trial26 starts real Q64
  Frequency HOLDOUT smoke; restart remains pending.
- Q32 groundwork: strict capacity/native/source-specific receipt validator and
  physical-tensor probe added; no Q32 NPU test has run yet. The probe records
  explicit staging separately from measured allocation/workspace effects.

Local evidence: evidence/physical15/20260920-full-model-v24 andv25,
each with framework-summary.json and frozen source/payload hashes.
ROUND9_FRAMEWORK_COMPLETE remains NO; F1 is not frozen until restart and real
loaded map checks pass. F2/F4/F5/F6 integration/F7 device/F8–F13 remain pending.

## F1 gate complete; F7 device started

Trial26 and27 completed: real48maps matchmanifest,2578resident replacements versus
Uniform, HOLDOUT2canary391, activeabort/reuse, cleanrestart identical output IDs.
F1=FULL_MODEL_SMOKE forMXFP4/C128/Q64/StaticFrequency+routecollection only.
See45_STATIC_FREQUENCY_FRAMEWORK_GATE.md for exact evidence and limitations.

F7 bounded real module probev18 launched on exclusivephysical15 using fresh source
framework-dev-20260921-f7-v2. Actual32 processed expert tensors verified original
BF16 identity and NZ29; migration/numeric completion still pending.

F2 standalone private-shadow transport andF4 live-prefill transaction are local
groundwork only. F4 retains the live request lease and reuses the existing engine;
copy/sync/publish negatives and old transaction regression pass. Its source SHA
changed, so fresh device gates must bind it before integration. No old receipt
is silently inherited. ROUND9_FRAMEWORK_COMPLETE=NO.

## F7 module gate and initial system failure

Probe18: PASS_Q32_REAL_KTEP_LOCAL_MERGE. Actual32 original BF16 experts in NZ29,
reverse/restore identity, three migration transactions and nine numerical cases
with three repeats pass. Measured migration extra allocation629147648bytes;
explicit projection200MiB is not the complete workspace requirement.

Trial28 loads the full model but fails before its first forward in the new Q32
physical recorder. Root cause: KTEP constructs its primary logical map with
torch.full without an explicit device inside the loader's NPU device context.
The recorder incorrectly assumed CPU from the upstream comment. Its inference
map is also NPU; the reverse map remains CPU. Accept CPU/NPU for the primary map
only, record the actual location, retain exact int32/shape/value checks and the
strict NPU inference-map requirement. A regression first reproduced the failure,
then the fix and17other relevant tests passed. Trial28 stays FAIL, never relabelled.
Fresh source f7-v3 / overlay5 / trial29 retests Q32 Static. Core migration math
is unchanged from the separate probe18 source gate.

F2 review found and fixed FP32 contribution, max32token workspace admission,
transient constructor peak and poisoned-sync cleanup issues. Fourteen host tests
pass; its own device probe remains pending. F8 context andF5 async work proceed
as isolated default-OFF contracts; they have no device/full-model qualification.

## F7 bounded gate complete; F2 device started

Trial29 Q32Static passes actual48physical maps/tensors, HOLDOUT2 and activeabort/reuse,
ownedpeak25446MiB. Trial30 Q32Dynamic passes HOLDOUT2 and abort/reuse, four real
transactions generations1..4, ownedpeak26242MiB. BothscopeMXFP4/C128/exclusive15.
See46_Q32_PHYSICAL_CAPACITY_FRAMEWORK_GATE.md. Product/performance remainpending.

F2 standalone layerwise probev1 starts underfreshguard with private32BF16shadow,
original-source identity, completeTop10 groupedroute numeric oracle, template
nonaliasing, abort/reuse and separateconstructor/executionpeak measurements.
Coordinator hosttests now coverfailurecleanup, event-sink failure, between-chunk
abort onidle/decode, and immutable request/span/generation continuation.
New profile schema2 preserves schema1 exactly and adds explicit default-OFF
layerwise settings; complete model integration awaits its own realdevicegate.
F5 async andF8context hostgroundwork exist but are not yet integrated/qualified.
ROUND9_FRAMEWORK_COMPLETE=NO.
