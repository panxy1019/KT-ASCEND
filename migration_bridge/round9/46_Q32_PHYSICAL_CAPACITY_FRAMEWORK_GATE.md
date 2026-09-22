# F7 — Q32 physical capacity gate

Date:2026-09-21. Scope: exclusive physical15 on110.123.0.3, C128,
MXFP4 complete CPU backing, original BF16 NPU residents. Framework short tests;
no quality/performance or long-context qualification.

## Implementation

Resident capacity is an independent SHA-bound framework profile field. Historical
four-field CPU bank/candidate identities are unchanged. Default remains Q64.
Q32 launch requires its own physical32 device receipt, native identity and source
gate. A Q64 receipt cannot grant Q32 permission. Actual model tensor shapes and
all48logical/inference/reverse maps are inspected before the first forward.

## Real module gate

`evidence/physical15/20260920-c128-expert-v18/result.json` records
PASS_Q32_REAL_KTEP_LOCAL_MERGE. Actual32 processed original BF16 experts inNZ29
match checkpoint weights. Reverse/restore of the maximum32expert copy is exact.
Three promote/demote/promote transactions preserve pinned mask pointers; nine
numeric cases with three repeats pass.

Per-layer tensors: w13[32,2560,1280], w2[32,640,2560], BF16,314572800bytes.
Explicit maximum projection staging is200MiB; observed additional migration
allocation including conversion/workspace is629147648bytes. These quantities
have different scopes and must not be substituted for one another.

## Failure and correction

Trial28 loaded the model successfully but the physical recorder rejected the
primary logical map as non-CPU before the first forward. KTEP creates this tensor
with torch.full without an explicit device under the loader's NPU context;
the upstream CPU comment is not its actual placement contract. In contrast,
the inference map must remain on NPU and the reverse map is CPU.

The correction accepts and records CPU/NPU primary maps while retaining exact
int32,512-element,logical-value checks and strict inference-map device checks.
A local regression reproduced the failure before the correction;18 relevant
tests then passed. Original migration math and native code did not change.
Trial28 stays FAIL with its source/logs intact.

## Static full-model gate

Fresh trial29/source f7-v3/overlay5 passes:

- Two HOLDOUT requests; finite output, exact integer391 and the translation
  “Seawater contains salt.”
-48physical layers,32slots each, maps/masks agree. Primary maps observed onNPU.
- Total routed resident tensor bytes15099494400 =14.0625GiB; exactly half of
  the corresponding Q64 tensor geometry. This is actual physical allocation,
  not merely fewer True values in a Q64 mask.
- Active HTTP abort followed by identical Static request output on reuse.
- Observed owned-process HBM peak25446MiB, board peak28213MiB. Q64 Static
  trial24 observed39848MiB owned; these are scoped short-run resource observations,
  not a performance benchmark or a future peak guarantee.
- Controlled operator stop after completion; isolated source and raw evidence
  in `evidence/physical15/20260920-full-model-v29`.

## Dynamic full-model gate

Trial30/source f7-v3/overlay5 passes the same two HOLDOUT requests, known-answer391,
real request-boundary migrations and activeabort/reuse. The four transactions are:

| Request | Generation | Changed layers | Copied experts |
|---|---:|---:|---:|
| sched-holdout-01 |1|48|1431|
| sched-holdout-02 |2|48|845|
| r9-cancel-active |3|48|850|
| r9-post-cancel-reuse |4|43|98|

Every published layer retains exactly32unique logical experts. Post-cancel reply
is391. Observed owned-process HBM peak26242MiB. Full-model source/payload, actual
physical records, raw HTTP and migration plans are retained in
`evidence/physical15/20260920-full-model-v30`; controlled stop followed completion.

F7=FULL_MODEL_SMOKE for these MXFP4/C128/Q32 Static/request-boundary combinations.
This is not a migration-performance or quality comparison against Q64.

## Remaining

Q32 Frequency, BF16/Q8Q32 model combinations, product quality and performance
remain unqualified. ROUND9_FRAMEWORK_COMPLETE=NO.
