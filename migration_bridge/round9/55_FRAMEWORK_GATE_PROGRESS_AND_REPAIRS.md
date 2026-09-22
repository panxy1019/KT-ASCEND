# Framework gate progress and repairs — 2026-09-21

The F0–F13 sprint is **in progress**. This checkpoint records new narrow evidence;
it does not freeze the framework or grant product/performance qualification.

## PLE asynchronous original-row gate

Physical15 probe v2 exited0 and records `PASS_PLE_ASYNC_ORIGINAL_ROWS`. It reads
the original checkpoint's rows, preserves duplicate and cross-shard identities,
orders the consumer stream after transfer, cancels a deliberately late gather,
and reuses slots across request generations. All credits return to zero after
close. The two pinned slots occupy327680bytes and the two device slots occupy
327680bytes; measured additional device peak is493056bytes.

The earlier v1 receipt is preserved, but its framework-config source hash became
stale when schema4 gained the required joint receipt. v2 repeats the device gate
against the current schema rather than inheriting that old qualification.
Evidence: `evidence/physical15/20260921-ple-v2/`.

## Joint transfer budget gate

Physical15 probe v3 exited0 and records
`PASS_JOINT_BUDGET_SYNTHETIC_CONTENTION`. Current PLE and optional prefetch hold
live credits while two synthetic expert payloads undergo actual H2D copy.
The19,660,800 copied bytes match independently constructed expected BF16 values
and a fixed digest. All six ledger snapshots match the specified amounts and
the final counters are zero. Additional device peak is20,154,368bytes.

This proves bounded transport interaction, not expert arithmetic or full-model
joint operation. The complete Q32 migration budget is checked arithmetically;
only two expert payloads are physically copied in this component test. Runtime
joint mode now requires this separate source-bound receipt and the exact PLE
receipt it references. See54_JOINT_BUDGET_GATE.md and
`evidence/physical15/20260921-ple-v3/`.

## Dynamic source refresh and CPU package validation

The Q32 expert v19 probe passes with the current migration core; v18 remains
historical evidence for its own source. CPU snapshot v4 passes274 tests in a
container with no NPU devices and installs a fresh overlay against the preserved
real SGLang baseline. The FP8 codec tests ran with actual Torch CPU tensors.
The first CPU package v2 had six missing-fixture/header errors; v3 corrected
the package and passed230 tests. Those failures are retained, not relabeled.
Evidence directories carry source hashes and raw logs.

## Abort smoke could confuse a commit with a drain

Inspection found that the HTTP abort driver accepted any Dynamic event naming
the target request. A future Prefill-End Dynamic commit also names the request,
although its KV/GDN/PLE consumers remain live. That predicate would incorrectly
report successful drainage before the final request-boundary event.

The driver now distinguishes the existing request-boundary commit schema from
`PREFILL_END_REQUEST_DRAINED`, requires the latter to report zero second
migration, and rejects `PREFILL_END_COMMIT` as drain evidence. Four targeted
tests cover the premature event, final drain, old schema, and another request.
The HTTP timeout is600seconds, matching the existing layerwise smoke envelope;
this is not a performance target. No new HTTP abort success is claimed until
the revised driver executes against a freshly admitted model.

## Complete-model admission remains blocked

Training had occupied physical15 at05:04UTC and our exclusive guard stopped only
our model. It later became idle, allowing fresh component admission. Full-model
v33 still never started: NUMA2 MemFree remains below the unchanged80GiB minimum.
The05:50UTC sample was82,934,124KiB, approximately79.09GiB.

Only project-owned unmapped weight-file cache was reclaimed. Identity metadata
and file bytes were not modified; a scan found no resident pages left in those
275 canonical files. No unrelated cache was dropped and no other process was
stopped. The three prelaunch failures and owned-cache evidence are archived in
`evidence/physical15/20260921-framework-resource-block/`.

Next full-model execution must pass fresh CPU/NUMA/exclusive-HBM admission.
F4/F5/F6 model integration, actual context-tier ownership, streaming and the
interaction matrix remain open. `ROUND9_FRAMEWORK_COMPLETE = NO`.

## Later full-model integration repairs

This section supersedes the point-in-time open list above without rewriting the
preserved failures.

### PLE async feature-OFF differential

Full-model v35 passes exact-input PLE async with 11 real async drains, HOLDOUT2,
finite outputs and integer canary 391. v36 runs the same candidate with the
feature absent. It emits no PLE event file, returns identical output token IDs
and has maximum output-logprob absolute difference 0.0. The comparison proves
the bounded default-OFF seam only; no performance conclusion follows.

### Persistent ring reservations are a baseline, not a leak

v38 initially required the joint budget to return to absolute zero after every
request. This was wrong because the two reusable PLE ring slots intentionally
retain 327680 pinned bytes and two ownership records for the model process.
The failure is preserved. A regression froze the ledger immediately after ring
construction, and request end now requires exact equality with that immutable
persistent baseline. v39 passes Q32 request-boundary Dynamic, async PLE, joint
budgeting, HOLDOUT2 and active abort/reuse. Four request drains return exactly to
the baseline, with zero in-flight and cancelled work.

### Idle scheduler callbacks are not active leases

The first F9 full-model attempt v40 called a post-batch hook during an idle
scheduler iteration. The reference runtime rejected the missing lease even
though it had no active request. Streaming had the same latent problem. Tests
now distinguish inactive idle callbacks from an active runtime that lost its
lease: the former return and the latter still fail closed. Final CPU snapshot
v16 passes 331 tests and a fresh overlay install. v41 then passes the real
FP8-host-record to BF16-stage QSA path on all 12 QSA layers, actual 4096-token
pool measurement, 11 batches, two request drains and HOLDOUT2.

### Streaming launch bookkeeping

The first v42 launch attempt used the source directory as the overlay path. The
launcher failed before container creation at strict path resolution; the
evidence directory remains preserved. Reusing version 42 correctly failed on
the existing evidence directory, which is the intended no-overwrite behavior.
v43 uses the exact v16 overlay path and is the separately identified streaming
gate. Neither v42 attempt is relabelled as a model result.
