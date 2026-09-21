# F5/F6 — PLE consumer and shared migration-credit progress

Date:2026-09-21. Historical implementation notes are retained below. The latest
bounded device/full-model results are recorded in the next section and supersede
the earlier `NOT RUN` statements for F5/F6.

## Latest bounded gates

F5 now reaches `FULL_MODEL_SMOKE` for the exact MXFP4/C128/Q64/Static profile.
Trial35 uses the v11 overlay and records 11 real PLE batches with
`async_engaged=true`; HOLDOUT2 is finite and preserves the integer-391 canary.
Trial36 uses the same overlay with the feature flag absent. It creates no async
event file, produces exactly the same output token IDs, and has maximum output
logprob absolute difference 0.0 versus trial35. These are short correctness
smokes, not overlap or performance results.

F6 now reaches `FULL_MODEL_SMOKE` for MXFP4/C128/Q32/Request-Dynamic + Async PLE
+ shared `HostWorkBudget`. Trial39 completes HOLDOUT2 and active abort/reuse,
records 18 real async PLE drains, and returns to the frozen persistent ring
baseline at four request boundaries. That baseline contains two ring-slot
allocations and 327,680 pinned bytes; request host/H2D bytes, in-flight work and
cancelled work are absent.

Trial38 is intentionally preserved as a failed development run. Its first
request executed correctly, then the new audit asserted that the whole budget
must be zero. The assertion incorrectly counted the two lifetime ring slots as
request leaks and stopped the scheduler. The fix freezes the budget immediately
after ring construction and requires exact equality to that baseline after every
request. A new unit case covers a nonzero persistent baseline; CPU snapshot v15
passes 329 tests and fresh overlay installation before trial39.

## Why a ring alone was insufficient

The original PLE table gather occurs in `Qwen4ReferencePLE.prepare`, before
layer0. The actual arithmetic/state consumer is the PLE block at layer1.
The new bridge starts exact row gathering in prepare, defers the device event
wait to that consumer, then records and drains the consumer stream before
returning the ring slot. It preserves the original `PLE.run` calculation and
convolution-state update. N-gram history is still committed only at the end of
the complete model forward.

Only already-known input IDs are admitted: the current prefill chunk or the
previously sampled token entering decode. No future token is predicted. The
serial scheduler binds request ID, placement generation, exact input IDs and
the allocated recurrent-state slot; the real prepare call verifies them before
submitting work. Pool slot alone is never treated as request identity.

The synchronous fallback uses the same batch-stamped payload wrapper as async
rows. This matters because a tensor with the right shape can still belong to
a prior request. Both paths verify shape and exact payload ownership. A model
failure poisons the consumer runtime even if buffer cleanup succeeds: partial
GDN/PLE/model state cannot be made valid merely by draining the copy stream.

## Default-off real seams

- Framework schema3 retains schema1/2, adds explicit `ple:null` by default, or
  `ple:{mode:"ASYNC",device_gate:{path,sha256}}`.
- The fresh-overlay installer patches only the original prepare gather seam;
  the flag-OFF branch still calls the synchronous original table directly.
- `ple_model_ready` verifies the exact original bank/model identity, component
  receipt, source hashes and Torch stack before allocating two pinned/device
  slots. The actual one-layer/16-head/160-wide geometry is checked.
- Persistent buffers allocate before KV pool sizing. The measured transient
  component allowance is subtracted before token/GDN pool sizing.
- Scheduler hooks bind exact inputs before the forward and drain the PLE
  consumer before the prefill-end Dynamic barrier snapshots live state.
- Request-end releases transient budget request metadata; lifetime pinned
  reservations remain owned by the model's ring.

The component probe v2 has run and passed. It reads original rows
across all128 shards, checks duplicates and actual device bytes, uses a separate
consumer stream, cancels an intentionally delayed CPU gather, verifies stale
ticket rejection and slot/request reuse, and requires zero credits after close.
Its source-bound receipt is `evidence/physical15/20260921-ple-v2/result.json`.

## Shared host-work accounting

`HostWorkBudget` tracks byte reservations, not memory bandwidth. PLE charges
twice its row payload for simultaneous original-row cache plus joined bytes,
once for H2D payload, and separate persistent pinned slot capacity. Python object
and allocator overhead remain covered by the process resource guard, not an
exact RSS claim by this logical-byte budget.

`BudgetedMigration` wraps the existing original-BF16 adapter. It reserves four
expert payloads of host staging and the requested batch payload of H2D credit
before any read or copy. It calls the existing copy implementation and its real
synchronize method, then returns credits. Failed copies that drain successfully
still poison the wrapper; failed synchronization retains the allocation and
requires reload. A denied budget never starts a copy or waits indefinitely.

Framework schema4 adds `joint_host_budget:false` by default. Enabling it requires
both Async PLE and an actual Dynamic scheduler binding. The model owns one shared
budget; the existing controller's adapter receives a credit wrapper while the
same migration engine and publication path remain in use. PLE cannot retire the
request's budget cursor early: the Dynamic request-end hook does so after both
planes drain. Prefill-end publication preserves that cursor through decode.

The CPU contention test holds PLE credits while migrating and verifies combined
usage never exceeds the shared limit. Failure tests verify no copy on denial,
drain-before-release and unsafe-copy quarantine. The runtime binding is implemented
but **not device or full-model qualified**; therefore F6 remains PARTIAL.
Strict priority is not starvation freedom, and none is claimed.

## Outstanding gates

1. Fresh component launch for PLE under current resource admission.
2. Full-model Sync versus Async exact-row/output differential smoke, real
   async engagement, abort/reuse, feature-OFF and clean restart.
3. F4 fresh Dynamic gate and live-state-preserving full-model migration.
4. Explicit F6 enablement and shared-owner request retirement after both PLE
   and expert migration drain; synthetic and real contention evidence.
5. Feature interactions and subsequent quality/performance qualification.

The remaining work is F12 interaction accounting, final restart sampling and
later quality/performance qualification. No overlap benefit is claimed.
