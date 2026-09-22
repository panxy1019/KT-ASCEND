# PLE async failure-lifecycle corrections

Date: 2026-09-21

Scope: host-only review and correction of the F5 two-slot asynchronous PLE
ring and the shared F6 byte-budget cancellation handoff. No NPU process,
container, model, or remote runtime was started. These changes do not qualify
device execution or performance.

## Safety invariants

Each ring slot owns reusable pinned host storage and persistent device storage.
The storage may return to `IDLE` only when the ring can prove that no CPU gather,
H2D transfer, or consumer-stream operation can still touch it. Its host/H2D byte
credits follow the same lifetime. A failure that loses the only completion
handle therefore requires quarantine; returning the slot would turn an API
validation error into possible exact-row corruption.

Cancellation is a signal, not reclamation. It is deliberately idempotent for
the same live `Allocation`, because the shared budget can signal optional work
before the owning ring observes the cancellation. Object identity validation
is unchanged: copied, forged, released, and otherwise stale allocations still
fail.

## Confirmed failures and corrections

### Submission returned no trustworthy completion handle

`submit_exact` previously called `backend.submit_gather`, validated the returned
object afterward, and used `_clear_slot` on every exception. A backend could
have started a worker and then returned a malformed object, or raised after
partially submitting work. Cleanup immediately released the storage and byte
credits even though the ring had no handle with which to prove completion. A
later ticket could reuse the same pinned slot while the unknown worker wrote it.

The ring now marks itself `BROKEN_RELOAD_REQUIRED` after any attempted backend
submission that does not produce a valid Future-like handle. It retains the
slot, allocation, and credits and does not call `release_slot`. This is a
deliberately conservative process-reload boundary.

### Cancellation raced admission before `mark_in_flight`

The shared budget can cancel optional allocations immediately after admission.
If that happened before the ring called `mark_in_flight`, the mark correctly
failed, but the old exception cleanup unconditionally called `complete`. That
secondary call failed because work never became in flight, masked the primary
error, and left the allocation and ring slot active.

Admission cleanup is now split at the backend-call boundary. A failed
`mark_in_flight` releases the admitted allocation and resets local slot metadata
without touching backend storage, because no backend call has occurred. Once
submission is attempted, the quarantine rule above applies.

### External and repeated cancellation could not make progress

`HostWorkBudget.cancel_prefetches` can signal a live prefetch before the ring's
own cancellation path. A later `ring.cancel` then failed with "allocation is
already cancelled." Separately, `_cancel_slot` returned immediately whenever
its local cancelled flag was set. If a nonblocking event query had failed after
setting that flag, a retry could never synchronize and retire the slot.

Budget cancellation is now idempotent for the same validated live allocation,
and `_cancel_slot` skips only the duplicate signal. It still performs the state
specific completion check, event synchronization, release, and reset. This
lets external cancellation hand off to the owning ring and lets a caller retry
after a transient query failure.

### Constructor cleanup could leak both lifetime reservations

If physical slot allocation failed and `backend.close` also raised, the close
exception replaced the allocation failure and prevented release of the two
pinned lifetime reservations.

Constructor cleanup now attempts backend close and budget release independently.
It re-raises the primary construction error and attaches cleanup failures as
exception notes. Thus a close failure cannot skip credit release or obscure the
operation that made construction fail.

## Regression coverage

The CPU-only regressions exercise these boundaries directly:

- a started gather returning a malformed handle leaves one slot active, keeps
  its credits, releases no backend storage, and poisons the ring;
- cancellation between budget admission and `mark_in_flight` returns only the
  unstarted ticket credits and never calls the backend;
- backend allocation plus backend-close failure preserves the allocation error
  and releases both pinned lifetime reservations;
- externally signalled and repeated cancellation retires after the late gather
  completes;
- cancellation retries after an event-query failure synchronize and retire the
  pending transfer;
- cancelling the same live allocation twice is idempotent while a copied
  allocation remains invalid.

The tests use controlled CPU fakes and make no real `torch_npu` execution claim.

### Scheduler and consumer failures must drain an unconsumed ticket

The consumer bridge originally released only after `ring.wait` returned. A wait
failure, or a scheduler boundary reached before the PLE layer consumed its rows,
left an async ticket active while poisoning only the bridge. The ring now exposes
`cancel_and_drain`: it blocks for a late CPU gather, retires completed host work,
and synchronizes launched transfer or consumer events before reuse. The bridge
uses it on prepare failures after admission, pre-consumer failures, and undrained
scheduler boundaries while preserving the primary exception. A backend release
failure during failed-gather retirement poisons the ring and retains credits.

Both synchronous fallback rows and asynchronous rows now travel through the
same batch-stamped wrapper. The bridge rejects a prior batch's synchronous rows,
checks the synchronous gather shape against the exact expected shape, and keeps
the original PLE call as the sole arithmetic and convolution-state consumer.

### Host byte credits cover the gather peak

For an `F`-byte all-distinct request, `ShardedBF16Rows.gather_bytes` can hold `F`
bytes of distinct cached rows while constructing the `F`-byte joined result. The
worker's `raw` variable refers to that same joined bytes object rather than a
third copy. Each ticket therefore reserves `2F` transient host bytes, `F` H2D
bytes, and no additional pinned bytes; the two persistent pinned slots retain
their separate lifetime reservations.

## Validation record

`python3 -m unittest migration_bridge.round9.tests.test_ple_async
migration_bridge.round9.tests.test_ple_runtime
migration_bridge.round9.tests.test_host_budget
migration_bridge.round9.tests.test_ple_rows` passes all 40 focused tests.

Repository-wide discovery ran 232 tests. The available host environment completed
the other tests without an assertion failure, but the overall command reported
three import errors in unrelated modules: `test_candidate_gguf` requires NumPy,
while `test_dynamic_tensor_adapter` and `test_qwen4_exp_overlay` require PyTorch.
Neither dependency is installed in the system Python used for this host-only
check.
