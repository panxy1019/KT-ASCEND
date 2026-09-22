# Round9 P0 Cancel Closure and P1 Timeline Results

Date: 2026-09-21
Baseline: `6f091b1b8ebcddabc9746e8b44b4aec72c9a248c`
Task-1 trace commit: `841f3e2789`
Scope: P0 mid-prefill cancellation closure and P1 unified bounded profiling only.

## 1. Qualification boundary

This sprint does not add Context Tier allocator integration, FP8 KV authority,
CMMLU, formal performance comparison, or real long-context qualification. Both
new facilities remain default OFF. A bounded profile is evidence about operation
ordering and host-observed duration; it is not a product or performance claim.

Current state:

| Item | State | Evidence |
|---|---|---|
| HTTP → tokenizer → Scheduler cancel trace | LOCAL QUALIFIED + FULL-MODEL ROOT-CAUSE TRACE | v50 |
| Submit fence | FULL-MODEL SMOKE | v58 and v61 stop before start 64 |
| Direct drained-chunk cleanup without another forward | FULL-MODEL SMOKE | v58 and v61 terminal abort |
| Post-cancel reuse | FULL-MODEL SMOKE | v58 and v61 same-process reuse |
| Unified bounded timeline | FULL-MODEL PROFILE | v61, 2,043 events, zero drops |
| Timeline-driven optimization | NO CHANGE SELECTED | CPU gather window exists; no H2D/expert overlap or repeated shadow identity |

## 2. v50 root-cause result

Evidence root: `/mnt/data1/kt-round9/evidence/20260920-full-model-v50`.

The abort was not lost between HTTP and the tokenizer manager. The exact host
monotonic sequence was:

| Event | `monotonic_ns` | Delta |
|---|---:|---:|
| second chunk normal submit, start 32 | 1048708260517047 | reference |
| HTTP abort received | 1048708311374719 | +50.858 ms |
| control sent | 1048708311670873 | +0.296 ms after HTTP receipt |
| Scheduler abort received | 1049065541799501 | +357.230 s after control send |
| `CANCEL_SEEN` | 1049065541926547 | +0.127 ms |
| submit fence published | 1049065541983460 | +0.057 ms |
| next streaming chunk rejected | 1049065545157410 | +3.174 ms |

The serial scheduler did not poll the control channel while the already-submitted
32-token chunk occupied `run_batch`. It accepted cancel at the next chunk
boundary. This establishes control polling starvation until a safe boundary; it
does not establish a dropped message.

The second failure occurred after the fence worked. Upstream SGLang cancellation
sets `FINISH_ABORT` and intentionally runs one more cheap forward so its ordinary
result path performs cleanup. The Round9 streaming coordinator had already
started and drained context, PLE and expert ownership for the prior chunk, then
correctly rejected the extra post-cancel chunk. As a result, upstream never
reached its terminal output path and the HTTP generation thread remained alive.

The first probe labelled this state `OBSERVED_CANCEL_SEEN`; that label was too
weak because `generation_thread_alive` was true. The repaired probe reports
`FAIL_TERMINAL_RESPONSE_TIMEOUT` and requires terminal abort plus reuse before it
can pass.

## 3. P0 implementation

`ROUND9_CANCEL_TRACE_PATH` is observation-only. It emits
`CANCEL_OBSERVED_NO_FENCE` and `POST_CANCEL_SUBMIT_OBSERVED` and does not change
submission behavior. `ROUND9_CANCEL_CLOSURE_ENABLE=1` is a separate strict switch
that publishes the epoch fence and rejects post-cancel normal submission.

The cancellation linearization point is Scheduler acceptance of the cancel epoch,
recorded as `CANCEL_SEEN`. Work already submitted to the device may drain. After
that event, the same request epoch cannot submit another streaming chunk.

After upstream `Scheduler.abort_request` assigns `FINISH_ABORT`, the closure path
handles the exact drained `chunked_req` without another model forward:

1. Convert `to_finish` into the terminal `finished_reason`.
2. Reject cleanup if expert, PLE, context or streaming work is still active.
3. Retire inactive expert transport ownership.
4. Retire PLE credits when present and clear PLE request ownership even when the
   synchronous fallback created no async credit.
5. Abort context and streaming ownership.
6. Release KV and GDN state through SGLang's existing `release_kv_cache` path.
7. Set completion time and publish the terminal abort response.
8. Clear `chunked_req`, retire the cancel identity, and emit
   `CANCEL_CLEANUP_COMMITTED`.

The final scheduler boundary now has a separate fail-closed check before
Dynamic, Context, PLE, Expert or timeline hooks run. It is a backup invariant:
the normal path removes the drained `chunked_req` inside `abort_request`, so a
later batch should never reach this check. Trace-only mode does not enable the
fence and retains the original serving behavior.

The source audit also closed two lifecycle leaks before the physical rerun. A
normally completed traced request now retires its strong request-object identity,
and cancellation retires PLE credits through either the PLE-local budget or the
joint host budget, checking that a joint budget returned to its
post-construction baseline. KV release imports its authoritative implementation
directly from `sglang.srt.mem_cache.common`.

Any exception emits `CANCEL_CLEANUP_FAILED` and is re-raised. The path does not
restore an old mask or publish partially copied weights. Prefill-End Dynamic is
outside the current static full-model retry; its existing transaction semantics
still require complete migration or `BROKEN_RELOAD_REQUIRED`.

The full-model driver now requires all of the following:

- `CANCEL_SEEN` and `CANCEL_CLEANUP_COMMITTED`;
- no chunk whose start exceeds the cancel trigger;
- `STREAMING_ABORTED`;
- an HTTP terminal event with finish reason `abort`;
- the generation thread exits;
- a fresh short request succeeds in the same process.

## 4. Full-model attempts after root-cause isolation

| Run | Result | Interpretation |
|---|---|---|
| v49 | pre-start CPU-idle admission failed | Resource gate; no model result |
| v50 | fence worked; terminal cleanup hung | Root cause above |
| v51 | NUMA2 below the MXFP4 48 GiB floor | Resource gate; no model result |
| v52 | guard stopped on unbound PID `1119298` | External process entered physical15 |
| v53 | guard stopped on training PID `1194144` | External termination before Scheduler saw cancel |
| v54 | source SHA mismatch at layerwise gate | P1 initially changed a device-gated file; no container started |
| v55 | server reached ready, then guard stopped on new training PID `1327122` | Correct fail-closed identity handling in shared mode |
| v56 first launch attempt | second admission saw NUMA2 below 48 GiB | Training restarted between targeted cache advice and launch; no container was created |
| v56 | guard stopped during load on unbound PID `1548350` | Training entered after an idle preflight; no request result |
| v57 | guard stopped during second chunk on unbound PID `1610700` | Abort HTTP/control were sent, but Scheduler could not see cancel before the external guard stop |
| v58 | `PASS_MIDPREFILL_CANCEL_AND_REUSE` | First uninterrupted P0 closure and same-process reuse under actual shared physical15 operation |
| v59 | timeline observer saw `backend.adapter=None` before preallocation | P1 instrumentation startup bug; no request ran, `OOMKilled=false` |
| v60 | timeline observer attempted to replace a read-only slotted adapter method | P1 instrumentation startup bug; no request ran, `OOMKilled=false` |
| v61 | P0 PASS plus `PASS_BOUNDED_TIMELINE_PROFILE` | Two requests, 2,043 timeline events, zero drops; bounded P1 decision evidence |

v52, v53, v55, v56 and v57 were not OOM failures. Their containers were stopped
by the ownership guard, and the unrelated training processes were left
untouched. v57 reached second-chunk admission and sent abort 11.92 ms later, but
the guard stopped it before the in-flight chunk drained; `FAIL_CANCEL_NOT_SEEN`
is therefore an external interruption rather than a P0 result.

v59 and v60 were also not resource failures. Both completed model weight loading,
then failed before server readiness in the new default-OFF observation path. v59
proved that the private expert adapter is created by
`LayerwiseTransport.preallocate()`, so the observer must bind after transport
construction. v60 proved that the qualified `_CopyOnlyAdapter` is intentionally
slotted and read-only. The repair uses an outer forwarding proxy rather than
modifying the qualified adapter or any device-gated core. Both failures became
regression tests before v61 was built.

## 5. Restored physical15 shared mode

The user restored shared-mode authorization on 2026-09-21. The full-model
launcher now has explicit `shared` and `exclusive` contracts. Shared mode uses:

| Budget | MiB |
|---|---:|
| User-attested training-process peak | 15360 |
| Candidate allocator cap | 37376 |
| Candidate process cap | 38400 |
| Absolute board cap | 56832 |
| Minimum board reserve | 8704 |

The guard admits exact training PIDs observed during preflight. A new unknown PID
causes only the owned Round9 container to stop. v55, v56 and v57 showed that PID
binding alone is too narrow because the same long-lived training/evaluation
container launches short-lived accelerator children after an idle preflight.

Read-only `/proc` evidence for v57 PID `1610700` established the stable boundary:

```text
container name: /mzy-train
container id:   0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef
started:        2026-09-10T03:21:04Z
parent chain:   run_beam_fine.sh -> eval_decode_search.py
```

The next shared launcher therefore resolves the explicit container name to its
full 64-hex ID. Admission and the live guard accept rotating physical15 PIDs only
when `/proc/<pid>/cgroup` exactly matches that ID. A recreated container receives
a new ID and is rejected; processes outside it remain unbound and stop only the
owned Round9 container. All of the existing 15 GiB training-process, 56,832 MiB
board, 38,400 MiB owned-process and 8,704 MiB reserve limits remain unchanged.

v58 exercised this contract while `/mzy-train` had a real physical15 child. Its
observed peaks were 30,498 MiB total board HBM, 1,358 MiB training-process HBM,
and 26,454 MiB Round9-owned process HBM. No guard limit was approached. v61 used
the same shared contract, while the training container happened to have no
active NPU child during the measured requests; its peaks were 29,220 MiB total
board HBM and 26,454 MiB Round9-owned process HBM. Therefore v58 is the evidence
for actual concurrency, while v61 is the evidence for the P1 timeline semantics
under the restored shared-mode admission and guard.

Shared mode does not weaken the MXFP4 NUMA contract. The current floor remains
48 GiB genuinely free per selected node, with conservative reclaim checked
separately. No global `drop_caches` is used. Targeted
`POSIX_FADV_DONTNEED` is limited to project bank/checkpoint files and every attempt
is recorded. A bank-cache eviction attempted while training was active blocked
for several minutes and was interrupted; the threshold was not lowered.

During the later v56 window, training was absent and a project-bank-only advice
completed successfully for 49 files / 64,186,240,119 bytes. NUMA2 rose from
31,423,128 KiB to 50,420,416 KiB. The launcher's independent sample then saw
training restart and NUMA2 fall below 48 GiB, so it rejected the launch before
container creation. The transient reclaim value was not treated as admission.

## 6. P1 timeline design

`ROUND9_TIMELINE_PATH` enables one serial request recorder. Its identity is:

```text
request_id + placement_epoch + allocation_epoch
```

The event count is bounded by `ROUND9_TIMELINE_MAX_EVENTS` in `1..65536`. Excess
events increment `dropped_events`. The recorder writes and `fsync`s one JSON
document when the request completes or aborts; it does not write one file per
expert operation. A short request below the selected streaming threshold has no
timeline lease, and its terminal flush is a no-op.

Every timestamp declares one of two domains:

- `HOST_MONOTONIC` for scheduler and host lifecycle observations;
- `HOST_OBSERVED_DEVICE_OPERATION` for host enqueue/wait observations around NPU
  operations.

The profiler never subtracts a host clock from an NPU clock and does not claim a
kernel-level device duration.

The event families are:

| Plane | Events |
|---|---|
| Scheduler | `batch_begin`, `batch_end` |
| Expert | `shadow_prepare`, `compute_enqueue`, `device_wait` |
| PLE | `gather_submit`, `sync_gather`, `gather_wait_and_h2d_dependency`, `h2d_enqueue`, `consumer_dependency_enqueue`, `consumer_enqueue`, `transfer_consumer_retire_wait` |

The first implementation instrumented `layerwise.py`, `dynamic_adapter.py` and
`ple_async.py` directly. v54 correctly rejected it because those exact files are
bound into existing device-gate receipts. The final design restores their bytes
to the frozen qualified versions and attaches observers in `prefill_binding.py`,
`ple_binding.py` and `ple_runtime.py`. The profiler therefore observes the
qualified core without silently inheriting a gate from different source.

Work between `gather_submit` and
`gather_wait_and_h2d_dependency` can demonstrate overlap with CPU row gathering.
It does not prove H2D overlap. H2D/expert overlap is reported separately by
checking expert work between `h2d_enqueue` and
`transfer_consumer_retire_wait`.

## 7. v58 P0 closure result

Evidence root: `/mnt/data1/kt-round9/evidence/20260920-full-model-v58`.

The result is `PASS_MIDPREFILL_CANCEL_AND_REUSE`. The request admitted chunks at
start 0 and start 32 only. HTTP abort arrived 8.869 ms after start-32 admission,
and the tokenizer control message followed 0.130 ms later. Because execution is
serial, Scheduler accepted the control only after the already-submitted chunk
drained, 360.407 s after send. From that safe boundary:

| Transition | Host-observed delta |
|---|---:|
| Scheduler receive → `CANCEL_SEEN` | 0.127 ms |
| `CANCEL_SEEN` → fence publication | 0.062 ms |
| fence → request retirement | 29.853 ms |
| retirement → cleanup commit | 0.108 ms |

The streaming trace contains `STREAMING_ABORTED` at next start 64 and no start-64
submission. The terminal response has finish reason `abort`, and a five-token
reuse request completed in the same server process at allocation epoch 1.

## 8. v61 unified timeline result

Evidence root: `/mnt/data1/kt-round9/evidence/20260920-full-model-v61`.

v61 repeated the P0 result and produced the physical P1 profile. HTTP-to-control
send took 0.144 ms. Scheduler accepted cancel after the in-flight chunk drained;
then `CANCEL_SEEN` took 0.126 ms, fence publication 0.062 ms, retirement
41.283 ms, and cleanup commit 0.108 ms. The same-process reuse request again
completed.

The summary reports:

| Field | Result |
|---|---:|
| Status | `PASS_BOUNDED_TIMELINE_PROFILE` |
| Requests | 2 |
| Events | 2,043 |
| Dropped events | 0 |
| Scheduler batches | 3 complete begin/end pairs |
| Expert shadow prepares | 673 |
| Host-observed shadow bytes | 187,986,739,200 |
| PLE launches/consumers | 3 / 3 |
| Independent work before PLE wait | observed |
| H2D/expert compute overlap | not observed |
| Repeated complete shadow identities | 0 |

The large shadow-byte total is traffic over three serial batches, not resident
HBM. It is also not a performance comparison: device work is measured from host
enqueue/wait boundaries, and the shared run has no ABBA control. The profile is
valid for ordering and candidate selection because it contains all three planes,
complete scheduler pairing, declared clock domains, and zero drops.

Evidence hashes:

| Artifact | SHA256 |
|---|---|
| `midprefill-cancel-result.json` | `ad68bedae8b191e174d2ce2611e114e39c387df01cb863ad560222141d963bcb` |
| `working-set-timeline.jsonl` | `6eac0e98c55955d70ad890bfd8b43e5027e97357448b83c82032c4015cac5840` |
| `working-set-timeline-summary.json` | `f331871c0a8c7b0e0a3d847db9408097a88865a6823513b300b4a4823a252d46` |
| `monitor.jsonl` | `dc883bd8f0354df0d26a9263f3e03b7a02136364424987d7ed0413ab9e81f665` |

## 9. Optimization decision

The physical trace resolves the bounded P1 decision:

- PLE has a legal window between CPU gather submission and its dependency wait.
  This can justify a later experiment that moves only the existing legal gather
  earlier.
- No expert operation appears between H2D enqueue and transfer/consumer retire.
  The current trace therefore does not prove a useful H2D overlap window.
- No complete shadow identity repeats in a legal layer span. The trace does not
  select shadow reuse, changed-slot copying, or next-layer staging.

No optimization is added in this stabilization sprint. PLE Async remains default
OFF. A later, separately gated P1 experiment may move the existing launch/wait
only if it preserves exact row identity and can show real H2D overlap; the
current evidence does not authorize deeper prefetch or shadow caching.

## 10. Verification

- `layerwise.py`, `dynamic_adapter.py` and `ple_async.py` remain byte-identical to
  baseline `6f091b1b8`.
- Python compilation and `git diff --check` pass for the modified files.
- Final host discovery ran 332 tests: 326 passed, 3 skipped, and the same 3
  NumPy/Torch import-only modules errored. There were no assertion failures.
- The frozen Torch container completed all 13 tests from
  `test_candidate_gguf`, `test_dynamic_tensor_adapter` and
  `test_qwen4_exp_overlay` without NPU device mounts.
- Fourteen focused P1 timeline/prefill tests pass in that same frozen container,
  including both physical startup regressions found by v59 and v60.
- The final observation build is preserved at
  `/mnt/data1/kt-round9/framework-dev-20260921-p1-timeline-v6`. Its receipt SHA256
  is `0a00966bcd9f96dca6b39e87474ce39fd902d65de46ce03c65c8855fdab250d5`,
  and its source tree SHA256 is
  `8f68fb1157353c0e95c2e52ad93c8a2136d1d0a7dce8867324a3d17f0d6f2099`.
- v58 and v61 both return `PASS_MIDPREFILL_CANCEL_AND_REUSE`; v61 also returns
  `PASS_BOUNDED_TIMELINE_PROFILE` with zero dropped events.

## 11. Closed scope and remaining qualification

The approved stabilization scope is complete:

```text
P0_CANCEL_CLOSURE_FULL_MODEL = FULL_MODEL_SMOKE
P1_TIMELINE_PHYSICAL_PROFILE = FULL_MODEL_PROFILE
P1_OPTIMIZATION_DECISION = NO_CHANGE_SELECTED
PRODUCT_QUALITY = PENDING
PERFORMANCE_QUALIFICATION = PENDING
```

The last two lines remain pending by design. This sprint did not run CMMLU,
formal ABBA, real long context, or deployment soak, and it does not promote the
profile into a performance claim.
