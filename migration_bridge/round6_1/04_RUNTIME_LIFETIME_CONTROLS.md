# Round 6.1 runtime and lifetime controls

Evidence state: `A3_VERIFIED` plus `CODE_INSPECTED`.

## Buffer immutability

For all relevant layer-7 and layer-8 internal calls, live pinned buffers were
captured immediately after host visibility and again after CPUInfer sync.

| Buffer class | Layers | Calls | Result |
| --- | --- | ---: | --- |
| hidden input | 7, 8 | 7 paired calls | exact |
| immediate IDs | 7, 8 | 7 paired calls | exact |
| deferred IDs | 7 | 4 paired calls | exact |
| routing weights | 7, 8 | 7 paired calls | exact |

In total, all 28 paired live-buffer checks are byte-value exact. No host-side
mutation of an in-use input, ID, or weight region was observed.

## Slot and pending ownership

For the 54-row target call, trace sequence numbers and monotonic timestamps show:

```text
17  L7 submit: current slot 1, next slot 0, outgoing pending true
19  L7 sync begins with diagnostic full drain
20  L7 sync complete
21  L8 submit: current slot 0, incremental_from_previous true
22  previous pending consumed: producer L7 -> consumer L8
23  pending state becomes empty
25  L8 sync complete
```

This establishes one producer group, one completion barrier, one incremental
consumer, and one clear transition for the target request. The native CPUInfer
API does not expose a stable task identifier, so `trace_sequence` plus
layer/slot is the audit identity; the report does not claim a nonexistent
native task ID.

The completed production deferred tensor also matches the explicit replay
tensor exactly, which rules out stale output or an overwrite before consume for
this case.

## Ascend ordering

Static inspection and A3 trace confirm that the active path uses framework
managed tensor copies and explicit device visibility before host CPU work:

```text
hidden / IDs / weights D2H: Tensor.copy_(..., non_blocking=True)
host visibility:            explicit device wait on NPU path
CPU completion:             CPUInfer sync
CPU output H2D:             Tensor.copy_(..., non_blocking=True)
```

No raw `acl.rt.memcpy` transfer was introduced by Round 6.1. The host callback
does not launch arbitrary nested NPU work.

## Schedule perturbation

`KT_R61_CAPTURE_DRAIN_DEFERRED=1` forces the normally pending deferred CPU task
to complete at the producer sync boundary solely for capture, while retaining
the same delayed visibility at L8. Its final logits remain exact against normal
production Deferred and the explicit no-pending replay. This is the safe
schedule perturbation used here.

Direct reversal of the shared worker-pool queue was not attempted because it
would alter global task ordering beyond the mathematical work under test. The
result is recorded as `NOT_RUN_UNSAFE_DIAGNOSTIC`, not silently called PASS.

## Instrumentation incident

The first control launch failed before producing evidence because the new
after-sync capture referenced underscored local names. The opt-in diagnostic
was fixed, syntax checked, reinstalled in the disposable venv, and the entire
control was rerun. No failed-run tensors are used by the result.

## Evidence

- Control verdict: `evidence/runtime_controls/result.json`
- Trace: `evidence/runtime_controls/trace.jsonl`
- Cross-check against B/C: `evidence/runtime_controls/stagewise_comparison.json`
- Raw tensor index: `evidence/runtime_controls/raw_sha256sums.txt`

```text
BUFFER_IMMUTABILITY             = PASS
PENDING_SLOT_ORDERING           = PASS
ASCEND_COPY_ORDERING            = PASS
SAFE_SCHEDULE_PERTURBATION      = PASS
DIRECT_WORKER_QUEUE_REVERSAL    = NOT_RUN_UNSAFE_DIAGNOSTIC
ROUND6_1_RUNTIME_LIFETIME_CONTROLS = PASS
```
