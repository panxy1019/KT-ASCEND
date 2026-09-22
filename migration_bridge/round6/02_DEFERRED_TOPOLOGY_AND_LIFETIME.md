# Deferred Topology and Lifetime Audit

Evidence state: `CODE_INSPECTED`

## Finding T1 — sparse placement has no consumer

`create_kt_config()` returns `None` for an all-accelerator layer.  The parent
pending registry is keyed by physical layer index and the consumer checks only
`layer_idx - 1`.  Consequently a deferred task from CPU-enabled layer `L`
requires a KT wrapper at exactly `L+1`.

The existing P1/P2/P3 placements are mostly sparse.  Enabling the global CLI
flag on them would permit a pending contribution at an isolated layer without
an incremental consumer.

Required invariant:

```text
defer_allowed(L) = has_cpu_wrapper(L) and has_cpu_wrapper(L + 1)
```

Every contiguous CPU-enabled block terminal must use defer zero.  The old
P1/P2/P3 profiles remain defer-zero regressions.

## Finding T2 — model-final guard is insufficient

The current child disables defer only when `layer_idx == num_layers - 1`.
That is sufficient only when every earlier deferred layer has an immediately
following wrapper.  It does not protect the terminal layer of a sparse or
partial CPU-enabled block.

## Finding T3 — no-op pending needs a behavioral test

`submit_forward()` tests `deferred_ids is not None`, while other code paths test
whether any deferred ID is valid.  With defer enabled, the former can submit a
task and set pending even when the deferred set contains no CPU-executable
route.  This has not yet been proven at runtime because the CPU ownership mask
is applied in the backend.

Required behavior:

```text
valid CPU deferred work == 0
  -> no deferred task
  -> pending false
  -> allow_n_pending 0
```

## Finding T4 — class-level state scope

`BaseMoEWrapper._layer_has_pending_deferred` is a process-global class dict.
Construction initializes per-layer entries, and normal submit overwrites the
current layer entry.  There is no explicit request ID or model-instance key.

Round 6 v1 limits execution to one model, one scheduler request at a time, but
must still test sequential requests, exceptions and shutdown.  The result must
not be generalized to concurrent serving.

## Finding T5 — slot identity depends on contiguous parity

The buffer is shared by batch/dtype key and has depth two.  `L` writes deferred
output to `(L+1)%2`, which is the current slot for adjacent `L+1`.  This is the
mechanism enabling incremental merge.  A missing adjacent consumer or early
slot reuse violates the design.

## Finding T6 — numerical contract changes

Deferred deliberately changes when a routed contribution becomes visible in
the residual stream.  Round 4A's pairwise contract remains a hard regression
for defer zero.  Defer one needs QD derivation, a frozen candidate contract and
independent HD validation, plus downstream quality.

## Implemented topology decision

SGLang child commit `eaa58ccea4ee372b8b207dd16f398eafe3da37ff`
implements the minimal topology-aware per-layer rule:

```text
requested defer > 0
and wrapper(L + 1) exists
  -> preserve requested budget

otherwise
  -> layer-local defer = 0
```

The A3 D1 lifecycle trace proved that this was an incorrect pending task.  In
the pre-fix four-prompt run, layer 7 marked all 134 submissions pending even
though only 13 contained a CPU-owned deferred route; layer 8 consequently ran
incremental on all 134 submissions.  Commit `0fbbc572b1b43894a3205c39f24a92132ebf302b`
now suppresses empty CPU-deferred work after the Ascend path's existing D2H
visibility wait.  The same workload produced exactly 13 pending events and
exactly 13 adjacent layer-8 incremental consumers, with zero false or missed
pending events.  CUDA retains its previous asynchronous behavior.

## Gate

```text
R6_G1_TOPOLOGY = GUARDED_IN_SGLANG_CHILD
R6_G1_EMPTY_PENDING = A3_VERIFIED
```
