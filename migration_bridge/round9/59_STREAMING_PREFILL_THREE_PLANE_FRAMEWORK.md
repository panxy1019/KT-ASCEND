# Round9 F11 — bounded three-plane streaming prefill framework

Date: 2026-09-21

## Contract

Streaming is a separate default-off profile. It can bind only when all three
working-set planes are already bound to the same serial scheduler:

- 48-layer expert transport;
- exact-input asynchronous PLE consumer;
- 12-layer QSA reference consumer.

The streaming threshold must equal the layerwise prefill threshold and the chunk
size is fixed at the previously measured 32-token bound. The launcher, server
admission and model hook each verify the profile SHA independently.

## Lifecycle

For each scheduler EXTEND batch the coordinator binds:

```text
request object
+ placement generation
+ context allocation epoch
+ total input span
+ contiguous (start, count) chunk
```

The shared chunk barrier publishes only after:

1. the expert plane completes all 48 layers;
2. the PLE ticket and transfer event drain;
3. the context reference completes all 12 QSA layers.

The next chunk must start exactly at the previous `start + count`. A changed
request object, placement generation, allocation epoch, total span, gap,
duplicate plane completion or missing drain fails closed. Abort clears only the
matching request identity. A request cannot retire before the final chunk and all
three plane barriers complete.

## Qualification boundary

This framework coordinates existing bounded planes. It does not claim that SGLang
has released QSA HBM pages between chunks, and the context barrier records
`hbm_release=false` explicitly. It also does not claim performance overlap.

The final v16 CPU snapshot passes 331 tests and installs the scheduler/model
hooks into a fresh copy of the preserved SGLang baseline. It also fixes an idle
scheduler hook bug: an idle iteration now returns only when the coordinator is
inactive, while an active coordinator without a matching lease still fails
closed.

Physical-15 v43 uses the exact v16 source with C128, MXFP4, real Q32 physical
capacity, Layerwise threshold 4, async PLE, a BF16 compact QSA reference consumer,
actual 4096-token pool sizing and a 32-token SGLang chunk bound. HOLDOUT2 passes
with finite outputs, integer canary 391 and exact per-request three-plane drains.
A forced 193-token input completes two contiguous chunks with identical request
and epoch identity and enters the third. Two HTTP abort calls return 200, but the
upper scheduler continues queuing prefill chunks instead of promptly delivering
an abort to the coordinator. The observation is preserved as a limitation, not
an abort PASS, and v43 is stopped under operator control. v44 cleanly restarts,
repeats integer canary 391, completes an exact 37-token request as contiguous
`(0,32)` and `(32,5)` chunks with exact three-plane drains, then passes the
ordinary post-prefill active abort/reuse gate. This paragraph must not be read as a performance,
overlap, HBM-release or prompt mid-prefill-interrupt claim.
