# Round9 F9/F10 — QSA reference consumer and request-local context tier

Date: 2026-09-21

## Scope

This checkpoint separates two context-plane questions that were previously mixed:

1. can an actual QSA call consume a bounded compact K/V staging tensor; and
2. can a completed request-local page move to an immutable host record and later
   return under a new physical-page generation without losing identity?

Neither feature is enabled by default. Neither currently claims native FP8 QSA,
HBM savings, long-context qualification, or HiCache compatibility.

## F9 implementation

`context_reference_runtime.py` implements two explicit reference modes:

- `BF16_COMPACT`: copy only selected BF16 rows through bounded host snapshots and
  compact device staging before calling the original sparse QSA function.
- `FP8_HOST_REFERENCE`: encode resident BF16 pages to the canonical host FP8
  record, decode selected rows to BF16 staging, then call the original sparse QSA
  function.

The model seam remains a conditional expression around the original
`sparse_gqa(q, keys, values, selected)` call. With the feature flag absent, the
original call is executed unchanged. The runtime requires one serial request,
all 12 ordered QSA layers, a fresh allocation epoch for reused request IDs, and a
complete per-batch layer drain. Any lifecycle failure poisons the runtime as
`BROKEN_RELOAD_REQUIRED`.

The reference path deliberately retains the original HBM K/V tensors as the
authority. Its workspace is reserved before the real KV/GDN pool profiler runs.
This makes the correctness experiment bounded, but it does not reduce resident
KV capacity.

## F10 implementation

`context_page_store.py` owns canonical immutable host records keyed by:

```text
(request_id, allocation_epoch, logical_page)
```

Each record contains exactly the 12 QSA-layer BF16 page payloads, canonical
metadata, per-layer digests, one page generation and explicit leases/reload
tickets. Only complete four-token pages can be published.

`context_tier.py` adds a synchronous request-local state machine:

```text
RESIDENT
  -> export all 12 layers
  -> atomically publish host record
  -> clear logical mapping
  -> release exact (physical_page, physical_generation)
SPILLED
  -> acquire exact host-generation reload lease
  -> allocate a new physical-page generation
  -> import and publish mapping
RESIDENT
```

The active tail page and its predecessor cannot be evicted. A failure before host
publication leaves the resident mapping authoritative. A failure after host
publication has no safe per-page rollback in this skeleton and changes the tier
to `BROKEN_RELOAD_REQUIRED`.

## Validation

The v9 CPU snapshot passed 317 tests and installed a fresh overlay containing the
actual Torch QSA reference tests. The final v16 CPU source snapshot includes the
tier, streaming work and idle-scheduler repairs; it passes 331 tests plus a fresh
overlay installation. Both CPU containers had no NPU device mapped.

The local context reference tests execute real Torch BF16 QSA and verify exact
BF16 compact output. The host-FP8/BF16-stage result is finite and remains within
the fixture-specific relative-L2 bound. The request-local tier tests verify
byte-digest identity after spill/reload into a different physical generation,
tail protection, duplicate-owner rejection, request isolation, abort cleanup and
fail-closed exporter/importer/mapping failures.

The physical-15 context-tier v2 component gate then moved one complete BF16 page
for all 12 QSA layers from physical page `(0, 0)` to an immutable host record and
back to `(1, 1)`. Every K/V byte digest matched, request isolation held, all
leases drained and peak additional allocation was 110592 bytes. This is a
component state-machine gate; it does not hook the SGLang page allocator.

Full-model v41 bound `FP8_HOST_REFERENCE` to the actual 12-layer QSA consumer,
allocated the measured 4096-token pool, completed 11 QSA reference batches and
two exact request drains, and passed HOLDOUT2 with finite outputs and integer
canary 391. The first v40 attempt is retained as a failure: an idle scheduler
iteration was incorrectly treated as an unleased active request. A regression
test reproduced the distinction, and v16 now returns on an idle/inactive hook
while continuing to fail closed when an active runtime loses its lease.

## Qualification boundary

- install a real KV allocator/mapping adapter before claiming any HBM release;
- qualify native FP8 only if the Ascend QSA backend exposes a compatible contract;
- keep HiCache naming out of status until an actual tier backend is installed.

Accordingly, F8 context sizing and F9 FP8-host-reference consumption have bounded
full-model smoke. F10 has a device-qualified request-local tier skeleton only.
`hbm_release=false` remains part of every full-model context event.
