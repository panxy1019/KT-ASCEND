# F10 request-local host page ownership core

Date: 2026-09-21

Status: `CPU_REQUEST_LOCAL_OWNERSHIP_ONLY`

Published record state: `HOST_PUBLISHED_NO_HBM_ACTION`

This module is a synchronous ownership capability for a future QSA cache
integration seam. It stores immutable host artifacts and proves which request
allocation owns them. It does not call an allocator, free or overwrite a
physical page, allocate NPU staging, update a page table, install a model hook,
or establish a real F10 device result. Its telemetry therefore reports zero
allocator free calls and zero claimed HBM bytes freed.

## Ownership identity

Every page is keyed by all three fields:

- nonempty `request_id`;
- nonnegative exact-integer `allocation_epoch`;
- bounded exact-integer `logical_page` in `0..32768`.

The allocation epoch prevents a late callback from an earlier allocation of
the same request ID from acquiring a newly published page. Invalidation also
requires the request ID and allocation epoch. It never acts on a different
epoch or request that happens to use the same logical page.

Each successful publication receives a monotonically increasing generation
for that exact key. Reload tickets and leases bind the key and generation.
Ticket and lease operations additionally require the exact object issued by
the store; a copied dataclass with equal fields is stale authority.

## Exact BF16 artifacts

`BF16LayerPayload` retains K and V as exact bytes. It performs no numeric
conversion. Version 1 fixes the actual TP1 geometry used by this groundwork:

- full page: 4 tokens;
- K/V heads: 2;
- head width: 256;
- bytes per BF16 element: 2;
- QSA model layers: `3, 7, 11, ..., 47`.

A layer artifact may represent a bounded 1-to-4-token tail so the format does
not erase valid-token information. Publication for eviction accepts only a
completed 4-token page and requires the ordered set of all 12 QSA layers.
Missing, duplicated, mis-keyed, corrupt, or partial layer sets never become
visible in the store.

Every layer binds canonical metadata, K bytes, and V bytes with SHA-256. The
page record binds its ownership identity, generation, state, and the 12 layer
digests with another SHA-256. Public records are revalidated on reads, with
exact integer and fixed-shape checks. A successful pin/read returns the same
immutable byte strings that were published.

## Atomic publication and host limits

`publish_export` validates all layer copies and digests before entering the
store. It then computes the immutable page record and checks both configured
limits while holding the store lock:

- maximum published page count;
- maximum canonical host artifact bytes.

If either limit would be exceeded, publication fails without changing page or
byte accounting. `host_bytes` covers the canonical metadata, K/V payloads, and
binary SHA-256 digests. It deliberately does not claim to measure Python object
overhead or transient caller-owned copy buffers.

## Pins, reload callbacks, and invalidation

A host pin is an exact lease. The page remains readable through that lease
while invalidation drains, and removal occurs only after the last lease is
released. New pins and reloads are rejected as soon as invalidation starts.

`begin_reload` issues a generation-bound callback ticket. Because this is the
CPU ownership core, `complete_reload` creates a
`CPU_STAGING_REFERENCE_ONLY` lease over the immutable host record; it performs
no device copy. A completion for an invalidating page is rejected and helps
drain the page. An aborted reload removes its ticket and also advances pending
invalidation. Reused tickets, copied tickets, and stale-generation callbacks
fail closed.

`invalidate(request_id, allocation_epoch)` marks every matching page as
invalidating. Pages without active pins or reload callbacks are removed
immediately. Others remain unavailable to new work until their exact leases
and tickets release or abort. The return value is true only when all pages
selected by that call have drained.

## Validation coverage

`tests/test_context_page_store.py` covers:

- exact BF16 K/V byte round trips across all 12 layers;
- canonical record and digest creation;
- incomplete, corrupt, mis-keyed, wrong-shape, and partial-page rejection;
- atomic page-count and canonical-byte caps;
- exact-object pin and release semantics;
- invalidation delayed by an active lease;
- generation-bound reload completion, stale completion, and abort;
- request and allocation-epoch isolation;
- strict keys, versions, shapes, capacities, leases, and tickets;
- explicit zero allocator-free and HBM-free claims.

These tests exercise the synchronous CPU ownership state machine. A later QSA
integration must separately bind real allocator authority, page-table updates,
copy completion events, device staging lifetime, scheduler cancellation, HBM
accounting, and end-to-end cache correctness before making an F10 claim.
