# Round9 F12 — bounded feature interaction matrix

Date: 2026-09-21  
Scope: one or two request framework smoke; no quality or performance inference

## Matrix

| Expert scheduling | Prefill | PLE | Context | Evidence | Result |
|---|---|---|---|---|---|
| Q64 Static Uniform | Hybrid | Sync/OFF | BF16 baseline | full-model v36 | `PASS_FRAMEWORK_HTTP2`; feature-OFF oracle |
| Q64 Static Frequency | Hybrid | Sync | BF16 baseline | full-model v26/v27 | HOLDOUT, map replay, abort/reuse, clean restart PASS |
| Q32 Request Dynamic | Hybrid | Sync | BF16 baseline | full-model v30 | HOLDOUT, four generations, abort/reuse PASS |
| Q32 Prefill-End Dynamic | Layerwise | Sync | BF16 baseline | full-model v34 | HOLDOUT, four commits/drains, abort/reuse PASS |
| Q32 Prefill-End Dynamic | Layerwise | Async | BF16 baseline | full-model v45 | one request, 48 layers, four async drains, generation-1 commit/final drain and 391 PASS |
| Q64 Static Uniform | Hybrid | Async | BF16 baseline | full-model v35 | 11 async drains and HOLDOUT PASS |
| Q32 Request Dynamic + joint budget | Hybrid | Async | BF16 baseline | full-model v39 | exact persistent-baseline drains and abort/reuse PASS |
| Q32 Static | Streaming 32 | Async | BF16 compact QSA | full-model v43/v44 | HOLDOUT, clean restart and exact 32+5 chunk completion PASS; prompt mid-prefill HTTP interruption unqualified |
| Q64 Static | Hybrid | Sync | FP8 host record → BF16 QSA staging | full-model v41 | actual 4k pool, 11 QSA batches and HOLDOUT PASS |
| Static component | Hybrid | Sync | request-local host tier | context-tier v2 | device byte-identity PASS; full-model allocator integration NOT PRESENT |

## Interpretation

The matrix records exact combinations that executed. A component-only result is
not promoted to full-model PASS. In particular, the final row is the F10 tier
skeleton required by the framework plan; it cannot be called HiCache integration
because SGLang still owns the authoritative resident K/V pages.

The v35/v36 pair checks the PLE flag seam. The v39 row checks simultaneous real
request Dynamic and PLE traffic under one budget. The v43/v44 row checks the
three working-set planes at each SGLang chunk boundary. The v45 row exists because
passing Prefill-End Dynamic and async PLE separately does not prove their combined
lifecycle. v45 supplies that missing cell: the same request produces one complete
48-layer prefill record, four real async PLE drains, a generation-1 Prefill-End
commit, a final request drain and finite output exactly equal to 391. Peak owned
HBM is 26544 MiB and cgroup high/OOM/failcnt remain zero.

Every full-model row uses serial TP1, C128 and physical 15 under the exclusive
resource guard. Model dtype and QSA staging are BF16 unless the context column
states the FP8 host reference. CPU expert storage is MXFP4 for these bounded
framework interactions; that format choice is not a performance conclusion.
