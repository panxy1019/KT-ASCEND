# KT-ASCEND Round9 Framework Completion Report

Date: 2026-09-21  
Model: Qwen3.8-Flash-Next, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`  
Branch: `codex/round9-heterogeneous-working-set`  
Execution host: `110.123.0.3`, physical Ascend device 15, exclusive mode  
Framework source baseline: commit `dc52a63caf7449441d49e94dc204e84eeccd1c5e`  
Frozen implementation commit: `53902048d35ad4cc6e7fda2ef41706552a03bb88`

## Result definition

Round9 Framework Completion freezes the principal expert, PLE and context-plane
interfaces with default-OFF feature flags, fail-closed identity/lifecycle rules,
CPU tests, bounded device probes and selected full-model HTTP smoke. The result
does not include CMMLU256, formal performance, real 64k/128k generation,
production concurrency or deployment soak.

The final status uses six separate levels:

1. `IMPLEMENTED` — source and contract exist;
2. `LOCAL_QUALIFIED` — deterministic CPU/native lifecycle gates pass;
3. `DEVICE_COMPONENT_QUALIFIED` — bounded physical-15 component probe passes;
4. `FULL_MODEL_SMOKE` — selected real model requests pass;
5. `PRODUCT_QUALIFIED` — deferred;
6. `PERFORMANCE_QUALIFIED` and `DEPLOYMENT_QUALIFIED` — deferred.

## Frozen foundation

- C128 uses four NUMA pools and 32 physical workers per pool. Each pool produces
  an FP32 partial for one 160-wide slice; the deterministic merge returns to the
  BF16 model path.
- The complete 48-layer CPU expert bank exists in BF16, Q8_0 and MXFP4. Q8_0 and
  MXFP4 remain CPU backing formats; every NPU promotion reads the original BF16
  checkpoint.
- Q64 and Q32 are actual distinct NPU physical capacities, with 64 or 32 resident
  tensors per layer and exact logical/reverse mapping validation.
- The full model path contains MoE Top-10, QSA, GDN, Gated Residual and host-backed
  PLE. Existing synchronous and feature-OFF paths remain available.

## F0–F13 outcome

| Phase | Deliverable | Highest bounded status | Principal evidence |
|---|---|---|---|
| F0 | Exact baseline identities | FROZEN | `FRAMEWORK_BASELINE_MANIFEST.json` |
| F1 | Qwen3.8 Static Frequency | FULL_MODEL_SMOKE | v25 route producer; v26/v27 HOLDOUT, abort/reuse and restart |
| F2 | Layerwise expert transport | FULL_MODEL_SMOKE | device probe v2; full-model v32/v34 |
| F3 | Complete prefill route telemetry | FULL_MODEL_SMOKE as F4 producer | 48-layer count and weight-mass records in v34 |
| F4 | Prefill-End Dynamic | FULL_MODEL_SMOKE | v34, four commits/drains and active abort/reuse |
| F5 | PLE async exact-row consumer | FULL_MODEL_SMOKE | component v2; full-model v35; feature-OFF v36 |
| F6 | Expert + PLE host budget | FULL_MODEL_SMOKE | component v3; full-model v39 |
| F7 | Q32 physical capacity | FULL_MODEL_SMOKE | component v19; full-model v29/v30/v39 |
| F8 | QSA context sizing/plumbing | 4K FULL_MODEL_SMOKE; larger lengths LOCAL_QUALIFIED | v41 actual 4096 pool plus synthetic boundary tests |
| F9 | KV quantization capability/reference | FULL_MODEL_SMOKE reference; native UNSUPPORTED | component v4; FP8 host record to BF16 QSA staging v41 |
| F10 | Request-local host context tier | DEVICE_COMPONENT_QUALIFIED | context-tier v2 byte-identity spill/reload |
| F11 | Three-plane Streaming Prefill | FULL_MODEL_SMOKE with mid-prefill HTTP-abort limitation | v43 HOLDOUT/multi-chunk; v44 clean restart and ordinary abort/reuse |
| F12 | Feature interaction matrix | BOUNDED_MATRIX_PASS | `ROUND9_CAPABILITY_MATRIX.json` |
| F13 | Framework freeze | COMPLETE within this report's definition | final source/test/evidence digests |

## Scheduling and prefill plane

F1 adds Qwen3.8-specific DEV and HOLDOUT corpora, original Top-10 route capture,
per-layer counts and weight mass, deterministic tie-breaks and replayable Q64/Q32
masks. Frequency placement is tied to model, tokenizer, source, corpus and initial
resident-map identity. The full-model restart reproduces the same maps and output
token trajectory.

F2 introduces a private BF16 layerwise transport with a measured 32-token bound.
F3 requires complete contiguous input spans and all 48 real MoE layers before a
summary is usable. F4 reuses the existing promotion/demotion engine and adds one
generation transaction across prefill drain, BF16 migration, device synchronize
and atomic map publication. Request, placement and allocation epochs are checked
separately.

The bounded v34 run completes HOLDOUT2, integer canary 391, four real Prefill-End
transactions and active abort/reuse. The first sampled request data remains on a
stable generation, and the final request-boundary drain owns retirement.

## PLE and joint host-work plane

F5 retains synchronous PLE as the reference and adds a two-slot async ring. Every
slot binds request and generation identity, token span, row IDs, pinned host
storage, device storage, a side-stream event and explicit state. Duplicate and
cross-shard rows preserve exact original BF16 identity. Cancellation drains the
transfer event before reuse.

Full-model v35 executes 11 real async drains and passes HOLDOUT2. v36 disables the
feature and matches v35 output token IDs with maximum output-logprob difference
0.0; no async event file exists in v36.

F6 adds `HostWorkBudget` credits for pinned, host and H2D bytes and for expert and
PLE work. Current-token correctness traffic has priority over migration needed by
decode, which has priority over speculative PLE prefetch. v39 combines Q32
request-boundary Dynamic, async PLE and the joint budget. Four requests return to
the exact persistent ring baseline with no in-flight or cancelled work.

F12 v45 separately exercises the dependency-sensitive combination that cannot be
inferred from those individual gates: one Q32 Layerwise Prefill-End Dynamic
request produces a complete 48-layer route record, four async PLE drains, a
generation-1 commit, a final request drain and integer output 391. Peak owned HBM
is 26544 MiB; cgroup high/OOM/failcnt stay zero.

## Context plane

F8 derives KV, QSA-index, GDN, page-metadata and PLE-history bounds for 4k, 16k,
64k and 128k profiles. Boundary arithmetic, overflow rejection and lifecycle are
locally tested. v41 performs the actual 4096-token pool allocation; the larger
lengths are not real generation claims.

F9 records the current Ascend QSA capability decision: no demonstrated native
FP8 KV contract. The implemented reference stores canonical host FP8 pages with
scale metadata, decodes selected rows to bounded BF16 device staging and calls the
real sparse QSA consumer. v41 completes 11 such QSA batches, two request drains,
finite HOLDOUT2 output and the integer canary.

F10 adds immutable host page records keyed by request, allocation epoch and logical
page. The physical-15 v2 probe moves all 12 QSA-layer K/V payloads from physical
page generation `(0, 0)` to host and back to `(1, 1)` with exact byte identity,
request isolation and drained leases. No SGLang allocator adapter is installed,
so the full model retains its HBM K/V authority and every context event records
`hbm_release=false`.

## Streaming integration

F11 binds a serial request, placement epoch, allocation epoch and contiguous
32-token chunks to one barrier. A chunk may publish only after the 48-layer expert
plane, async PLE ticket and 12-layer QSA reference plane all drain. Gaps, duplicate
completion, request switches or epoch changes fail closed.

v43 runs C128, MXFP4, Q32, Layerwise, async PLE, BF16 compact QSA staging and an
actual 4096-token pool on physical 15. HOLDOUT2 passes, including strict 391. A
forced input larger than one scheduler chunk proves two contiguous complete
chunks and entry into the third. HTTP abort returns 200 twice, but the scheduler
continues to queue chunks; this is retained as an explicit limitation rather than
called a pass. v44 provides the clean restart, a bounded complete two-chunk
request and the ordinary post-prefill active abort/reuse gate. This validates the
coordinated barriers and restart path, not prompt mid-prefill interruption,
overlap or speed.

## Problems found and repaired

The sprint preserves every failed attempt instead of overwriting it:

- CPU package v2 omitted historical fixtures and a native header. v3 added only
  the missing package inputs; later source changes received fresh CPU snapshots.
- Full-model v28 assumed the primary logical map was CPU. The real loader created
  it on NPU; validation now records CPU or NPU for that primary map while retaining
  the strict NPU inference map.
- Full-model v31 changed FusedMoE wrapper parameter names. Restoring `layer` and
  `dispatch_output` preserved keyword dispatch.
- An external training PID appeared during v32; the exclusive guard stopped only
  the owned container. The test resumed only after fresh exclusive admission.
- The historical 80 GiB-per-node gate blocked MXFP4 despite large clean file-cache
  headroom. A storage-scoped policy retains 80 GiB for BF16/Q8_0 and uses 48 GiB
  for MXFP4 plus an 8 GiB conservative runtime floor. Trial34 qualified this exact
  scope without global cache drops.
- v38 treated persistent PLE ring reservations as a leak. The fixed invariant is
  exact return to the immutable post-construction baseline; v39 passes it.
- v40 treated idle scheduler callbacks as missing active leases. Tests now return
  only for inactive idle state and retain fail-closed behavior for a lost active
  lease; v41 passes.
- v42 used the wrong overlay path and failed before model start. The immutable
  evidence directory then correctly rejected reuse. v43 uses a new version and
  the exact v16 overlay.

Detailed phase notes are in documents 43 through 60 and
`55_FRAMEWORK_GATE_PROGRESS_AND_REPAIRS.md`.

## Verification freeze

- Final CPU source snapshot: `framework-cpu-v16`, 331 tests passed, installer
  passed against a fresh preserved SGLang copy, no NPU device mapped.
- The later declarative F12 combination profile has no implementation-code delta;
  its exact schema parses locally and executes in full-model v45.
- Local static verification: Python compilation, `git diff --check` and 19
  targeted configuration/streaming/NUMA tests pass.
- Physical-15 runs use an exclusive container, one mapped device (`davinci15`),
  C128 CPU admission, 640 GiB cgroup memory limit and resource monitor.
- The final matrix and capability JSON bind evidence paths and SHA256 digests.
- Old Round7 timers remain paused. No Round8 performance work was restored.

## Freeze statement

`ROUND9_FRAMEWORK_COMPLETE = YES` means F0–F13 framework contracts are frozen at
the bounded levels above. `PRODUCT_QUALITY`, `PERFORMANCE`, `REAL_LONG_CONTEXT`
and `DEPLOYMENT` remain `PENDING`. Native FP8 QSA and integrated HiCache remain
unsupported, and the request-local tier skeleton is reported at its actual device
component status.
