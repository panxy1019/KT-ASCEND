# Round9 Framework Completion — approved execution plan

Date: 2026-09-21. User approved the reviewed revisions and continuous execution.
Baseline documentation commit: dc52a63caf7449441d49e94dc204e84eeccd1c5e.
Resource evidence anchor: 25ccdda5d. Do not reinterpret historical trial results.

## Scope and completion

Implement the working-set framework with real consumers and bounded gates.
Default OFF for every new feature. No CMMLU256, formal performance, real64k/128k
generation, production soak, multi-NPU benchmark, timer restart, main merge/tag.
Physical15 on110.123.0.3 remains exclusive subject to fresh live admission.
Existing unrelated containers and the frozen baseline remain untouched.
Use fresh source overlays/evidence directories for device tests.

Framework completion requires implementation, local correctness and bounded
full-model integration where applicable for all stages below. Unsupported native
KV quant is an allowed capability decision only with a real reference path.
Placeholders, fallback-only execution and host tests do not grant device PASS.
Current BF16 request-boundary Dynamic strictJSON failure remains a known failure.

## Approved dependency order and deliverables

| Order | Phase | Deliverable and gate |
|---|---|---|
| 1 | F0 | FRAMEWORK_BASELINE_MANIFEST.json: commit/model/bank/native/overlay/profile identities, six historical scopes and failures; verify SHA |
| 2 | F0a | Separate request identity, placement epoch and allocation epoch; resident-capacity/schema contract; budget and cancellation ownership; negative host tests |
| 3 | F3a/F1 | Shared logical-route telemetry, DEV/HOLDOUT corpus, deterministic per-layer frequency masks Q64/Q32, source-bound statistics; real collector and static loading, restart sample |
| 4 | F7 | Q32 physical allocation, map/mask/merge, distinct gates from Q64; clean-process tensor-byte/HBM measurement and short static/Dynamic smoke |
| 5 | F2/F3b | Bounded real layerwise expert transport; hybrid remains reference; token-span/phase completeness; 48-layer engagement and abort/fallback evidence |
| 6 | F4 | Prefill-end barrier preserves live KV/GDN/PLE state, reuses migration engine, epoch transitions; copy/sync/abort/stale/incomplete-stat negatives; short differential smoke |
| 7 | F5 | Two-slot pinned PLE ring, async host gather and device transfer/events; sync exact rows, cross-shard/duplicate/abort/late completion/reuse and real consumer |
| 8 | F6 | Joint outstanding-host/H2D/pinned budget, priority and cancellation accounting; bounded competition/no correctness deadlock; no bandwidth or prefetch fairness claim without proof |
| 9 | F8 | QSA context plans4k/16k/64k/128k, page/index ownership and synthetic boundaries; no real-long-context qualification |
| 10 | F9/F10 | Audited KV quant native capability and bounded reference storage; request-local KV tiering, page ownership/byte identity; prefix sharing rejected until recurrent-state support |
| 11 | F11 | Streaming execution distinct from existing chunked prefill, QSA/PLE/expert lifecycle and abort integration |
| 12 | F12 | Short interaction matrix, feature-OFF regressions, unsupported-combination rejection, Q32+Dynamic; receipt per exact combination |
| 13 | F13 | Completion report, capability matrix, flags, limitations, qualification backlog; fresh independent review and bound evidence |

## Cross-stage contracts / rulings

- F1/F3 share one bounded route-stat schema; frozen trace replay is exact, live
  corpus repeatability additionally binds tokenizer/runtime/placement/sampling.
- Frequency v1 fixes resident count per layer; no global cross-layer redistribution.
- F0a exposes capacity early; no Q32 launch permission until F7 device admission.
- Prefill-end waits for placement consumers but preserves live request state;
  request-end releases it. The existing finish() is not a prefill barrier.
- First output token may be sampled from prefill logits; record the epoch boundary.
- Layerwise uses separately owned bounded staging, not silently overwritten resident
  slots. Explicit evidence of actual transport is mandatory; fallback is not PASS.
- Async cancellation retires buffers only after in-flight events complete. Request
  identity, placement epoch and buffer allocation epoch have different lifetimes.
- PLE decode row IDs become exact after sampling; unknown future tokens are not
  presented as exact prefetch. Prefill inputs can be prefetched ahead.
- Byte credits bound outstanding resources, not DRAM bandwidth. Required traffic
  must progress; optional prefetch may be cancelled. Strict priority is not fairness.
- KV/QSA index/GDN/PLE state jointly define request/prefix identity. Request-local
  tiering does not imply general HiCache or prefix-sharing qualification.
- Finite output alone is insufficient: short differential gates against sync,
  hybrid or same-placement reference accompany new algorithm paths.
- v10 and later trials have different overlays. A unified new candidate requires
  its own feature-OFF smoke; six old passes are not one binary's qualification.

## Evidence and reporting

Each stage records implementation/local/system/content/restart/resource/quality/
performance/deployment separately, scoped to exact feature combination and SHA.
Tests start RED before implementation, then GREEN; retain failure/recovery notes.
Minutes are a target for bounded probes, not a guarantee for full-model loads.
Full-model source identity is frozen per process; changed code needs a fresh load.
Progress and rulings: FRAMEWORK_PROGRESS.md. Existing execution history stays intact.
