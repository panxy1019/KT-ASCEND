# Round9 framework known limitations

Date: 2026-09-21

## Qualification status

Framework Completion provides implemented interfaces, local tests, device
component gates and bounded full-model smoke. It does not grant product quality,
performance, long-context or deployment qualification. CMMLU256, formal ABBA,
large repeat matrices, 64k/128k generation and deployment soak remain deferred
by the approved framework-first scope.

## Expert and prefill plane

- Frequency masks come from the small independent Qwen3.8 DEV corpus and are
  checked on HOLDOUT. They are deterministic framework evidence, not proof that
  Frequency improves quality or latency.
- Q32 and Q64 are distinct physical allocations. Only selected bounded
  combinations have full-model smoke; there is no all-storage/all-strategy
  Cartesian qualification.
- Layerwise uses a private BF16 expert shadow and a maximum 32-token workspace.
  It does not claim overlap or a production transport policy.
- Prefill-End Dynamic commits after complete prefill statistics and before later
  decode work under a generation transaction. Injected failure paths are locally
  tested; the bounded full-model run does not exhaust every failure injection.

## PLE and shared host budget

- PLE async preserves exact requested BF16 row identity through a two-slot
  bounded ring, side stream and events. The full-model gate proves engagement and
  drain, not latency overlap or speedup.
- The joint budget accounts pinned bytes, H2D bytes, PLE work and expert migration.
  Persistent ring reservations are an explicit baseline. The policy uses static
  correctness priorities; it is not an optimized bandwidth scheduler.
- The system does not yet measure a stable per-NUMA DRAM bandwidth service level.

## Context plane

- 4k is actually allocated in the full model. 16k/64k/128k are configuration and
  boundary-arithmetic gates only; no real long generation is claimed.
- `FP8_HOST_REFERENCE` stores canonical FP8 host records and decodes selected rows
  into bounded BF16 NPU staging before real QSA. Native Ascend FP8 QSA remains
  unsupported by the demonstrated backend contract.
- The current QSA reference retains original HBM K/V as authority and records
  `hbm_release=false`; it produces no KV-capacity saving.
- The request-local context tier passes byte-identity spill/reload on a device
  component probe, but is not connected to the SGLang KV allocator. It must not
  be described as integrated HiCache.
- Prefix-cache and multi-request sharing are disabled in these serial gates.

## Streaming, scale and deployment

- Streaming is serial (`max_running_requests=1`), 32-token bounded, and coordinates
  completion barriers. It does not overlap planes and does not release QSA HBM
  pages between chunks.
- v43 shows that an HTTP abort acknowledged during a long chunked prefill is not
  promptly delivered to the coordinator; later prefill chunks can still be
  queued. Coordinator abort cleanup is locally tested and the ordinary
  post-prefill active abort/reuse gate is exercised separately, but prompt HTTP
  interruption of an in-progress multi-chunk prefill is not qualified.
- The runtime is TP1 on one physical Ascend 910 device. Multi-NPU, graph capture,
  MTP and multimodal paths are outside this freeze.
- Resource admission is exact-host and storage-specific. MXFP4 uses a provisional
  48 GiB startup floor on NUMA nodes 2/3/6/7 plus the conservative runtime guard;
  BF16 and Q8_0 retain 80 GiB. The 48 GiB result is not portable to other hosts or
  workloads without fresh monitored validation.
- The model server is intentionally isolated with no external network. These runs
  do not certify an external service, concurrency, availability or production
  security posture.
