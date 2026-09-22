# Round9 qualification backlog after framework freeze

Date: 2026-09-21

This backlog begins only after the F0–F13 framework interfaces and identities are
frozen. It deliberately does not reopen the cancelled Round8 performance work.

## Q1 — quality

1. Freeze evaluation software, tokenizer, prompt template and CMMLU256 records.
2. Qualify BF16 Static Q64 as the model reference without inheriting any earlier
   model's status.
3. Compare MXFP4 Static Q64 and selected Dynamic candidates against that reference.
4. Admit Q32 only for candidates that meet the quality threshold.
5. Keep scheduler calibration DEV disjoint from HOLDOUT and final quality data.

## Q2 — performance

Run formal isolated ABBA only after quality candidates are selected:

- C64 versus C128;
- BF16 versus MXFP4, with Q8_0 as the fallback track;
- Q64 versus Q32;
- Uniform versus Frequency;
- request-boundary versus Prefill-End Dynamic;
- PLE Sync versus Async;
- Hybrid versus Layerwise/Streaming prefill.

Each comparison needs an idle-host admission, exact source/profile hashes, warmup,
request-level samples, resource telemetry and a predefined regression rule. A
framework smoke result must never be reused as a performance data point.

## Q3 — real context qualification

Execute 4k, 16k, 64k and 128k campaigns in order. At each length validate output,
QSA indexes, GDN state, PLE history, KV ownership, abort/reuse and resource headroom
before proceeding. Native FP8 KV, if later available, requires a separate oracle
against the BF16 reference. Integrate a real allocator adapter before measuring
HiCache capacity or latency.

## Q4 — deployment and advanced work

- concurrency and request isolation beyond the serial framework profile;
- multi-NPU and all-NPU placement/collective tests;
- long-duration restart and failure-recovery soak;
- graph capture, MTP and multimodal tracks;
- observability, operational rollback and externally reachable service controls.

The backlog is complete when evidence supports the distinct states
`PRODUCT_QUALIFIED`, `PERFORMANCE_QUALIFIED` and `DEPLOYMENT_QUALIFIED`; none of
those states is asserted by Framework Completion.

