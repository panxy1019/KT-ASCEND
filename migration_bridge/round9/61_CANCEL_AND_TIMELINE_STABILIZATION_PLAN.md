# Round9 P0 Cancel Closure and P1 Timeline Execution Plan

Date: 2026-09-21
Baseline: `6f091b1b8ebcddabc9746e8b44b4aec72c9a248c`
Scope: P0 mid-prefill cancellation closure and P1 unified bounded timeline only.

## Constraints

- All features default OFF and fail closed.
- Do not implement Context Tier allocator integration, FP8 KV authority, performance optimization, CMMLU, formal performance, or real long-context qualification.
- Do not alter historical evidence snapshots.
- Use fresh generated overlays and fresh evidence directories.
- Physical15 remains the only NPU target and requires fresh admission before every model process.
- Do not restore Round7 timers, merge main, tag, or push without a later explicit delivery request.

## Task 1 — P0 observable control path

Add bounded cancellation lifecycle events and isolated overlay seams for HTTP receipt, TokenizerManager send/no-send, Scheduler receipt and matched request identity. Preserve existing API behavior. Prove default-OFF behavior, exact identities and installer idempotence with RED/GREEN tests.

## Task 2 — P0 submit fence and cleanup state machine

Using Task1 evidence, reproduce v43 locally where possible and on physical15. Identify the first divergent boundary. Add the smallest cancellation fence needed so CANCEL_SEEN prevents new normal chunk/prefetch/decode/result publication. Preserve safe completion of already-submitted work and fail closed on uncertain migration state. Add local fault tests before implementation.

## Task 3 — P0 full-model qualification

Generate a fresh overlay and run one bounded three-chunk cancellation request on physical15, followed by resource-baseline and reuse checks. Run one Prefill-End Dynamic boundary negative if the static path passes. Preserve every failure and repair as fresh evidence.

## Task 4 — P1 unified bounded timeline

Add a default-OFF bounded in-memory timeline for scheduler chunks, PLE gather/H2D/wait/consumer and expert shadow read/transform/H2D/compute identities. Keep host and device clock domains explicit. Batch-flush at request end; do not write one file per high-frequency event. Add RED/GREEN tests for bounds, dropped counts, identity fields and OFF regression.

## Task 5 — P1 bounded profile and decision

Run a small physical15 profile without changing launch/wait or shadow reuse behavior. Record critical wait, transfer bytes, legal overlap observations and repeated copy candidates. Select at most one later optimization or explicitly retain the current implementation.

## Task 6 — verification and documentation

Run the complete CPU suite, fresh overlay install/compile checks, diff review and machine-readable capability update. Document root cause, failed attempts, repair, evidence scope and remaining limitations. No product/performance claims.
