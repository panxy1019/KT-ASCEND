# D1 Target-Row Trace Preflight

Evidence state: `A3_VERIFIED` for a single real serving request. Full QD/HD
teacher-forced sweeps have not begun.

Parent `e3084d309d6f39b8a9f7185d36ac27e4b77526d7`, SGLang child
`eaa58ccea4ee372b8b207dd16f398eafe3da37ff`, D1 placement and frozen
Round 6 GGUF were used in the disposable A3 container. The installed Python
`kt_kernel.experts_base` had exactly the same SHA256 as the source file and
exposed `_trace_last_row_cpu_routes`.

One 8-token greedy request for `s6_en_01` ran under the opt-in JSONL trace.
The trace contained target-row immediate/deferred CPU expert IDs and routing
weights. The request produced one real deferred target-row route and ended
with no pending layer. Its 8-token output was exactly the prefix of the
previously frozen 64-token D1 trajectory.

This preflight only proves that target-row fields are produced and readable
under a real request. It does **not** establish a numerical envelope, held-out
validation, or quality qualification. The trace-based teacher-forced sweep
must still verify each request's own record span before reporting per-position
CPU/deferred hits.

Raw A3 artifacts:

```text
/home/admin/kt-artifacts/round6-deferred/d1-target-trace-smoke/
```

## Gate

```text
D1_TARGET_ROW_TRACE_PREFLIGHT = A3_VERIFIED
QD_MATCHED_HISTORY_SWEEP = NOT_RUN
HD_HELDOUT_VALIDATION = NOT_RUN
```
