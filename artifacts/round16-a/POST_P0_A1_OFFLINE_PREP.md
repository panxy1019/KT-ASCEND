# Round16-A post-stage P0/A1 offline preparation

Updated: `2026-09-28T17:13:06Z`

Status: `READY_OFFLINE_MAP_DEVICE_REQUIRED_NOT_RUN`

The new path is default `OFF`. This preparation only restores bounded Round15 evidence and maps a future Round16 device-stage contract. It did not rerun Round15, start a model, contact a remote container, expose an NPU, or change the shared contract.

## Frozen Round15 disposition

Round15 ended as `ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION`. Its P0 summary accepted two fresh measurement blocks and selected QSA index selection as the first A1 screen. The A1 component gate passed, but its Decode-heavy W2 request screen did not demonstrate request gain. `FULL_SET_CAUSAL_SHORT_CONTEXT` therefore remains `OFF`; it is evidence of a screened component, not a Round16 promotion.

## Real S0 startup chain

The actual model process does not start in the profile driver. The immutable v338 chain is:

1. `round15_p0_controller.main` verifies the snapshot, claims the controller ledger, and invokes `source/tools/launch_hybrid.py`.
2. `launch_hybrid.py` creates the monitored S0 container and accounts the one allowed model start.
3. The container entrypoint executes `/out/source/runtime/candidate_server.py` with the frozen server, resolved-profile, and admission documents.
4. `round15_p0_supervisor.main` waits for the real server-ready marker.
5. The supervisor starts exactly one `round15_p0_driver.run` using `docker exec`, checks the four-generation ledger, then retires the owned container.

The exact paths and SHA-256 values are in `POST_P0_A1_MAP.json`. A future Round16 stage may reuse this control pattern, but it needs new immutable block identities, fresh ledgers, and a separate approved execution budget. The Round15 role table ends at v338 and must not be replayed as if it were a new run.

## Two accepted profile blocks

| Block | Evidence | Status | Boundary |
|---|---|---|---|
| A / v335 | `p0-results/v335-control-failure/.../round15-p0-raw.json` | `PASS_ROUND15_P0_PROFILE_DRIVER_RAW` | Fresh measurements were salvaged after a control-only failure. Root `startup` and `cleanup` fields are absent, so this is not a clean final process document. |
| B / v338 | `p0-results/v338/.../round15-p0-final.json` | `PASS_ROUND15_P0_PROFILE_PROCESS` | Clean final block with startup and cleanup evidence. |

Each block contains four generation attempts in the exact `WARMUP`, `W1`, `W2`, `DRAIN` order, 150 observed scheduler batches, and 3,600 spans. Each has 450 observations for `CPU_SUBMIT`, `CPU_RESULT_WAIT`, `NPU_RESIDENT`, and `ROUTED_EXPERT`, plus the QSA/GDN/PLE sampled points recorded in the JSON map.

## Events and clock domains

| Domain | Producer | Meaning | Arithmetic boundary |
|---|---|---|---|
| `HOST_MONOTONIC_NS` | `time.monotonic_ns()` | Batch and Python-call start/end/duration within one process and boot. | Absolute values are not comparable across fresh processes. |
| `NPU_EVENT_ELAPSED_MS` | `torch.npu.Event(enable_timing=True).elapsed_time()` | Relative device duration for one recorded event pair. Query is deferred to a later scheduler boundary. | It has no absolute alignment with host monotonic time. No synchronize is inserted. |

The observer explicitly forbids parent/child summation. `ROUTED_EXPERT` overlaps its child work, and the CPU/NPU paths execute concurrently.

## CPU Expert, NPU, dispatch, sync, and UNKNOWN

| Bucket | Existing Round15 record point | What it proves |
|---|---|---|
| CPU Expert | No direct record | The asynchronous CPU worker has no worker-clock start/end event. Its compute duration and overlap remain unknown. |
| NPU | `NPU_RESIDENT` around `gpu_method.apply` | Host call duration plus deferred NPU-event elapsed duration for the sampled resident NPU call. |
| Host dispatch | `CPU_SUBMIT` around `_submit_cpu_forward` | Host-side submission-call duration only. It is not CPU Expert compute time. |
| Host sync | `CPU_RESULT_WAIT` around `_sync_cpu_forward` | Host-side wait/retrieval-call duration only. |
| Mixed parent | `ROUTED_EXPERT` around `KTEPWrapperMethod.apply` | End-to-end sampled parent observation with overlapping children. |
| UNKNOWN | Explicit unmeasured bucket | Async CPU work, unwrapped mask/remap/merge/stream waits, and every cross-clock residual. It must never be calculated as parent minus children. |

The raw span schema also lacks `clock_domain`, stream/event IDs, parent span IDs, absolute device timestamps, and CPU-worker correlation timestamps. Source inspection identifies the current clock producers, but does not manufacture those missing raw fields.

## Budget counters

The restored Round15 P0 shape used two fresh model processes and four generations per accepted block. The newly frozen Round16 fresh A1 contract uses two new immutable block IDs, six generation attempts per block, twelve attempts total, zero abort controls, and zero automatic retries. The exact six requests are `W1_WARMUP`, `W2_WARMUP`, `W1_MEASUREMENT`, `W2_MEASUREMENT`, `SHORT_DRAIN`, and `FIXED_CANARY`. The historical Round15 request-direction screen used one fresh baseline and one fresh candidate model process, 14 generation attempts and one abort control per arm; that remains reference evidence and is not added to the Round16 twelve-generation contract.

This offline preparation consumed zero model processes, generations, aborts, components, and device hours. Repair or retry budget is never implied.

## Candidate selection gate

The restored P0 order is `QSA_INDEXER`, followed by GDN RMSNorm only if A1 is inapplicable. The Round15 QSA component gate required both A/B blocks, 12 rows per block across layers 3/23/47 and contexts 32/64/128/256, exact selected sets, downstream relative L2 at most 0.001, three identical repeat hashes, every candidate/baseline median ratio at most 0.85, lifecycle/resource/source identity checks, and at most 10 GiB peak NPU allocation.

Passing that component gate opened only the separate `DECODE_W2` request gate. Round15 failed that request-level promotion gate, so the Round16 selected candidate is `null` and the path remains disabled. A later stage must provide new Round16 P0 blocks, component evidence, and separately authorized request evidence before changing this disposition.

The frozen Round16 comparator accepts only the two new immutable block identities and rejects historical trace substitution. It verifies all six request identities and token contracts, complete sampled QSA/GDN/expert events, one source identity, no observer synchronization, and SHA-bound raw/observer files. It uses only direct Host critical-region intervals divided by a direct Decode wall in `HOST_MONOTONIC_NS`; it never adds parent/child spans or Host/NPU clocks. GDN and NPU Expert kernel durations are diagnostic and cannot substitute for A2/A3 evidence. A2 additionally requires a same-layer unused independent NPU-window DAG proof; A3 requires a Host launch/metadata region outside the old resident-only graph. Missing native CPU callbacks remain `UNKNOWN_NATIVE_ABI_NO_CALLBACK`, not zero. With the two real blocks absent, `A1_CRITICAL_PATH.json` and `A_CANDIDATE_DECISION.json` stay `DEVICE_EVIDENCE_PENDING` with a null selection.

## Exact missing evidence

- v335 A raw evidence lacks root `startup` and `cleanup`; it must not stand in for a new clean Round16 final block.
- Both raw span ledgers lack `CPU_EXPERT.start_ns/end_ns`.
- Both raw span ledgers lack self-described `span.clock_domain`.
- Both raw span ledgers lack `stream_id`, `event_id`, `parent_span_id`, absolute device timestamps, and CPU-worker correlation timestamps.
- No Round16 fresh block A/B final result exists.
- The immutable A/B IDs and twelve-generation offline contract now exist, but no v295/S15-derived device snapshot build, preflight, shared-ledger result, or controller result exists.

These omissions keep candidate selection and execution closed. They are not relaxed by the Round15 terminal status.
