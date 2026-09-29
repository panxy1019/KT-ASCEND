# Round16-A current execution

Updated: `2026-09-28T23:52:21Z`

Status: `PASS_DEVICE_DRIVER_SUPERVISOR_AND_ANALYSIS_INPUT_OFFLINE_DEVICE_NOT_RUN`

## Fresh A1 real-start readiness audit

- The prior injected-driver placeholder has been replaced by `round16/a/fresh_a1_device.py` and `tools/round16_a1_container_driver.py`. One controller process issues the frozen six requests in exact order. Each item performs one HTTP POST, the ledger is spent before the call, and no retry loop exists.
- `tools/round16_a1_remote_block.py` now provides the host lifecycle: exact fresh admission SHA/physical12/freshness validation, one patched v295 launch, readiness wait, one container driver process, exact owned-container stop, guard ACK, removal, post-cleanup physical12 sample, and SHA archive manifest. It was not executed.
- Fresh block A/B launchers now use distinct containers `kt-round16-a1-a-v001` and `kt-round16-a1-b-v001`, distinct evidence directories, one model-start budget each, profile environment `ON`, and a six-attempt/zero-retry budget.
- Admission is fail closed: the caller supplies the exact new launcher-specific receipt SHA; host control and the patched launcher both require status `PASS`, the exact card6/chip0/physical12/PCI/NUMA identity, and current result/raw completion plus validation times no more than 600 seconds old. Host control also rehashes both wrapper-named raw rounds and their >=60-second ordered envelope. The launcher still performs its two immediate v295 resource samples around guard readiness.
- The runtime observer now follows the real v295 ownership shape (`module.quant_method`), requires Host expert bindings for every layer 0..47, records device events only at 0/24/47, and emits a binding receipt. Not-ready device events stay in a cross-batch pending queue and are queried later without `synchronize`.

The analysis-input blocker is closed by `round16/a/fresh_a1_evidence.py`. It verifies the A1 Host 0..47 binding receipt and the frozen Round13 QSA/GDN/expert binding receipt, copies only complete direct Round13 QSA/GDN/NPU-resident device-event observations for W1/W2 measurements, and forms a request Decode bounding interval from directly observed serial scheduler batch boundaries. A serialized `SHORT_DRAIN` Round13 batch after W2 is required as the FIFO flush witness; the final canary may remain device-event pending. The copied launcher binds Round13 to `/out/round13-spans.jsonl` and `/out/round13-bindings.json`; the separate A1 observer supplies the all-layer Host rows. The profile hook map is a boundary catalog, not installed wiring, and A1 does not install `NPU_SHARED`. Because the bounding interval does not isolate A2 Host wait or A3 launch/metadata cost, two otherwise complete blocks produce `A1_INSUFFICIENT_ATTRIBUTION`, `selected_candidate=null`, and A2/A3 `NOT_SELECTED_EVIDENCE_GAP`. CPU native work remains unknown; nested, parent/child, and cross-clock arithmetic remains forbidden.

The real launch path now locks the exact reviewed source-manifest SHA before the first launcher subprocess and passes the same SHA into the container driver. `execution-budget.json` is outside the immutable source-file set and has a separate transition contract: initial unspent state to exactly one matching full-container-ID start, with before/after SHA evidence. Launcher admission validation rehashes the result and both adjacent raw rounds, checks two >=60-second ordered rounds, exact envelope, physical12 identity, and <=600-second freshness for both raw completion and validation. The archive repeats raw SHA checks and records SHA/size for server, guard, and driver logs.

Cleanup covers running, terminal, and pre-start `created` containers. The created-state branch requires the exact full container ID and PID zero, terminates the recorded guard or confirms it already exited, and removes the container only after that guard state is safe. A mismatched live PID command remains fail closed.

## Delivered

- A0: runtime-neutral manifest, P0/P1/P2 classification, explicit/idempotent registration, and side-effect-free import.
- A1: QSA Decode contract with eager fallback selected before backend/state mutation.
- A2: GDN Prefill/Decode post-core normalization contract with preserved leased-state boundary.
- A3: existing resident grouped-MoE provider boundary, recorded as current provider rather than a new candidate.
- Metadata adaptation with tuple snapshots, completion/lease/write validation, a pre-execution identity snapshot that lets backend/fallback alias checks survive input-container mutation, and best-effort retryable lifecycle close.
- Exact accepted tensor envelope: one `hidden_states [1..128, 2048]`, BF16, ND tensor with omitted or `(2048, 1)` stride. Every unresolved shape/layout falls back before backend invocation.
- Actual runtime inputs are independently duck-typed for count/shape/dtype/layout/stride/device and must match the frozen spec. Mapping keys establish logical input names; sequence and single inputs use positional spec names, so standard torch/SGLang tensors do not need a `.name` attribute. Independent `runtime_stream_kind` metadata must match the declared stream. Mismatches raise `MetadataContractError` before backend or fallback selection.
- Raw runtime shape and stride dimensions must each satisfy `type(value) is int`; bool, float, and numeric strings are rejected without coercion before backend or fallback selection.
- Standard-library offline installer plus isolated-child import validation.
- Default-off post-stage P0/A1 offline map covering the real S0 startup chain, the accepted v335/v338 profile evidence, clock and event boundaries, CPU Expert/NPU/Host dispatch/sync/UNKNOWN attribution, bounded counters, and fail-closed candidate gates.
- Fresh A1 standard-library observer/control contract with self-described clock, parent, stream, event, and correlation identities; immutable A/B block IDs; a six-per-block/twelve-total ledger; and disabled, non-executable checked-in controller templates.
- Reviewable fresh-snapshot builder and offline preflight: eleven exact v295 source/helper/fixture/budget/monitor SHA preconditions are checked before copying, outputs inside the frozen parent are rejected, and only the copied tree receives scheduler/KTEP/launcher observer wiring. A full local trusted-donor fixture build and preflight covered 2,628 files and returned `INITIAL_UNSPENT`. The actual remote v295 parent has a historically spent budget, so the approved route is local A/B build plus fixed reviewed manifest/budget/ledger identities and whole-directory transport; remote self-build is rejected.
- Installed A1 observer boundaries cover routed-expert parent, KTEP CPU submit/sync, and Host merge after the existing main-stream `wait_event` for layers 0..47. Direct QSA/GDN/NPU-resident samples come from the already-verified Round13 attribution hooks. The broader hook map is only a boundary catalog; A1 does not install `NPU_SHARED`. Both observers add no synchronization and preserve wrapped returns and exception identity in CPU mocks.
- A gated controller function that rechecks the source-manifest and immutable ledger SHA immediately before creating output, spends the copy-on-write ledger before each injected driver call, accepts no retry/abort budget, supports sequential A=6 then B=6, and writes a raw receipt on successful completion. No device driver was invoked.
- Corrected six-item frozen queue per block: W1 warmup, W2 warmup, W1 measurement, W2 measurement, short drain, and fixed canary. Short drain inherits the S15 27/4 `canary_math391` request with `ignore_eos=true`; the independent fixed canary uses normal EOS, 32 maximum new tokens, exact text `391`, and finite logprob.
- Strict offline A1 evidence loading and comparison. The loader binds each raw receipt to observer metadata and JSONL bytes by SHA-256, and the comparator rejects missing requests/layers/events, duplicate event identity, pre-aggregated or nested spans, cross-clock addition, inserted synchronization, and request/token/source drift. Candidate gates use direct Host critical intervals divided by a direct Decode phase wall in the same Host clock domain; GDN and NPU Expert kernel durations are diagnostics only and cannot stand in for A2 or A3 evidence.
- The A2 gate requires exposed Host sync wait of at least 10% of Decode wall in both blocks, primary-region status in each block, and a same-layer unused independent NPU-window DAG proof. The A3 gate requires Host launch/metadata critical time of at least 10% in both blocks, primary-region status in each block, and proof that the region lies outside the old resident-only graph.
- Frozen dependency DAG and pending review outputs: `A1_CRITICAL_PATH.json`/`.md`, `A_REGION_DEPENDENCY_DAG.json`, and `A_CANDIDATE_DECISION.json`. Until both new immutable A/B blocks validate, their sole status is `DEVICE_EVIDENCE_PENDING`, candidate selection is null, and historical traces are ineligible.
- CPU Expert direct time stays `UNKNOWN_NATIVE_ABI_NO_CALLBACK` with a null duration and no candidate-score contribution. Host dispatch/sync/merge and the routed-expert parent remain inspectable non-additive records; no parent/child subtraction or addition is performed.

## Evidence

- Round16-A tests: 103 passed; all Round16 tests: 109 passed.
- Shared contract tests: 6 passed.
- Shared contract SHA-256: `7234dc017690eea39313b7a2776c54466cc896ed10294708970cdb2bda23aa01`.
- Direct Round14a v235 source paths, symbols, and hashes are in `PROVIDER_MANIFEST.json` and `RESOLVED_CODE_MAP_A.md`.
- Round15 provider-map and offline-capability documents are source/API evidence only. Round15 is frozen terminal as `ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION`; no S15 promotion is claimed.

## Execution boundary

The package does not import or execute SGLang, KTransformers, torch, torch_npu, CANN, or device APIs. It does not copy kernel implementations. A P2 backend and a P1 eager fallback must be explicitly injected by the host runtime.

The fallback branch is chosen from the immutable `ProviderSpec` and adapted metadata before calling the backend. Therefore unsupported shapes, phases, state formats, versions, dtypes, layouts, streams, or devices cannot enter provider state-writing code through this adapter.

The post-stage P0/A1 map is offline-only and default `OFF`. It records Round15 v335 A as salvaged raw evidence and v338 B as a clean final block. It does not treat the Round15 component gate as a Round16 promotion because the Round15 Decode-heavy W2 request gate did not show a gain. CPU Expert execution and cross-clock residuals remain explicitly unknown rather than being derived from overlapping parent/child spans.

The fresh A1 observer is also default `OFF`; its disabled path calls the wrapped operation directly without reading a clock or emitting a row. Host records use `HOST_MONOTONIC_NS`. Device records accept only an already-completed deferred event duration and never claim absolute Host alignment. The native async CPU worker exposes no start/end callback through `submit_forward`/`sync_forward`, so direct CPU Expert timing remains `UNKNOWN_NATIVE_ABI_NO_CALLBACK` with null timestamps and preserved correlation identity. No native ABI or production synchronization changed.

The checked-in controller input artifacts remain `enabled=false` and `executable=false`; each embeds the six-item queue plus its SHA-256, and execution rejects schedule mutation before calling the driver. Block B additionally requires an exact A=6/B=0 predecessor ledger with the six frozen ordered A request IDs, then binds the fixed A raw receipt, ledger-after SHA, PASS cleanup supervisor, physical12 post-sample, and final archive before any B subprocess. The initial shared ledger remains A=0, B=0, and total=0. CPU-only controller tests used temporary fixtures to prove six plus six accounting and raw-receipt formation; they did not start a model, import torch, or consume device budget.

Provider close marks the provider closed before touching dependencies, attempts every distinct dependency in the close pass, aggregates failures, and remembers only successful closes. Later calls retry failed resources without closing successful resources twice.

## Graph boundary

QSA and GDN graph support is not advertised. The A3 resident grouped-MoE operator identity is recorded as graph-stable evidence from Round15, while capture/replay eligibility remains a later B/device gate. No graph was created or captured.

## Not run

- Ascend/torch_npu import or device execution: `DEVICE_REQUIRED_NOT_RUN`.
- Numerical, model, request, and graph qualification: `NOT_RUN`.
- CPU or device performance measurement: `NOT_RUN`.
- Full local trusted-donor snapshot build/preflight: `PASS_CPU_ONLY` on 2,628 files with `INITIAL_UNSPENT`; no remote copy or model execution occurred. The two device-run snapshots must still be rebuilt locally into fresh final transport paths and frozen under externally reviewed manifest hashes before transfer.
- Fresh A1 device comparator execution: `DEVICE_EVIDENCE_PENDING`; both SHA-bound real raw receipts and normalized observer ledgers are absent, so A2/A3 remain unselected. Incomplete attribution must remain `A1_INSUFFICIENT_ATTRIBUTION`, never be rewritten as `NO_CANDIDATE`.
- Full model, full bank, SSH, shared-object changes, global environment changes, and top-level KTransformers changes: `NOT_RUN` / not performed.

## Residual risks

- Real SGLang registry API wiring is represented by a generic mapping or `get_provider/register_provider` adapter and requires P1 integration validation.
- Standard framework tensors may be passed in a logical-name mapping or positional sequence; the adapter reads shape/dtype/layout/stride/device through duck typing without importing torch.
- A real backend can mutate state internally before returning; host P2 code must honor the lease/write contract. This component guarantees unsupported requests never call it.
- Before metadata adaptation or execution, backend alias checking freezes the runtime input container and every original top-level mapping/sequence member into an immutable tuple. Backend/fallback mutation cannot remove an identity from this check. Framework-specific storage overlap requires tensor/runtime checks in the device gate and is `DEVICE_REQUIRED_NOT_RUN`.
- The inspected Round14a v235 model source differs from the later v295 hash recorded by Round15. No equivalence between snapshots is claimed.
