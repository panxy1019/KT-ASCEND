# Round16 current execution

Updated: 2026-09-28 12:54 UTC.

Status: `ROUND16_DEV_READY_WAIT_ROUND15`.

Round16 has completed the device-free development stage defined by contracts
60, 61, and 62. The shared protocol is frozen as `round16-contract-v2`; v2
supersedes the pre-freeze v1 development snapshot because final review added
deep immutability, type checks, and a complete completion-boundary contract.

## A — SGLang + KT provider

`round16.a` is an installable standard-library package containing the common
contract and the A carrier only. It provides explicit, idempotent registration,
P0/P1/P2 capability descriptions, strict support/fallback selection, actual
runtime tensor and stream validation, lease/write/alias checks, and best-effort
retryable close. Mapping keys supply logical tensor names, so ordinary
torch/SGLang tensors do not need a synthetic `.name` attribute. QSA, GDN, PLE,
and MoE compute remain owned by the injected S15 provider; A does not duplicate
their algorithms.

A result: `OFFLINE_COMPONENTS_READY_DEVICE_PENDING`; 30 A tests and 14 shared
contract tests pass. Ascend imports, real weights, numerical checks, complete
model, graph capture, and performance remain `NOT_RUN`.

## B — vLLM-Ascend + KT Expert bridge

`round16.b` is a separate standard-library install containing common plus B.
It freezes Top-10 once, splits global expert IDs by Q32, merges routed weights
and shared output once, plans Q32 allocation before allocation, validates all
512 logical-to-physical entries, preserves supported aliases, rejects
unprovable fused/alias ownership before allocation, and rolls back allocations
best-effort without hiding the primary error. Lifecycle identity covers request,
allocation, batch, layer, token range, completion, slot, replay epoch, and input
fingerprint. Cancellation and NPU/CPU submission failures retire pending tasks.

The real vLLM 0.26 constructor/worker API is deliberately unresolved.
Unverified registration returns `VERSION_API_UNRESOLVED` without changing the
registry; only an explicitly injected `VerifiedModelBinding` can register the
CPU-tested dynamic wrapper. This is an offline registry-shape scaffold, not a
qualified model plugin.

B result: `OFFLINE_BRIDGE_SCAFFOLD_READY_VERSION_DEVICE_PENDING`; 45 B tests
and 14 shared contract tests pass. Real vLLM construction, torch_npu, NPU
execution, weights, graph capture, numerical qualification, and performance
remain `NOT_RUN`.

## Verification and barrier

The combined suite passes 95/95: A 30, B 45, common 14, and Round15 barrier 6.
All Python files compile; JSON artifacts parse; isolated A/B installation and
fresh imports create no runtime threads, change no environment variables, and
load none of torch, torch_npu, SGLang, KTransformers, or vLLM. The six promoted
Round14a v295 source-lineage hashes match their files.

Round15 is currently `P0_COMPLETE_A_COMPONENT_SCREEN_NEXT`; no terminal receipt or
released resource lease exists. The barrier therefore remains
`CLOSED_WAIT_ROUND15`. Round16 used zero NPU calls, model groups, generations,
abort controls, device component processes, NPU hours, or CPU performance
benchmarks. It did not access or mutate the remote runtime.

The user authorized selecting another safe NPU and compatible NUMA set for the
later VALIDATE phase. No card or NUMA is selected now. After Round15 becomes
terminal, cleanup passes, and its lease is released, Round16 must recover S15's
latest caps and perform fresh process, HBM, CPU/SMT, NUMA, host-memory, training
identity, and unknown-process admission before choosing that resource.

No background watcher or timer is installed. No push, merge, tag, reset, clean,
cache retirement, or Round15 environment/shared-library mutation was performed.
