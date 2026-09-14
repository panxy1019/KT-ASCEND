# Lifecycle and Error Paths

## Ownership audit

| Object | Allocation | Owner/consumer | Release |
|---|---|---|---|
| submission trampoline state | `make_unique` on submitting host thread | CANN callback after successful launch | callback-entry `unique_ptr` |
| `SyncArgs` for ordinary `sync()` | stack | calling host thread | scope exit |
| `SyncArgs` for stream sync | `make_unique` on submitting host thread | CANN callback after successful launch | callback-entry `unique_ptr` |
| MoE task parameters | binding task factory | submit callback, then copied into `TaskQueue` closure | callback-entry `unique_ptr` |
| pinned transfer tensors | Python | Python pipeline/test | after stream and CPU completion |
| stream | torch_npu/Python | caller | after explicit stream synchronize |
| CPUInfer | Python | caller | after callback and task-queue synchronization |

The pre-existing unreleased `new SyncArgs` paths and MoE callback-argument leaks were fixed separately in commit `d2d8be0`. Callback exceptions are captured without crossing the C ABI and surfaced on a host thread.

## Stress and negative paths

| Case | Result |
|---|---|
| create stream → callback → sync → destroy, ×100 | PASS |
| create CPUInfer → callback pipeline → destroy, ×20 | PASS |
| null stream | rejected with `ValueError` before CANN launch |
| CANN launch error | adapter test returns a non-success code and verifies surfaced API/code/location |
| callback C++ exception | captured and rethrown on host thread |
| invalid transfer buffer | explicit `ValueError` |

Lifecycle module: `4 passed in 16.07s`. Full-pipeline and callback stress provide the additional 1,000-cycle/10,000-callback stability evidence.

Evidence: `lifecycle_rerun.log`, `vendor-adapter-tests.log`, `ascend_full_matrix.log`.
