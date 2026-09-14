# Ascend Stream Handle

## A3 observation

| Property | Result |
|---|---|
| Python type | `torch_npu.npu.streams.Stream` |
| public property | `npu_stream` |
| representation | non-zero Python integer, stable across enqueued operations |
| example | `187650976451584` (process-local; not a persistent identifier) |
| C++ conversion | `uintptr_t` to borrowed `aclrtStream` |
| ownership | borrowed; CPUInfer never creates, destroys or retains ownership of the stream |

`kt_kernel.get_current_device_stream_handle("npu")` uses only the public Python surface. It rejects an absent, non-integer or zero handle. No `torch_npu` private C++ headers, symbols, ABI or pointer extraction hacks are used.

The device-neutral CPUInfer methods are `submit_with_device_stream` and `sync_with_device_stream`. Existing `*_with_cuda_stream` names remain compatibility aliases, without changing ownership.

Result: **A3_VERIFIED**. Four stream-handle tests passed in the focused run; the combined matrix executed three Ascend cases and conditionally skipped one CUDA-only case.

Evidence: `torch-npu-stream-audit.log`, `stream-handle-tests.log`, `ascend-import-api-final.log`.
