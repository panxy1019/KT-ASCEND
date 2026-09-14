# CPUInfer Device Callback Bridge

## Design

`cpu_backend/vendors/vendor.h` provides one device-neutral callback abstraction. The Ascend implementation maps it to `aclrtLaunchHostFunc`; CUDA-family backends keep their existing host-function launch API behind the same interface.

`CPUInfer::submit_with_device_stream` places a host callback after prior device work. The callback only transfers an already-created task into `TaskQueue`. `CPUInfer::sync_with_device_stream` places a host callback that waits for the CPU task queue before later H2D work can proceed. Neither callback submits nor synchronizes device work.

A `noexcept` trampoline prevents C++ exceptions from crossing CANN's C ABI. The first callback exception is retained and rethrown by `sync()`, a later device-stream submission/sync, or `rethrow_device_callback_error()` on a host thread.

## A3 evidence

| Test | Result |
|---|---|
| submit callback | PASS |
| sync callback | PASS |
| 1,000 submit/sync cycles | PASS |
| CPU/NPU concurrency | CPU 150.054 ms, NPU 27.378 ms, wall 150.401 ms |
| conservative overlap lower bound | 27.032 ms, acceptance ≥5 ms |

Focused callback result: `3 passed in 7.43s`; equivalent combined-matrix cases also passed.

Evidence: `cpuinfer-callback.log`, `ascend_full_matrix.log`.
