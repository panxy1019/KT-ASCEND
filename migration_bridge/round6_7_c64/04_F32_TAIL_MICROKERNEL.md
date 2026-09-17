# F32 tail microkernel qualification

Evidence state: **A3_VERIFIED** (target 910B3 host; this is a CPU SGEMM test).

The isolated extension links the deployed LLAMAFILE objects and frozen llama archive;
it does not substitute PyTorch for production GEMM. PyTorch FP32 is the reference.
Seed 67064, 10 repetitions per shape, one caller thread.

- M: 32, 64, 128, 192, 224, 256.
- N: 1, 2, 8, 16, 32.
- K: 2048 and 704.
- Required relative L2 ≤ 1e-4, finite, identical repeated output, successful kernel return.
- **60/60 PASS**, including the 192-row remainder and the full 704-element down reduction.
- Maximum observed relative L2: `9.696490224087029e-7`.

Raw results: `evidence/development_smoke/sgemm_tail_microprobe.json`.
This proves tile support, not full-model C64 qualification or speedup.

## Environment bring-up failures preserved

The first operator attempt could not initialize the NPU (507899). Two older
project-owned probe containers were inspected: each contained only `sleep infinity`.
They were stopped, not removed; NPU tensor allocation then succeeded. This sequence
does not establish the driver's exact internal root cause. No device reset or product
process termination occurred.

The next attempt exposed missing `tbe` Python environment initialization. The runner
now sources the same CANN/toolkit/ATB environment as the retained serving launcher.
Both failed logs remain under `/workspace/c64-dev`; neither is counted as a pass.
