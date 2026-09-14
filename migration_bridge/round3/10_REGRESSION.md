# Regression Results

## Round 2A

- 21 passed in 5.21 s
- 1,000 CPU forward cycles
- RSS delta: 0 bytes
- Evidence: `logs/round3/round2a-regression.log`

## Round 2B

- 23 passed, 1 expected CUDA-only skip in 26.98 s
- 1,000 pipeline cycles
- RSS delta: 6,184,960 bytes, below the 16 MiB gate
- Evidence: `logs/round3/round2b-regression.log`

## Round 2C

- Final result: 44 passed, 1 warning in 18.19 s
- CPU/NPU real interval overlap lower bound: 1.156 ms
- 1,000 lifecycle cycles RSS delta: 6,184,960 bytes
- Formal decode repeat regression: 20 iterations
- Evidence: `logs/round3/round2c-regression-final.log`

## SGLang bridge

- 4 passed in 17.45 s
- Covers all-accelerator skip, int32 mapping, negative CPU-route sanitation,
  and CPU/NPU mixed-route E2E
- Evidence: `logs/round3/sglang-kt-ep-ascend-e2e-final.log`

Warnings are environment/configuration warnings (`hwloc` membind permission,
torch_npu internal-format notice, and unsupported unrelated quant backends), not
test failures.

