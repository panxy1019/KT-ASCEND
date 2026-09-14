# SGLang KT Ascend Bridge

## Runtime bridge

The bridge reuses Round 2B CPUInfer and its native accelerator stream handle.
KTEP creates the auxiliary stream and event from the active torch device module,
records producer completion, starts CPU work, runs the original NPU expert method,
then merges the two disjoint contributions.

For Ascend decode staging, raw CPUInfer pointers cannot participate in torch_npu's
private copy-stream dependency tracking. Repeated `qlen=1` testing exposed partial
D2H reads even after `torch.npu.current_stream().synchronize()` and
`torch.npu.synchronize()`. The verified boundary therefore:

1. synchronizes the exact native ACL stream supplied by the caller;
2. uses synchronous `acl.rt.memcpy` for D2H into owned host tensors;
3. retains those tensors until `CPUInfer.sync()` completes;
4. uses synchronous ACL H2D for the CPU contribution.

CPU expert computation still overlaps the NPU expert kernels. The measured real
interval overlap lower bound was 1.156 ms (`CPU 1.772 ms`, `NPU 2.139 ms`, wall
2.755 ms).

## Synthetic E2E

- KTransformers Round 2C suite: 44 passed
- Decode race regression: 20 repeated `qlen=1` calls in one process passed
- Additional cold-process stress: 10/10 passed
- SGLang KT EP Ascend E2E: 4 passed
- Covered CPU-only, NPU-only, mixed, and reverse-mixed route rows
- Maximum synthetic row absolute difference: 0.000244140625

Evidence:

- `logs/round3/round2c-regression-final.log`
- `logs/round3/round2c-overlap-acl-stream-sync-repeat10.log`
- `logs/round3/sglang-kt-ep-ascend-e2e-final.log`

