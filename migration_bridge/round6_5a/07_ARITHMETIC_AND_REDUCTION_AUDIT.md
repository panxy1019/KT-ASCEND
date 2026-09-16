# Arithmetic / scaling / reduction audit (in progress)

Evidence states are deliberately separate: source inspection is CODE_INSPECTED;
the captured-input microprobe results are A3_VERIFIED within their stated scope.

| Boundary | Deployed CPU path | Deployed NPU / merge path |
|---|---|---|
| Weights | F32 GGUF preserves source BF16 values | BF16 checkpoint tensors |
| Hidden input | BF16, exact captured D2H | BF16 |
| Gate/up | F32 arithmetic, explicit BF16-valued rounding | grouped GEMM BF16 output |
| SwiGLU multiply | explicit BF16-valued output rounding | npu_swiglu BF16 output |
| Down | F32 arithmetic, explicit BF16-valued rounding | grouped GEMM BF16 output |
| Route weights | explicit BF16-valued rounding | converted to hidden dtype before finalization |
| Partial output | FP32 | FP32 contribution for LLAMAFILE Hybrid |
| Hybrid merge | CPU FP32 + NPU FP32 | one final cast to hidden dtype |
| Routed scaling | not inside CPU expert kernel | runner factor None; outer DeepSeek applies factor |

Inspected deployed locations:

- installed `kt_kernel/utils/llamafile.py`: output_dtype FP32; bf16_numerical_compat=True.
- `kt-kernel/operators/llamafile/moe.hpp`: decode and prefill rounding branches.
- child `kt_ep_wrapper.py`: create_moe_runner disables internal routed scaling
  and requests FP32 NPU contribution for LLAMAFILE.
- child `unquant.py`: grouped GEMM, SwiGLU, BF16 route weights, FP32 finalization.
- child `deepseek_v2.py`: routed scaling outside experts and subsequent shared addition.

## Reduction topology

Frozen launch uses 32 workers, threadpool_count=1, numa_nodes=[0]. Native probe
logs report `TP 0: size=1408, offset=0` with the tail-safe single-TP path.
There is no positive evidence of multi-subpool down-projection reduction here.
Upstream PR2092's one-versus-two-subpool explanation therefore is not the
identified cause. CPU/GMM reduction and fused activation differences remain
possible arithmetic differences; their exact instruction-level cause is not proven.

This audit does not add any cast or scaling patch to make a particular token pass.
No CPU core expansion or NUMA policy change was performed.
