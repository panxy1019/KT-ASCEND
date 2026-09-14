# Ascend Expert Semantics

The audited implementation is
`third_party/sglang/python/sglang/srt/layers/quantization/unquant.py:552` in the
frozen child source.

## Production path

| Stage | Observed/audited dtype behavior |
|---|---|
| input storage | BF16 (`x.dtype`) |
| expert weights | BF16 model tensors |
| routing weights at method entry | FP32, converted to `x.dtype` before finalize |
| route expansion | preserves BF16 hidden values |
| gate/up grouped matmul | `output_dtype=original_dtype`, therefore BF16 |
| activation/multiply | fused `torch.ops.npu.npu_swiglu`, BF16 output |
| down grouped matmul | `output_dtype=original_dtype`, therefore BF16 |
| route finalize | BF16 hidden values and BF16 scales in the all-NPU reference |
| routed output | BF16 |

Source-visible output dtypes do not expose the internal GMM accumulator dtype.
The controlled real-weight probe therefore defines R4 by captured operator
output rather than claiming an undocumented accumulator type.

Reconstruction from captured down rows proves the finalize contract for the
tested passes:

- TopK weights are BF16 values;
- six route terms accumulate in FP32;
- the final routed vector is rounded once to BF16;
- this reconstruction is bitwise equal to production finalize;
- sequential BF16 accumulation and FP32 (unrounded) routing weights are not.
