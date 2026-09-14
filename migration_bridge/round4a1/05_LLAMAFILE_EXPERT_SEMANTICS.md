# LLAMAFILE Expert Semantics

The audited implementation is `kt-kernel/operators/llamafile/moe.hpp`.

## Original behavior

| Stage | Original behavior |
|---|---|
| GGUF storage | F32; source values are BF16-valued |
| hidden input | BF16 converted as required by the SGEMM vector-dot type |
| gate/up | LLAMAFILE SGEMM with F32 output/accumulation trajectory |
| activation | scalar FP32 SiLU |
| multiply | FP32 |
| down | LLAMAFILE SGEMM with F32 output/accumulation trajectory |
| routing scale/sum | FP32 |
| public output | previously converted directly to hidden dtype |

## Compatibility behavior

`GeneralMOEConfig.bf16_numerical_compat` is explicit and default false. The
LLAMAFILE wrapper enables it for the current Hybrid backend. Both decode
`forward_one` and prefill `forward_many` now:

1. retain F32 SGEMM accumulation;
2. round gate and up outputs to BF16 values;
3. compute fused `SiLU(gate) * up` in FP32 and round the fused result once;
4. run down SGEMM and round its output to BF16 values;
5. round route weights to BF16 values;
6. retain an F32 partial route sum for the CPU/NPU boundary.

This removes the semantic R0/R2 mismatch. It does not make ARM LLAMAFILE SGEMM
use the same reduction tree as Ascend GMM. Sparse residual BF16 differences are
therefore possible even when storage and externally visible dtype boundaries
match.
