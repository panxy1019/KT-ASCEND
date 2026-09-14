# Dtype and Accumulation Matrix

| Variant | Effective semantics | Observation against production Ascend |
|---|---|---|
| R0 | F32 input/weight/operator trajectory | down rel-L2 about 0.003-0.005; rejected |
| R1 | BF16 operator/storage reference | not used as the production CPU implementation; CPU fallback semantics must not be assumed |
| R2 | F32 SGEMM, BF16 round gate/up, BF16 round fused multiply, F32 down SGEMM, BF16 down | best practical LLAMAFILE-compatible semantic match |
| R3 | BF16-valued operands with F32 accumulation, BF16 output | equivalent intent to the implemented R2 boundaries for current BF16-valued F32 GGUF weights |
| R4 | captured production `npu_grouped_matmul` + `npu_swiglu` + finalize | reference, not an inferred implementation |

R2 reduces the large semantic error by orders of magnitude. Examples from the
initial real capture:

- E6 sample: down max absolute difference `1.52587890625e-05`, relative L2
  `5.260742563450784e-07`;
- E6 decode sample: down max absolute difference `4.76837158203125e-07`,
  relative L2 `2.6401522163919447e-08`;
- several gate/up and multiplied tensors are bitwise equal.

The expanded `v_struct_03` capture also proves R2 is not universally bitwise:
some rows differ in only a few BF16 positions, with maximum absolute differences
of 0.001953125-0.0078125, while other E25/E36 rows are exact. This dependence on
input values is characteristic of a reduction-order boundary case.

Separately rounding the SiLU result before multiplying is incorrect for the
observed Ascend fused operator and worsens relative L2 back to roughly
`1.7e-3`-`4.5e-3`.
