# Stagewise Expert Numerics

## Capture method

Two explicit, default-off diagnostic paths captured production tensors:

- KTEP CPU boundary: hidden state, logical TopK IDs and weights;
- Ascend unquantized MoE: routed input, expanded row mapping, expert counts,
  gate/up GMM output, SwiGLU output, down GMM output and finalize output.

Dumping requires both an environment configuration and an arm-file sentinel,
so model loading and warmup are not captured accidentally. No tensor fixture is
committed; the repository stores the capture recipe, hashes and compact JSON.

## First material growth

For matching real hidden states, R0 (original LLAMAFILE FP32 trajectory) first
materially differs at gate/up output. Typical relative L2 is approximately
`1.6e-3` to `1.8e-3`, grows through the fused activation/multiply, and reaches
approximately `3.0e-3` to `5.1e-3` at down output.

R2 applies BF16 value boundaries after gate/up, after fused `SiLU(gate)*up`,
and after down. On captured samples:

- gate/up versus production GMM: often bitwise exact;
- multiplied versus production `npu_swiglu`: often bitwise exact;
- down relative L2: from zero to low `1e-4` on the expanded corpus, with
  several bitwise-exact rows.

The remaining nonzero down differences are sparse BF16 elements and correlate
with the CPU SGEMM reduction implementation, not with storage, routing or an
omitted cast.

Compact evidence:

- `evidence/round4a1-current-probe.json`
- `evidence/round4a1-struct03-r2-probe.json`

Across the independent `v_en_01` and `v_struct_03` captures, the retained
sample counts are E6=6, E8=4, E25=5 and E36=4. Every selected expert therefore
has at least three real production hidden states as required by WP1.
