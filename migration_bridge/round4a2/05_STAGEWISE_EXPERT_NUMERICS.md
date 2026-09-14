# Stagewise Expert Numerics

Every reconstructed candidate expert preserves the Round 4A.1 boundary:

`gate BF16 -> up BF16 -> SiLU/multiply -> BF16 -> down GEMM -> BF16`.

OpenBLAS was the only backend to pass the initial stagewise gate. On the first
12 representative rows its down-output maximum rel-L2 improved from
`1.875118e-4` (LLAMAFILE) to `7.544104e-6`, with median zero. On all 33 rows,
both medians were zero and the worst row remained approximately `5.04841e-4`.

The result is not evidence of a universally closer reduction tree: different
captured rows move in different directions. In particular:

- v_struct03 pass8 E25: LLAMAFILE `1.518871e-4`, OpenBLAS exact;
- v_struct03 pass8 E36: both approximately `5.04841e-4`;
- v_struct03 pass9 E25: LLAMAFILE exact, OpenBLAS `3.584469e-4`.

This input dependence is consistent with backend-dependent blocking/reduction,
but does not prove one exact reduction-tree implementation.
