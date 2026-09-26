# Round11 T2 source and ISA audit

Status: **measurement pending**. The five audited source files are byte-exact between the local tree and the frozen remote candidate source.

The machine exposes ASIMD, FP16, DOTPROD, SVE, BF16 and I8MM. The HFINAL native library is built for FP16+DOTPROD and contains 8,304 `sdot` instructions; SVE/BF16/I8MM/native/LTO are off. This capability list does not justify enabling those flags. The actual MXFP4 fallback calls `ggml_vec_dot_mxfp4_q8_0`, whose frozen AArch64 implementation explicitly uses NEON table lookup and SDOT and deliberately avoids SVE/I8MM. Existing SVE/I8MM switches therefore do not select a distinct MXFP4 kernel.

Three source-backed roots remain conditional on v113-v115 measurements: the worker pool discards its calculated guided block and forces `block=1`; every TP partition quantizes the same token input locally; and the NEON row-wise MXFP4 kernel may dominate. No candidate is selected before the exclusive timing and thread/counter evidence are available. At most two different roots may be tried and at most one may be promoted.
