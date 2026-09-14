# Backend Integration Design

The OpenBLAS experiment changed only the six LLAMAFILE F32 SGEMM call sites.
An explicit selector `KT_CPU_GEMM_BACKEND=OPENBLAS_ROUND4A2_EXPERIMENTAL`
loaded an absolute pinned library path with `dlopen`, forced one inner BLAS
thread (CPUInfer already owns output-block parallelism), rejected non-F32 or
partial-task calls, and never silently fell back.

Routing, mapping, top-k, shared expert ownership, routed scaling, CPUInfer
scheduling, Hybrid merge, and NPU streams were unchanged. With no selector,
the original LLAMAFILE path remained byte-for-byte behaviorally active.

The adapter built and passed isolated-vs-integrated equality, C0–C4, overlap,
sequential and 1000-forward gates. Full P1 nevertheless failed. The adapter was
therefore removed before finalization. Final production diff: **zero**.

The probe-only ACL shim remains under `tools/`; it is not linked or imported by
production code.
