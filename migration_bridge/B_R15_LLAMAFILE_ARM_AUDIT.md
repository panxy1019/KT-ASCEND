# B Round 1.5 — LLAMAFILE ARM Audit

## Conclusion

**LLAMAFILE_ARM_READY**, for the current generic MoE core path tested in Round 1.5.

This status is based on both the actual A3 ARM build and a successful routed-expert runtime test. It is not based on compilation alone. GGUF file ingestion and whole-model integration remain outside this READY boundary.

## Current call path

```text
LlamafileMoEWrapper
  -> BaseMoEWrapper / CPUInfer
  -> kt_kernel_ext.moe.MOE + MOEConfig
  -> LLAMA_MOE_TP
  -> operators/llamafile
  -> third_party/llamafile matrix kernels
  -> aarch64 tinyblas / IQK translation units
```

`kt-kernel/python/utils/llamafile.py` builds a generic `MOEConfig`, supplies the worker-pool backend and weight pointers, and instantiates `kt_kernel_ext.moe.MOE`. The binding in the current `ext_bindings.cpp` exposes that generic MOE independently of `KTRANSFORMERS_CPU_MOE_KERNEL` and KML.

## Architecture dispatch

The LLAMAFILE tree is multi-architecture rather than x86-only:

- aarch64 dispatch exists in `third_party/llamafile/sgemm.cpp`.
- `iqk_mul_mat_arm.inc` has an aarch64 implementation branch.
- `tinyblas_cpu_sgemm_arm80.cpp`, `tinyblas_cpu_sgemm_arm82.cpp`, and the ARM 8.2 mixed-multiply source are present.
- `tinyblas_cpu_sgemm.inc` contains ARM NEON/aarch64 dot-product branches.
- x86 AVX/AVX2/AVX-512 code is guarded and is not the selected A3 translation path.

The captured A3 `compile_commands.json` confirms that the ARM 8.0/8.2 and IQK ARM sources were actually compiled with an ARM target flag. This closes the common gap between “ARM code exists in the repository” and “the current build actually selected it.”

## Weight and execution contracts

The generic LLAMAFILE MoE accepts `GeneralMOEConfig`/`MOEConfig` pointer fields for gate, up, and down projections. The implementation supports GGML types, with both floating and quantized calculation paths in `operators/llamafile/moe.hpp`.

Important current constraints:

- the intermediate size must satisfy the LLAMAFILE `QK_K=256` alignment used by this path;
- the wrapper normally obtains tensor metadata and addresses from `GGUFLoader`;
- execution uses `CPUInfer`'s worker pool;
- decode/small-batch selection is controlled by `group_min_len`, `group_max_len`, and `m_block` fields;
- NUMA memory policy depends on container/host capabilities beyond simple CPU affinity.

## x86-only risk audit

No unguarded x86 intrinsic was observed on the code path that compiled and executed on A3. An independent ARM runtime success also rules out a hidden mandatory x86 instruction in the exercised F32-weight/BF16-activation path.

There is one misleading x86 assumption outside the kernel dispatch: `_cpu_detect.py` falls back to the metadata label `avx2` on ARM. This is a reporting/selection-quality defect to fix later, not the instruction set actually present in the loaded aarch64 extension.

## Why READY is qualified

The dynamic test instantiated the same C++ MOE backend used by `LlamafileMoEWrapper`, but supplied deterministic in-memory F32 weights directly. Therefore:

- core ARM GEMM + activation + routing + CPUInfer path: **verified**;
- Python wrapper construction contract: **statically traced**;
- `GGUFLoader` parsing/address plumbing: **not dynamically verified**;
- quantized GGUF expert formats: **not dynamically verified**;
- full Hybrid MoE model behavior and throughput: **not verified**.

Route A can begin with this backend, but the next implementation round must add a wrapper-level GGUF fixture before claiming end-to-end readiness.
