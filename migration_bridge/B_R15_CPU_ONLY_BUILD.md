# B Round 1.5 — A3 CPU-only Build

## Result

**PASS / A3_VERIFIED**

At KTransformers commit `c40d37c63cb9b7d041c8489167b3b822961d12c5`, the current `kt-kernel` extension builds on the A3 Kunpeng 920 host with KML, the optional CPU MoE kernel, MLA, and every GPU backend disabled. No host package or source-tree change was needed.

## Isolation boundary

- Execution host: A3, `aarch64`, openEuler 24.03 LTS-SP1.
- Build container: `kt-r15-cpu-baseline`, created solely for Round 1.5.
- Image: `quay.io/ascend/verl:v0.8.0-cann9.0.0-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-vllm`.
- Limits: CPUs `0-7`, memory `32 GiB`, PID limit `2048`.
- Source: host audit copy mounted read/write at `/workspace/kt-src`.
- No NPU devices were mapped into the container.
- The A3 host package database, compiler, libraries, services, drivers, and environment were not modified.

`ports.ubuntu.com` returned HTTP 502 through the available proxy. Only the disposable container's `/etc/apt/sources.list` was switched to the Huawei Cloud Ubuntu Ports mirror; the original was retained as `/etc/apt/sources.list.round15.orig`.

## Dependencies installed inside the container

| Dependency | Observed version |
|---|---:|
| `pkg-config` | `0.29.2-1ubuntu3` |
| `hwloc` / `libhwloc-dev` | `2.7.0-2ubuntu1` |
| `libnuma-dev` / `numactl` | `2.0.14-3ubuntu2` |
| `ninja-build` | `1.10.1-1` |

The exact apt transcript is retained in `logs/r15/deps-install.log` on the isolated A3 audit copy. These are ordinary CPU build dependencies, installed under the user's explicit authorization and only inside the disposable container.

## Effective build semantics

```text
CPUINFER_FORCE_REBUILD=1
CPUINFER_ENABLE_KML=OFF
CPUINFER_ENABLE_BLIS=OFF
CPUINFER_ENABLE_MLA=OFF
CPUINFER_ENABLE_AMX=OFF
CPUINFER_ENABLE_AVX512=OFF
CPUINFER_USE_CUDA=0
CPUINFER_USE_SYCL=0
CPUINFER_USE_ROCM=0
CPUINFER_USE_MUSA=0
CPUINFER_USE_MACA=0
CPUINFER_PARALLEL=4
CMAKE_ARGS=-DKTRANSFORMERS_CPU_MOE_KERNEL=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Build command:

```bash
python setup.py build_ext \
  --build-temp /workspace/kt-src/build/r15-cpu \
  --build-lib /workspace/kt-src/build/r15-lib \
  --inplace
```

The configure log explicitly reports `No GPU support enabled, building for CPU only`. `SOURCE_DIR7` is empty, so the missing historical KML source directory is not compiled. The generated ARM flags include `-march=armv8.2-a+fp16+dotprod+sve+bf16`.

## Artifacts and evidence

- Extension: `kt_kernel_ext.cpython-311-aarch64-linux-gnu.so`.
- Build logs on A3: `/home/admin/kt_ascend_round1_c40d37c/logs/r15/`.
- Required evidence: `configure.log`, `build.log`, `compile_commands.json`.
- Combined invocation log: `cpu-only-build.log`.
- Dependency evidence: `deps-install.log`, `dependency-versions.log`.

`compile_commands.json` proves that the ARM LLAMAFILE sources were part of the actual build, including the ARM 8.0/8.2 tinyblas and IQK matrix multiplication translation units. It does not contain the deleted KML sources.

## Packaging observation

An additional `pip install --no-build-isolation --no-deps .` attempt failed because the copied audit tree contained a stale CMake cache whose original absolute path was `/home/admin/...`, while the container mount is `/workspace/...`. This is classified as **packaging / stale-cache**, not a compiler or CPU-only build failure. No destructive cache cleanup was used.

For the import/runtime tests, Python files were staged with `setup.py build_py`, and the successfully built extension was placed into that staged package. The tests then used an explicit `PYTHONPATH`. This preserves the source snapshot and separates build validity from the stale-cache packaging issue.

## ARM metadata defect found

The extension is an `aarch64` ELF and actually compiled ARM kernels, but `kt-kernel/python/_cpu_detect.py` has an x86-only feature hierarchy and falls back to the string `avx2` when no x86 flags match. The extension metadata has a similar default branch. Consequently, `kt_kernel.__cpu_variant__ == "avx2"` on A3 is misleading metadata; it is not evidence that AVX2 code ran.

This defect did not invalidate the executable test: an x86 AVX2 binary could not load as an aarch64 extension, and the compile database shows the ARM source set and ARM ISA flags. It should nevertheless be corrected in a later, separately scoped patch.

## Scope exclusions

No KML runtime, NPU, Ascend adapter, CANN callback, SGLang patch, model, TP2+, graph execution, deferred experts, or dynamic placement was used.
