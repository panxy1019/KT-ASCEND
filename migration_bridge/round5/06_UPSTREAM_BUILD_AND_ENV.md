# Round 5 — Raw Upstream Build and A3 Environment

Evidence state: `A3_VERIFIED` for the raw build artifact only.

## Frozen source

| Item | Value |
|---|---|
| parent | `440df948027128318bc30a11a545433eb0c7d383` |
| SGLang | `3424f35d0e60f03b9e75d9b90941f8f4cb944524` |
| recursive submodules | 11 entries, all exact gitlinks |
| local patches before build | none |

The first A3 HTTPS clone attempt failed before checkout with
`OpenSSL SSL_connect: SSL_ERROR_SYSCALL`. The pinned, locally verified Git
object database was therefore transported over SSH. A3 then initialized the
remaining recursive submodules from their canonical remotes and reverified all
gitlinks. This bootstrap difference is provenance, not a source-code change.

The source appeared as `m third_party/llama.cpp` after build because upstream's
own setup applies its two declared llama.cpp build patches in the submodule
worktree. The parent and child commits were exact before the build began.

## Disposable environment

| Item | Value |
|---|---|
| host | `a3-server-00` |
| container | `kt-r5-upstream-alignment` |
| container image | `kt-r4a5-env:p2-closure-20260911` |
| CPU affinity | `0-15` |
| NPU visibility | device nodes for NPU0 only |
| CANN | `9.0.0` |
| Python | `3.11.15` |
| torch | `2.9.0+cpu` |
| torch_npu | `2.9.0.post2` |
| GCC | `11.4.0` |
| CMake | `4.4.2` (system environment) |
| host modified | NO |
| business containers modified | NO |

The older project container `kt-r4a5-p2-closure-env` had to be stopped, not
deleted, because the A3 driver returned zero devices while two containers held
the same NPU0 device namespace. The unrelated `mzy-train` container was not
modified. After the old disposable container stopped, the new container saw
exactly one NPU and completed an NPU tensor smoke test.

The OS exposes DOTPROD, FP16, SVE, BF16 and I8MM CPU flags on this same A3 host.
This is recorded as observed capability even though earlier project shorthand
called the CPU "Kunpeng 920". Upstream auto-detection enabled those exposed
flags. This is not interpreted as a hardware identity claim.

## Build command contract

The wheel was built in a new venv (`venv-raw`) with system torch/torch_npu
visible and the following explicit backend controls:

```text
CPUINFER_USE_CUDA=0
CPUINFER_USE_ASCEND_NPU=1
CPUINFER_ENABLE_KML=OFF
CPUINFER_ENABLE_BLIS=OFF
CPUINFER_PARALLEL=16
CPUINFER_ARM_DOTPROD=ON
CPUINFER_ARM_FP16=ON
CPUINFER_ARM_SVE=ON
CPUINFER_ARM_BF16=ON
CPUINFER_ARM_I8MM=ON
pip install --no-build-isolation --no-deps . -v
```

`--no-build-isolation` was used because the upstream install wrapper attempted
to redownload an already installed 30 MB CMake wheel over an unstable A3
connection. This changes dependency acquisition only, not source or compiler
flags.

## Result

| Artifact | Result |
|---|---|
| kt-kernel version | `0.7.0.post4` |
| wheel | `kt_kernel-0.7.0.post4-cp311-cp311-linux_aarch64.whl` |
| wheel SHA256 | `5fb110e6b5285eeb25cbb4b308f025c0862b52db471471e74cded9c07304727d` |
| extension link | `libascendcl.so` resolved from CANN 9.0.0 |
| callback symbols | all four expected Ascend worker symbols present |
| build | PASS |

The setup banner says "CPU-only multi-variant" because its display-only
`gpu_backend` enumeration omits Ascend. CMake independently and explicitly
reported `Ascend NPU (CANN) backend selected`, compiled
`ascend_callback_worker.cpp`, and linked `libascendcl.so`. The misleading
banner is tracked as a packaging defect, not treated as a CPU-only binary.

```text
R5_G1_UPSTREAM_CLEAN_BUILD = PASS
```

## Evidence

- `evidence/a3_raw_submodule_status.txt`
- `evidence/a3_raw_submodule_update.log`
- `evidence/a3_raw_upstream_build.log`
- `evidence/a3_raw_wheel.sha256`
- `evidence/a3_raw_extension_ldd.txt`
- `evidence/a3_raw_environment.txt`

