# Round 2A 执行环境

## 结论

Round 2A 在 A3 的一次性容器中完成 CPU-only 构建和验证。宿主机未安装软件包、未映射 NPU/GPU 设备、未启用 KML，测试使用 `torch=2.9.0+cpu`。远端审计目录和日志保留，容器在交付前删除。

## Repository

- 本地仓库：`/home/admin/Desktop/KT_ASCEND/ktransformers`
- A3 隔离副本：`/home/admin/kt_round2a_c40d37c`
- base commit：`c40d37c63cb9b7d041c8489167b3b822961d12c5`
- branch：`feature/kt-arm-llamafile-cpu-plane`
- A3 构建审计所记录的 commit：base commit；本轮 tracked diff 单独同步到隔离副本后构建和测试

## A3 与容器

- A3 kernel：`Linux 6.6.0-72.0.0.76.oe2403sp1.aarch64`
- architecture：`aarch64`
- disposable container：`kt-r2a-arm-plane`
- image：`quay.io/ascend/verl:v0.8.0-cann9.0.0-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-vllm`
- image ID/digest：`sha256:e846d09283eb1ad17f7b219076df6fab7343f145dc9f72250799c68a279ee775`
- CPU cpuset：`0-7`
- memory limit：`32 GiB`
- pids limit：`2048`
- network：host
- device mappings：空；没有 NPU/GPU device request
- source bind mount：`/home/admin/kt_round2a_c40d37c:/workspace/kt-src`

容器镜像自带的 PyTorch 会尝试自动加载 `torch_npu`；所有测试命令均显式设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。实际导入结果为 `torch=2.9.0+cpu`，没有加载或调用 NPU backend。

## Toolchain

| Component | Version/result |
|---|---|
| Python | 3.11.15 |
| GCC | 11.4.0 |
| CMake | 4.4.2 |
| PyTorch | 2.9.0+cpu |
| pytest | 8.3.2 |
| gguf | available |
| hwloc | 2.7.0 |

仅在 disposable 容器中安装了普通 CPU 构建依赖：`pkg-config`、`hwloc/libhwloc-dev`、`libnuma-dev/numactl`、`ninja`、`python3-pytest`。容器内 apt 源临时切换为 Huawei mirror，原文件已备份；宿主机没有执行 apt/yum/dnf 安装。

## CPU-only 构建约束

构建使用独立输出目录：

- temp：`/workspace/kt-src/build/r2a-cpu`
- library：`r2a-lib`
- staged Python：`r2a-python`

关键开关：

```text
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
KTRANSFORMERS_CPU_MOE_KERNEL=OFF
```

结果：extension `kt_kernel_ext...aarch64.so` 构建成功，import 成功，运行时 metadata 为 `arm`。

## Evidence

所有 A3 原始证据保存在：

```text
/home/admin/kt_round2a_c40d37c/logs/round2a/
```

主要文件：

- `environment.log`
- `container-inspect.log`
- `deps-install.log`
- `cpu-only-build.log`
- `build-python.log`
- `compile_commands.json`
- `arm-variant-test.log`
- `fixture-generation-a.log`
- `fixture-generation-b.log`
- `fixture-reproducibility.log`
- `gguf-e2e-rerun.log`
- `routed-correctness-final.log`
- `cpuinfer-lifecycle.log`
- `full-test-matrix.log`
- `arm-isa-environment.log`

## Scope boundary

本轮未修改或验证 Ascend runtime、CANN、SGLang、NPU stream、D2H/H2D、真实模型、性能、KML、TP2+。这些不属于 CPU Expert Plane 的 Round 2A 范围。
