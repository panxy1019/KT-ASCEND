# E002 — KML Build on A3

状态：`FAIL / A3_VERIFIED`

## Scope and isolation

- 使用冻结 commit `c40d37c63cb9b7d041c8489167b3b822961d12c5`。
- 新建远端目录 `/home/admin/kt_ascend_round1_c40d37c`，不覆盖现有 checkout。
- 使用既有空闲开发容器 `verl-0.8.0-a3`。
- `taskset -c 0-3`，build parallelism 4。
- CPU only；未运行 NPU；未安装系统/Python 包。

## Command

```bash
cd /home/admin/kt_ascend_round1_c40d37c/kt-kernel
CPUINFER_FORCE_REBUILD=1 \
CPUINFER_ENABLE_KML=ON \
CPUINFER_USE_CUDA=0 \
CPUINFER_PARALLEL=4 \
taskset -c 0-3 python setup.py build_ext --inplace
```

## Observed configure

```text
version: 0.7.0
ARM detected
ARCH_FLAGS: -march=armv8.2-a+fp16+dotprod+sve+bf16
KTRANSFORMERS_CPU_USE_KML=ON
KTRANSFORMERS_CPU_MOE_KERNEL=ON (auto-enabled)
KML CPU detected
SOURCE_DIR7 only contains operators/moe_kernel/la/mat_kernel.cpp
```

setup 的 CPU feature classifier 打印 `vendor=unknown, arch=aarch64, features=[]`，尽管同一主机 raw `/proc/cpuinfo` flags 完整包含所需 ISA。原因边界：当前自动 feature 分类逻辑以 x86 feature 名为主；但该空 features 没有阻止显式 KML flag 和 CMake ARM flags 生效。

## Exact failure

```text
Could NOT find PkgConfig (missing: PKG_CONFIG_EXECUTABLE)
CMake Error at CMakeLists.txt:678 (message):
  FindHWLOC needs pkg-config program and PKG_CONFIG_PATH must contain the path
  to hwloc.pc file.
```

阶段：CMake configure；尚未编译或链接任何 KML/MoE target。

## Result

- Configure: FAIL
- Compile: NOT REACHED
- Link: NOT REACHED
- Extension artifact: NOT PRODUCED
- Root classification: `BUILD_CMAKE`，并存在已确认的后续 `KML_LIBRARY/KML_API` 缺口。

完整分析见 `../B_R1_KML_BLOCKER.md`。

