# B Round 1 KML Blocker

## Classification

```text
BUILD_CMAKE
KML_LIBRARY
KML_API
PYBIND
IMPORT (downstream blocked)
```

ISA 未阻塞：A3 实测具备当前 CMake flags 要求的 FP16、dotprod、SVE、BF16。没有执行到机器指令阶段，所以不存在 `ILLEGAL_INSTRUCTION` 结论。没有执行 kernel，所以不存在 `NUMERICAL`、`THREADPOOL`、`NUMA` 或 `SEGFAULT` 结论。

## Reproduction

冻结源码被复制到 A3 新建隔离目录 `/home/admin/kt_ascend_round1_c40d37c`。在既有空闲开发容器 `verl-0.8.0-a3` 中，以 CPU 0–3 和并行度 4 执行：

```bash
cd /home/admin/kt_ascend_round1_c40d37c/kt-kernel
CPUINFER_FORCE_REBUILD=1 \
CPUINFER_ENABLE_KML=ON \
CPUINFER_USE_CUDA=0 \
CPUINFER_PARALLEL=4 \
taskset -c 0-3 python setup.py build_ext --inplace
```

未安装任何依赖，未使用 NPU。

## First failing point（A3_VERIFIED）

Configure 已确认：

```text
ARM detected
ARCH_FLAGS: -march=armv8.2-a+fp16+dotprod+sve+bf16
CPUINFER_ENABLE_KML -> KTRANSFORMERS_CPU_USE_KML=ON
Auto-enabling KTRANSFORMERS_CPU_MOE_KERNEL=ON
KML CPU detected
SOURCE_DIR7: .../operators/moe_kernel/la/mat_kernel.cpp
Could NOT find PkgConfig (missing: PKG_CONFIG_EXECUTABLE)
CMake Error at CMakeLists.txt:678 (message):
  FindHWLOC needs pkg-config program and PKG_CONFIG_PATH must contain the path
  to hwloc.pc file.
```

首个停止点对应 `kt-kernel/CMakeLists.txt:674-679`。CMake 在任何 target compile 前无条件要求 pkg-config 与 hwloc。所选容器缺少二者；宿主也没有 hwloc/pkg-config。为保护集群环境，本轮没有安装它们。

## Independent later blockers（CODE_INSPECTED）

即使提供 hwloc，当前冻结 commit 还存在两个独立、可确定的后续问题：

1. `CMakeLists.txt:580-595,781-790` 引用的 `operators/kml` 和 `operators/moe_kernel/mat_kernel/kml_kernel` 不存在于 `git ls-tree -r HEAD`。这会在后续 source/add_subdirectory 阶段阻止完整 KML target。
2. A3 宿主和已审计的空闲开发容器未找到 KML header/runtime。CMake 的 MLA 分支还在 `:792-794` 引用 `kml_rt`。
3. 现有 `bench/bench_moe_kml.py:176-178` 使用 `KMLInt8_MOE/KMLInt4_MOE`，当前 pybind `ext_bindings.cpp:915-920` 使用 `Int8_KERNEL_MOE/Int4_KERNEL_MOE`。

因此“只安装 pkg-config/hwloc”不能被写成完整修复；它只会让 configure 前进到下一处确定缺口。

## Suspected root cause, bounded by evidence

最高置信的源码层原因是：**KML CMake integration 与仓库实际收录内容不同步**。这不是 submodule 未初始化，因为 `git ls-tree` 直接检查了固定 commit 对象。KML runtime 是否本应由外部 SDK 提供、缺失源码是否在未合入分支/历史提交/私有包中，本轮证据不能回答。

## Minimal candidate next steps（未执行）

1. 先向上游或 A 侧历史环境确认缺失 KML 目录的权威来源及兼容 commit，不应自行伪造 kernel。
2. 在专用可变更容器镜像中安装/挂载匹配版本的 pkg-config、hwloc-devel 与 KML SDK；不要改集群宿主。
3. 固定 KML headers/libs 版本和 source provenance 后重新 configure，确认真实 compile/link error。
4. 构建成功后先修正或替换过时 benchmark symbol，再做 import、CPUInfer 和单层 numerical test。

本轮按照任务约束没有做 speculative patch。

