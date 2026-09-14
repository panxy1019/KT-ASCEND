# B Round 1 Summary

## Repository

commit: `c40d37c63cb9b7d041c8489167b3b822961d12c5`

sglang commit: `0f36b26d7523351ec88b1e694e6d810234911c12`

version: `0.7.0`

## A3

CPU: 4 × Kunpeng 920 7280Z；320 physical cores / 640 logical CPUs；8 NUMA nodes

NPU: 16 × Ascend（board name `9382`），本轮未使用

OS: openEuler 24.03 LTS-SP1，aarch64，kernel `6.6.0-72.0.0.76.oe2403sp1.aarch64`

CANN: 审计容器 `9.0.0`

torch_npu: 审计容器 `2.9.0.post2`

KML: 宿主及所审计空闲容器未找到

## KML

build: **FAIL**（CMake configure 首先被 pkg-config/hwloc 阻塞）

kt_kernel import: **BLOCKED**

CPUInfer: **BLOCKED**；standalone TaskQueue test PASS

MOE_INT8 reference: **BLOCKED / NOT RUN**

## Main blocker

首个可复现错误是 `CMakeLists.txt:678` 要求 pkg-config/hwloc，而安全构建容器没有这两个依赖。更关键的是，当前 commit 的 Git tree 不包含 CMake 明确引用的 `operators/kml` 和 `operators/moe_kernel/mat_kernel/kml_kernel`，A3 环境也未发现 KML SDK/runtime；所以仅补 hwloc 不能完成构建。

## Current Ascend gap

CPUInfer vendors 只有 CUDA、HIP、MUSA、MACA；没有 `ascend.h`、`KTRANSFORMERS_USE_ASCEND`、ACL host callback/memcpy 或 kt-kernel 内的 `torch_npu`。SGLang KT wrapper 的 stream 和 synchronize 仍直接使用 `torch.cuda`。本轮只记录事实，没有添加 Ascend patch。

## Acceptance

```text
[PASS] latest repo frozen
[PASS] A3 environment documented
[PASS] Kunpeng ISA confirmed
[FAIL] kt-kernel KML build
[BLOCKED] kt_kernel import
[BLOCKED] CPUInfer smoke (standalone TaskQueue PASS only)
[BLOCKED] MOE_INT8/KML numerical test
[PASS] latest SGLang KT call graph documented
[PASS] Ascend vendor gap documented
[PASS] exact blocker and reproduction documented
[PASS] no speculative runtime patch
```

## Questions requiring A legacy inspection

Q001. 当前 CMake 引用但 commit 中缺失的 `operators/kml` 与 `operators/moe_kernel/mat_kernel/kml_kernel`，在 A 侧历史可运行环境中来自哪个确切 commit、patch 或外部源码包？

Q002. A 侧历史环境使用的 KML SDK/runtime 精确版本、安装路径、头文件/库清单及获取方式是什么？

这两个问题由已确认的 source/runtime provenance 缺口直接产生，不是人为扩展范围。

