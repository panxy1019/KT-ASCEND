# B Round 1 Current KML Map

状态：build switch 与源码链 `CODE_INSPECTED`；A3 configure 行为 `A3_VERIFIED`；完整 KML backend `BLOCKED`。

## Build switch chain

```text
CPUINFER_ENABLE_KML=ON
  -> kt-kernel/setup.py:586
  -> -DKTRANSFORMERS_CPU_USE_KML=ON
  -> setup.py:646-657 auto-adds -DKTRANSFORMERS_CPU_MOE_KERNEL=ON
  -> CMakeLists.txt:565-566 reports "KML CPU detected"
  -> CMakeLists.txt:580-595 collects ARM/KML and KML mat-kernel sources
  -> CMakeLists.txt:781-795 adds prefill/decode KML targets and CPU_USE_KML
  -> ext_bindings.cpp:915-920 exports Int8_KERNEL_MOE and conditional Int4_KERNEL_MOE
  -> python/utils/moe_kernel.py:11-24 imports those symbols
  -> GeneralMoEWrapper selected by experts.py:359-360
```

ARM compile flags are not feature-probed per host: `CMakeLists.txt:248` unconditionally appends:

```text
-march=armv8.2-a+fp16+dotprod+sve+bf16
```

A3 configure 实际打印了同一组 `ARCH_FLAGS`。A3 的 `/proc/cpuinfo` 确认这些要求均存在，详见 `experiments/E001_A3_CPU_ISA.md`。

## Expected source participation

CMake 期望以下 KML 源码树：

- `operators/kml`（`CMakeLists.txt:580-585`）
- `operators/moe_kernel/mat_kernel/kml_kernel`（`:593-595`）
- `.../kml_kernel/prefillgemm`（`:781-782`）
- `.../kml_kernel/prefillgemm_int4`（`:783-784`）
- `.../kml_kernel/batch_gemm.cpp` 与 `batch_gemm_kernels.cpp`（`:786-790`）

但在冻结 commit 上：

```bash
git ls-tree -r HEAD kt-kernel/operators | grep -E '/kml(/|_)|kml_kernel'
```

没有任何输出；工作树中相应目录也不存在。当前可见的 `mat_kernel` 只有 `aocl_kernel`。因此这些不是“子模块未初始化”的结果，而是 **Git commit tree 本身没有收录被当前 CMake 引用的路径**。

## MOE_INT8 runtime path as written

```text
method="MOE_INT8"
  -> GeneralMoEWrapper
  -> MOEConfig
     - num_experts/top_k/hidden/intermediate
     - gpu_experts_mask pointer
     - WorkerPool pointer
     - max_len
     - gate/up/down weights and scales
     - weight path and load/save flags
  -> Int8_KERNEL_MOE
  -> load_weights_task(physical_to_logical_map_cpu)
  -> forward_task(batch, expert ids, weights, input, output, incremental)
```

相关证据：`python/utils/moe_kernel.py:147-180,275-312`、`experts_base.py:422-455`、`ext_bindings.cpp:915-920`。

### Decode/prefill

`operators/moe_kernel/la/kernel.hpp:296-374` 中 `GemmKernelInt8` 定义 runtime tiling。decode 与 prefill 对 up/gate、down 使用独立 N block，且要求 N 可整除；pybind 在 `ext_bindings.cpp:923` 后暴露 tiling 参数。实际 KML GEMM 实现所在的引用目录缺失，故不能进一步从本提交证明具体 KML API、packing 或 numerical behavior。

### Quantization/scale/weight order

Python loader 支持 merged safetensor 与分文件指针；将 gate/up/down 和各自 scale 指针填入 `MOEConfig`，并把 `physical_to_logical_map_cpu` 传给 load task。代码表明存在 expert remapping，但由于 KML mat-kernel 源码缺失、扩展未链接，以下项目保持未验证：

- INT8 packing 的完整二进制格式；
- scale 的粒度、布局和 KML API 约束；
- decode/prefill 两条路径的数值等价性；
- Kunpeng 上的实际线程分块与 NUMA 性能。

## Pybind/benchmark drift

`kt-kernel/bench/bench_moe_kml.py:176-178` 仍实例化：

```python
kt_kernel_ext.moe.KMLInt8_MOE(...)
kt_kernel_ext.moe.KMLInt4_MOE(...)
```

当前 `ext_bindings.cpp:915-920` 导出的却是：

```text
Int8_KERNEL_MOE
Int4_KERNEL_MOE
```

所以即使构建成功，现有 KML benchmark 也与当前 pybind 命名不一致。这是 `PYBIND/KML_API` 层面的静态缺口，不是本轮运行得到的数值失败。

## Current capability verdict

| Layer | Verdict | Reason |
|---|---|---|
| setup env forwarding | present | explicit KML env 被转成 CMake flag |
| ARM ISA flags | present, A3 compatible | A3 实测包含 fp16/dotprod/SVE/BF16 |
| generic MOE_INT8 Python wrapper | present | GeneralMoEWrapper 和 Int8 symbol path 可见 |
| hwloc dependency | required | CMake 无条件要求 pkg-config + hwloc |
| KML installation on audited A3 env | not found | host/selected safe container搜索均未找到 |
| KML implementation sources in commit | missing | CMake 引用路径不在 Git tree |
| full extension build | FAIL | 首个失败为 pkg-config/hwloc；还有确定的后续源码/KML缺口 |
| import/runtime/numerical | BLOCKED | extension 未产生 |

