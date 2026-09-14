# ARM Variant / Metadata 分析

## 结论

AArch64 现在明确报告单一 native variant `arm`，不再错误落入 x86 `avx2`。构建期 vendor、运行期检测、extension metadata 和 loader fallback 四处一致；ARM 不会沿 x86 降级链加载 AVX/AMX 扩展。

## 调用与 metadata 路径

| 分类 | file:function:line | 行为 |
|---|---|---|
| build-time CPU detection | `kt-kernel/setup.py:CMakeBuild.detect_cpu_info:167`，ARM 分支 `184-185` | `platform.machine()` 为 `aarch64/arm64` 时，无条件设置 `vendor=arm`；不依赖 `/proc/cpuinfo` 是否含 Kunpeng/Huawei 品牌字符串。 |
| build-time use | `kt-kernel/setup.py:CMakeBuild.build_extension:574-588` | 打印检测结果并转发显式 backend 开关；KML 只有显式环境变量才启用，本轮明确 OFF。 |
| runtime detection | `kt-kernel/python/_cpu_detect.py:detect_cpu_features:58`，ARM 分支 `83-87` | 先看架构；`aarch64/arm64 -> arm`，不会继续解析 x86 flags。 |
| runtime override | `kt-kernel/python/_cpu_detect.py:detect_cpu_features:75-81` | `KT_KERNEL_CPU_VARIANT` 合法集合包含 `arm`；仍保留显式调试/覆盖能力。 |
| runtime validation | `kt-kernel/python/_cpu_detect.py:_validate_loaded_variant:34-42` | detected/loaded 任一为 ARM 时必须完全匹配，阻止 ARM host 加载 x86 extension 或反向误装。 |
| extension filename | `kt-kernel/python/_cpu_detect.py:load_extension:230-257` | 先找 `_kt_kernel_ext_arm.*.so`，再接受本架构 wheel 的单 variant `kt_kernel_ext.*.so`。A3 实际产物为后者且 ELF/文件名为 aarch64。 |
| fallback path | `kt-kernel/python/_cpu_detect.py:load_extension:279-304` | x86 保留 AMX→AVX512→AVX2；`arm: None`，ARM 不降级到 x86。 |
| extension metadata | `kt-kernel/ext_bindings.cpp:PYBIND11_MODULE:519-541` | `__aarch64__/_M_ARM64` 时导出 `__cpu_variant__="arm"`；x86 宏判断只在后续分支。 |
| Python package metadata | `kt-kernel/python/__init__.py:<module>:42-43` | 初始化后以 extension 的 `__cpu_variant__` 为权威值暴露给调用方。 |
| wheel/platform metadata | `kt-kernel/setup.py:CMakeBuild.build_extension` + setuptools platform tag | 未引入虚假的 SVE/BF16 variant；wheel/extension 仍由构建平台产生 aarch64 platform tag，CPU capability 名称仅为 `arm`。 |

## 为什么原逻辑会报 AVX2

原运行期检测只解析 `/proc/cpuinfo` 的 x86 `flags`，任何未匹配 AMX/AVX512/AVX2 的环境最终走 `avx2` fallback；ARM `Features` 并不满足该列表。因此 aarch64 虽能编译，却会得到错误的 x86 metadata。修复点放在 x86 flags 解析之前，以架构而非 CPU 型号决定 ARM native 路径。

## Tests

`kt-kernel/test/per_commit/test_cpu_detect_metadata.py` 覆盖：

1. 模拟 `aarch64`，期望 `arm`；
2. 模拟 x86 AVX2，期望仍为 `avx2`；
3. unknown arch/no flags，验证既有 fallback 行为；
4. ARM/x86 extension metadata mismatch，必须拒绝；
5. loaded extension metadata 透传。

A3 结果：`6 passed`；实际 import：`architecture=aarch64`、`variant=arm`。

## Non-goals

`arm` 是架构/本机构建类别，不宣称 SVE、BF16、dotprod 或任何具体 Kunpeng SKU。具体 ISA 风险见 `07_ARM_ISA_RISK.md`。
