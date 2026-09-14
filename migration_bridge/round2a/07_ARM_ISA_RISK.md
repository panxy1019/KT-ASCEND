# ARM ISA Portability Audit

## Executive conclusion

A3 构建和运行通过，但当前 ARM production build 对所有 ARM translation units 全局加入：

```text
-march=armv8.2-a+fp16+dotprod+sve+bf16
```

因此当前二进制的部署基线必须视为要求 FP16、dotprod、SVE、BF16。A3 当前 CPU features 确实包含这些能力，所以本轮验证有效；但不能由此推断所有“Kunpeng 920”或所有 aarch64 主机都兼容。

风险等级：`P1_BUILD_PORTABILITY_RISK`。它不阻塞 A3-specific Round 2A exit gate，但在发布通用 ARM wheel 前必须解决或明确 platform contract。

## Source and compiled evidence

- `kt-kernel/CMakeLists.txt:<ARM architecture block>:199-254`
- active hard-coded flag：`kt-kernel/CMakeLists.txt:248`
- A3 `compile_commands.json` 确认 production translation units 实际包含同一 `-march=armv8.2-a+fp16+dotprod+sve+bf16`，不是注释或未使用变量
- A3 `/proc/cpuinfo` features 包含 `fp`, `asimd`, `asimddp`, `sve`, `svei8mm`, `svebf16`, `i8mm`, `bf16` 等

## Required-feature answers

### Does current binary require SVE?

**是，按二进制兼容性契约必须视为 required。** 即使未证明每条热路径都发射 SVE 指令，global `-march ... +sve` 允许编译器在任何 production translation unit 生成 SVE。不能把“目前 grep 到的显式 SVE intrinsic 较少”解释为可在无 SVE CPU 上安全运行。

### Does current binary require BF16?

**是，按二进制兼容性契约必须视为 required。** `+bf16` 全局允许 BF16 ISA codegen；A3 支持并成功执行，但通用 ARM 兼容性未建立。这里的 ISA BF16 与 tensor dtype BF16 是相关但不同的两个层面。

### Does current binary require dotprod?

**是，且有更直接的代码证据。** `+dotprod` 全局启用，third-party LLAMAFILE/tinyblas 中存在 `__ARM_FEATURE_DOTPROD` guarded branches 和 `vdot` implementation。A3 的 `asimddp` 支持该能力。

### Does current binary require i8mm?

**本次 KML-off LLAMAFILE build 不要求。** A3 虽报告 `i8mm/svei8mm`，但 active `-march` 没有 `+i8mm`，本轮也没有启用依赖它的 KML/other optimized backend。不能因为 A3 支持它而提升通用 baseline。

## Why no code change in Round 2A

可靠修复需要选择明确策略，例如 baseline wheel + runtime-dispatched optimized objects、按 feature 构建多 wheel，或把 `-march` 改为受测试的最低 baseline。任何一种都会影响广泛 translation units、性能和 packaging，超出本轮 surgical CPU Expert Plane correctness 范围。

Round 2A 只记录事实和风险，不按“Kunpeng 920”型号硬编码能力、不创建虚假的 `sve`/`bf16` variant，也不做未经基准验证的 flag 修改。

## Follow-up acceptance for a portable ARM release

后续至少需要：

1. 定义支持的最低 ARM ISA baseline；
2. 在不含 SVE/BF16/dotprod 的目标或模拟器上验证 baseline build/import；
3. 对 optimized object 做 feature gating 或发布平台限定 wheel；
4. 保留 A3 的 current-feature build regression；
5. 检查 wheel tag/发布说明是否真实表达 CPU requirement。
