# KT-ASCEND Round15 文档索引

Round15 从 Round14a 的已晋级配置继续，研究两个彼此独立的问题：一是能否把成熟的 Ascend NPU 路径接入当前 QSA 热点；二是能否在不破坏 CPU Expert、PLE、状态和取消语义的前提下，把 Decode batch1 的一小段 resident NPU MoE 捕获为 `torch_npu.NPUGraph`。本轮没有改变模型 revision、CPU bank、resident map、精度、Hybrid 阈值、TP 或并发度。

最终状态为：

```text
ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION
PERFORMANCE_OBJECTIVE = NOT_ACHIEVED
```

两条候选线均完成组件和真实模型接入。A 的 QSA 候选在组件中明显变快，但 W2 请求略有退化；B 的 resident-MoE 分段图真实捕获并重放，W2 E2E 和串行吞吐约改善 `0.72%` 和 `0.73%`，但 TTFT 退化 `5.29%`，并增加约 `984 MiB` HBM。两个候选都保持默认关闭，最终配置继续使用 Round14a S0。

## 阅读顺序

1. [基线、范围与实验方法](01_BASELINE_SCOPE_AND_METHOD.md)：说明 S0 身份、冻结变量、工作负载和证据强度。
2. [P0 Provider 与 Trace 取证](02_P0_PROVIDER_AND_TRACE.md)：说明实际运行路径、两块 profile 和热点排序。
3. [A 线：QSA 成熟算子研究](03_A_QSA_OPERATOR_STUDY.md)：说明候选选择、组件门、请求结果和不晋级原因。
4. [B 线：Decode 分段图研究](04_B_SEGMENTED_GRAPH_STUDY.md)：说明图边界、输入合同、失败修复、真实命中和请求结果。
5. [故障、安全与共享 NPU 策略](05_FAILURES_SAFETY_AND_SHARED_NPU.md)：记录所有关键失败、fail-closed 行为、取消/回收与共卡规则。
6. [性能、资源与预算账本](06_PERFORMANCE_RESOURCES_AND_BUDGET.md)：汇总绝对指标、方向性变化、HBM、Host、缓存退休和预算消耗。
7. [证据与复现指南](07_EVIDENCE_AND_REPRODUCTION.md)：给出 JSON、CSV、脚本、归档、SHA 和离线复算入口。

## 关键边界

- 用户在执行中把原计划的四进程请求 ABBA 缩减为：每条候选线一个 fresh 基线进程、一个 fresh 候选进程；每个 W1/W2 各两次 measurement。
- 因此请求数据只用于判断方向和大致幅度，不能声称四进程稳定性、统计显著性、生产晋级或广泛质量资格。
- 组件正确性、数值、真实命中、取消、同进程复用、资源准入、清理和证据归档没有随之放宽。
- v356 和 v358 的失败记录原样保留；它们分别揭示了生产 dtype 漂移和首次 capture 后未 replay 的实现错误。
- Round15 没有启用整模型 Graph，也没有把 CPU Expert、PLE、QSA、GDN、KV 状态、scheduler 或取消逻辑放进图。

## 权威交付

- [最终中文报告](../../../artifacts/round15/FINAL_REPORT_ZH.md)
- [最终状态](../../../artifacts/round15/FINAL_STATUS.json)
- [最终配置](../../../artifacts/round15/FINAL_CONFIG.json)
- [统一性能表](../../../artifacts/round15/BENCHMARK.csv)
- [执行预算](../../../artifacts/round15/ROUND15_EXECUTION_BUDGET.json)
- [最终哈希清单](../../../artifacts/round15/HASHES.json)
- [执行计划](../../plans/2026-09-28-round15-npu-ops-segmented-graph.md)

本目录是对机器可读交付的解释和导航；出现字段差异时，以 `artifacts/round15/` 中的原始 JSON、CSV、归档和 SHA-256 为准。
