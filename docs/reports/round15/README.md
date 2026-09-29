# Round15｜成熟 NPU 算子与 KT 分段图执行

本目录整理 Round15 的基线、取证、实现、失败、安全、性能、资源和复现证据。完整导航从 [00_README.md](00_README.md) 开始。

最终状态：

```text
ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION
PERFORMANCE_OBJECTIVE = NOT_ACHIEVED
```

本轮完成了两条候选的组件与真实模型接入：QSA full-set short-context 路径在组件中明显加速，但 W2 请求没有收益；Decode batch1 resident-MoE 分段图在 48 层真实 capture/replay，W2 E2E 和串行吞吐仅方向性改善约 `0.72%` 和 `0.73%`，同时增加约 `984 MiB` HBM 并使 TTFT 退化 `5.29%`。两项都未晋级，默认配置继续使用 Round14a S0。

请求实验按用户最终要求缩减为每条候选线一个 fresh 基线进程和一个 fresh 候选进程，每个 W1/W2 各两次 measurement。因此这些请求结果用于判断大致方向，不能表述为四进程 ABBA、统计稳定性或生产资格。

## 文档

- [总索引与证据边界](00_README.md)
- [基线、范围与实验方法](01_BASELINE_SCOPE_AND_METHOD.md)
- [P0 Provider 与 Trace 取证](02_P0_PROVIDER_AND_TRACE.md)
- [A 线：QSA 成熟算子研究](03_A_QSA_OPERATOR_STUDY.md)
- [B 线：Decode 分段图研究](04_B_SEGMENTED_GRAPH_STUDY.md)
- [故障、安全与共享 NPU 策略](05_FAILURES_SAFETY_AND_SHARED_NPU.md)
- [性能、资源与预算账本](06_PERFORMANCE_RESOURCES_AND_BUDGET.md)
- [证据与复现指南](07_EVIDENCE_AND_REPRODUCTION.md)

权威机器可读交付位于 [`artifacts/round15/`](../../../artifacts/round15/)。Markdown 文档负责解释证据；若字段出现差异，以原始 JSON、CSV、归档及 SHA-256 清单为准。
