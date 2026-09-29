# Round16 文档索引

Round16 比较两条承载路线：

- **A：SGLang + KT**。保留当前 S0，重新定位真实请求关键路径，只在直接证据支持时研究精确同层 overlap 或更大的 NPU 分段图。
- **B：vLLM-Ascend + KT Expert**。将离线 scaffold 推进为真实的 512 逻辑专家／每层 32 个 NPU 物理专家实现，并完成完整模型和请求对照。

本轮最终状态为 **`ROUND16_PARTIAL_WITH_GAPS`**。B 已完成真实集成和四进程请求测量，但性能没有晋级；A 完成两个 fresh profile 块，却没有取得满足冻结合同的直接关键路径区间，因此没有创建优化候选。默认继续使用 S0 SGLang + KT。

## 阅读顺序

1. [范围、基线和最终判断](01_SCOPE_BASELINE_AND_FINAL_DECISION.md)
2. [A 线：关键路径归因与候选去留](02_A_CRITICAL_PATH_AND_DISPOSITION.md)
3. [B 线：Physical-Q32、选择性加载与 Hybrid runner](03_B_Q32_VLLM_INTEGRATION.md)
4. [KV 资格边界：为什么单页 `unsqueeze(0)` 不构成闭环](04_KV_QUALIFICATION_BOUNDARY.md)
5. [四进程公平请求基准](05_REQUEST_BENCHMARK.md)
6. [资源、缓存退休、证据与复现](06_RESOURCES_EVIDENCE_AND_REPRODUCTION.md)
7. [代码地图与测试说明](07_CODE_MAP_AND_TESTING.md)

## 权威产物

- 中文总报告：[ROUND16_FINAL_REPORT_ZH.md](../../artifacts/round16/ROUND16_FINAL_REPORT_ZH.md)
- 机器终态：[FINAL_STATUS.json](../../artifacts/round16/FINAL_STATUS.json)
- 当前执行状态：[ROUND16_CURRENT_EXECUTION.md](../../artifacts/round16/ROUND16_CURRENT_EXECUTION.md)
- B 运行时结论：[B_RUNTIME_RESULT.json](../../artifacts/round16/B_RUNTIME_RESULT.json)
- 四进程比较：[ROUND16_REQUEST_COMPARISON.json](../../artifacts/round16-b/ROUND16_REQUEST_COMPARISON.json)
- 完整基准表：[ROUND16_BENCHMARK.csv](../../artifacts/round16/ROUND16_BENCHMARK.csv)
- KV 资格边界：[ROUND16_B_KV_QUALIFICATION.json](../../artifacts/round16/ROUND16_B_KV_QUALIFICATION.json)
- 资源回执：[ROUND16_RESOURCE_RETIREMENT.json](../../artifacts/round16/ROUND16_RESOURCE_RETIREMENT.json)
- 最终验证：[ROUND16_FINAL_VERIFICATION.json](../../artifacts/round16/ROUND16_FINAL_VERIFICATION.json)
- 证据哈希：[ROUND16_FINAL_EVIDENCE_SHA256.json](../../artifacts/round16/ROUND16_FINAL_EVIDENCE_SHA256.json)
- GitHub 发布内容清单：[GITHUB_PUBLISH_CONTENTS.json](../../artifacts/round16/GITHUB_PUBLISH_CONTENTS.json)

## 一句话结论

vLLM-Ascend 已经能用同一套 KT 专家核心正确运行这个冻结 Qwen3.8 配置，但 eager B16 在两个独立 W2 配对块中都比 S0 慢：TPOT 分别增加约 22.5% 和 19.4%，串行吞吐分别下降约 17.2% 和 15.0%。这足以拒绝默认承载切换。A 线的 CPU exposed wait 与 Host launch 仍缺直接隔离证据，不能从嵌套 span 或跨时钟事件中推算。
