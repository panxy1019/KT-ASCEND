# Round15 current execution

Status: `ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION`

Round15 已按用户在执行中冻结的简化口径完成：A、B 的请求证据各由一个 fresh 基线进程和一个 fresh 候选进程组成，每个 W1/W2 有两次 measurement。该证据只回答方向和大致幅度，不构成四进程 ABBA、统计稳定性或生产晋级资格。

## P0

- 两个 fresh S0 profile 块通过，运行身份、输出、Hybrid 路径、五桶存储、QSA/GDN provider 一致。
- 两块均显示 QSA 是采样到的最大 NPU 成本：Decode attention 每次约 6.06 ms，indexer 约 2.8–3.0 ms；GDN 约 1.04–1.11 ms，routed-expert parent 约 1.95–2.02 ms。
- A3 实际环境为 aarch64 Ubuntu 22.04、Python 3.11.15、torch 2.9.0、torch_npu 2.9.0.post2、CANN 9.0.0。

## A：成熟 QSA 路径

- v339/v340 两个独立组件进程通过数值、20 轮生命周期、allocator 和清理合同。完整 parent 的候选/基线耗时比约为 0.170–0.198。
- 当前短上下文的 selected set 已是完整 causal set；`npu_lightning_indexer` 的 520-group Top512 形状不属于现有 <=256-token 资格，因此请求候选冻结为 `FULL_SET_CAUSAL_SHORT_CONTEXT`。
- v341 基线和 v342 候选各完成 14/14 请求，14 组输出 token 完全相同，有限 logprob、cancel/reuse 和候选真实命中均通过。
- W1 方向上 TTFT/E2E/串行吞吐约改善 3.6%/3.7%/3.8%；W2 的 TTFT、TPOT、E2E 和串行吞吐分别约退化 3.25%、0.41%、0.63% 和 0.63%。
- 结论：`ROUND15_A_DIRECTIONAL_NO_REQUEST_GAIN_KEEP_DEFAULT_OFF`。

## B：resident-MoE Decode 分段图

- 安全边界只包含 Decode batch1 的 resident NPU MoE 子调用。CPU Expert、PLE、QSA、GDN、状态、调度和取消均留在 eager；Prefill 在执行前回退 eager；没有跨层 deferred。
- v356 暴露生产 `topk_ids=torch.int32`，而探索组件固定了 int64，因此图未命中。该次只能证明 eager fallback 正确，不能作候选性能数据。
- v357 用真实 int32 合同完成 20 轮：relative L2 和 max abs 均为 0，重复 hash 完全相同；host/device 中位耗时分别从 0.483/0.364 ms 降为 0.347/0.241 ms。
- v358 在第一层首次 capture 自检失败，未发布错误输出。根因是 capture 后直接读取 `graph.output`，而组件路径始终先执行 replay。新增回归测试后，最小修复为首次 capture 后先 replay，再验证并消费输出。
- v360 完成 14/14 请求、取消/复用和清理；48 层 capture、48 个 capture event、192 个 replay marker 均真实命中，首次自检最大 relative L2 为 0，13 组可对齐输出 token 完全相同。
- 对 v354 基线，v360 的 W1 TTFT/E2E/串行吞吐约改善 3.97%/3.25%/3.36%；W2 TPOT/E2E/串行吞吐约改善 1.20%/0.72%/0.73%，但 W2 TTFT 退化 5.29%。owned/whole-card HBM 均增加 984 MiB。
- 结论：`ROUND15_B_DIRECTIONAL_SMALL_GAIN_BELOW_PROMOTION_GATE`，候选保持 OFF。

## 最终配置与资源

最终配置保持 Round14a S0：Uniform Q32、C128 NUMA2/3/4/7、MXFP4、BF16/NZ29、chunk32、PLE ASYNC、Vectorized Prefill、DIRECT_BF16_ROW Decode、promoted GDN、H129、TP1、并发1；A/B 新候选均关闭。

实际预算：13 个模型进程、83 次 generation、5 次取消控制、4 个组件进程、0.8372649977 NPU device-hour，均低于硬上限。每次自有运行均完成 guard ACK、容器删除、归档/SHA 后的定向只读文件缓存退休；未修改训练或其他外部任务。

最终只读核验时 A3 出现新的外部 `qwen38-vllm-*` 容器。它们不是 Round15 自有对象，未被停止或修改；Round15 已无待执行设备实验。

## 交付入口

- `A_PROVIDER_AND_CANDIDATE_STATUS.json`
- `B_GRAPH_BOUNDARY_AND_RESULT.json`
- `A_REQUEST_DIRECTIONAL_RESULT.json`
- `B_REQUEST_DIRECTIONAL_RESULT.json`
- `BENCHMARK.csv`
- `FINAL_CONFIG.json`
- `FINAL_STATUS.json`
- `FINAL_REPORT_ZH.md`
- `ROUND15_EXECUTION_BUDGET.json`
- `HASHES.json`
