# Round17 A 线：完整 QSA parent 去留

日期：2026-09-29
结论：`A_NO_BOUNDED_PARENT_CANDIDATE`
默认配置：保留 S0，不启动 QSA 请求 ABBA。

## 两块实测

v512 与 v514 均选取 W2 measurement 的 125 个稳态 Decode batch，去掉首尾 batch。每个 token 实际调用 12 个 QSA 层；Host 单调时钟和 NPU event 分开统计，没有把两者相加。

| 指标（12 层合计） | v512 | v514 |
|---|---:|---:|
| QSA parent Host | 73.417 ms/token | 70.077 ms/token |
| QSA parent device event | 73.666 ms/token | 70.327 ms/token |
| indexer Host | 35.772 ms/token | 34.125 ms/token |
| sparse score/softmax/PV Host | 9.128 ms/token | 8.962 ms/token |
| sparse score/softmax/PV device | 9.587 ms/token | 9.420 ms/token |
| 未被 immediate child 覆盖的 Host parent | 19.519 ms/token | 19.053 ms/token |

Host parent 与 device parent 很接近，说明这里主要包含真实的 Ascend 工作和依赖完成，不是一个纯 Python 空洞。

## A1：成熟 sparse-attention 原语

冻结 Ascend 栈中可复用的成熟入口是 `torch_npu.npu_fused_infer_attention_score` 一类 score/softmax/PV 原语。它能替换的是当前约 9 ms/token 的 sparse attention core，不能承担 Qwen3.8 的压缩 key indexer、signed ranking、causal selected-set、GQA 映射、K/V state 更新和模型特定 metadata。

即使做一个不可能的上限假设：替换后 sparse core 成本完全为零，完整 parent 的 Host 改善也只有 **12.43% / 12.79%**，device 改善只有 **13.01% / 13.39%**。这已经低于冻结的 **20% complete-parent 组件门**；真实 adapter、layout 和 state 成本只会进一步压低收益。因此 A1 在写代码前被硬上限否决。

这不是说成熟原语本身无效，而是说它覆盖的 parent 比例不足以让候选进入本轮请求实验。

## A2：Decode 长路径整体融合

真正覆盖超过 20% parent 的候选必须把 indexer、selected K/V materialization、state write 和 sparse attention 放进同一个 Qwen3.8 专用 Ascend 后端。当前冻结栈没有一个成熟原语满足这组 causal/GQA/state 合同。

历史因果实验进一步限制了局部切法：Round15 的同数学 selector 候选在两个组件块中快约 5–6 倍，selected set exact，并在完整模型命中全部 12 层；但 profiler OFF 的 W2 TPOT 从 266.511 ms 变为 267.615 ms，慢 0.41%，串行吞吐下降 0.63%。因此只融合 selector 或 indexer 的一小段不能作为请求收益候选。

能够继续推进的方案不再是薄适配，而是新的模型专用 fused-QSA backend：它必须同时实现压缩 indexer、K/V gather、sparse GQA、state generation、fallback 和取消语义。这超过“最后一轮、最多两个有根据的局部实现”的有界范围。

## 结论边界

- 没有候选达到 20% complete-parent 理论门，因此没有消耗 A 线模型 smoke 或请求 ABBA。
- 结论是“本轮没有有界可晋级候选”，不是“未来完整 fused QSA 永远无收益”。
- 若未来重启，入口应是完整模型专用后端，而不是继续拆 selector、softmax 或单个 projection。

主要机器证据：`QSA_PARENT_ACCOUNT.json`、`A_QSA_DISPOSITION.json`、`raw/profile-a-v512/round17-analysis.json`、`raw/profile-b-v514/round17-analysis.json`。
