# Round15 P0：Provider 与 Trace 取证

## 1. P0 的目的

P0 不是再做一轮泛化 profiler，而是回答三件事：S0 实际加载了什么 provider；W1/W2 的 Prefill 与 Decode 分别把时间花在哪里；A/B 应该选择哪个可独立验证的边界。为避免单个进程偶然性，P0 使用两个 fresh S0 块，并比较模型、fixture、输出 token、Hybrid 路径、五桶存储和 provider 身份。

最终 P0 状态为 `PASS_ROUND15_P0_TWO_FRESH_BLOCKS`。v335 的原 supervisor 控制失败被保留，只严格 salvage 已完成的测量；v338 是完整 process PASS。两块的输出 IDs、Hybrid 路径和五桶存储完全一致。

## 2. 实际 Provider

| 模块 | 本地实际入口 | Round15 解释 |
|---|---|---|
| QSA | `Qwen4ReferenceIndexer.forward` → `round9.qsa.select_tokens` / `sparse_gqa_candidate` | Prefill 使用 vectorized 路径，Decode 使用 direct BF16 row |
| GDN | `Qwen4ReferenceGatedDeltaNet.forward` → promoted native core → `sigmoid_gated_rms_norm` → `out_proj` | 保留本地 sigmoid 语义，不能替换为 swish |
| Resident MoE | `UnquantizedFusedMoEMethod.forward_npu` → routing/grouped matmul/swiglu/finalize | Q32 BF16/NZ29；是 B 线唯一捕获区域 |
| CPU Expert | KTEP CPU submit/wait/merge，C128＋MXFP4 | 必须留在图外 |
| PLE | `Qwen4ReferencePLE`＋既有 async prepare、ready/consumed 事件 | Host gather、H2D 和 consumer 依赖保持不变 |
| Graph API | torch_npu 2.9 `NPUGraph`、`graph`、`replay`、`reset` | 只用于内部 resident-MoE segment，不启用整模型 Graph |

完整映射见 [PROVIDER_MAP.md](../../../artifacts/round15/PROVIDER_MAP.md)。

## 3. W2 Decode 热点

下表给出两个 fresh 块在 W2 Decode 的每次采样调用中位数。Host 和 device 是不同计时域，不相加；表格只用于比较顺序和复现性。

| 组件 | v335 Host / Device ms | v338 Host / Device ms | 观察 |
|---|---:|---:|---|
| QSA attention | 6.064 / 6.129 | 6.062 / 6.086 | 两块均为最大单项 |
| QSA indexer | 2.894 / 2.989 | 2.798 / 2.907 | 排名/索引仍显著 |
| Routed Expert parent | 1.994 / 2.019 | 1.950 / 1.974 | 包含 CPU/NPU routed expert 父路径 |
| PLE consumer | 1.991 / 2.032 | 2.010 / 2.043 | consumer 成本稳定 |
| GDN | 1.057 / 1.113 | 1.039 / 1.091 | 已晋级实现后不再是第一热点 |
| PLE prepare | 0.818 / N/A | 0.869 / N/A | Host prepare |
| NPU resident | 0.405 / 0.444 | 0.400 / 0.438 | B 线能安全捕获的区域很小 |
| CPU result wait | 0.371 / N/A | 0.345 / N/A | 暴露等待低于 QSA |
| CPU submit | 0.379 / N/A | 0.362 / N/A | 下发成本存在但不是最大项 |
| PLE exposed wait | 0.132 / N/A | 0.135 / N/A | 不支持重启大规模 PLE overlap |

W2 有 127 个 Decode batch，每个 batch 在 observer 中对应三个目标层调用，所以 QSA、GDN、resident 和 routed expert 各记录 381 次。这个采样结构解释了调用次数，不能把 381 次当作 381 个模型层。

## 4. Prefill 与 Decode 的差异

W2 EXTEND 中，QSA attention 和 indexer 的 device 中位数在 v338 分别约 `24.322 ms` 和 `21.355 ms`，明显高于 Decode 单步；GDN 约 `4.107 ms`，routed expert 约 `3.667 ms`。这说明 Prefill 与 Decode 的热点形态不同，也解释了为什么 A 的短上下文选择优化可能改善 W1，却不一定改善 W2。

P0 没有把 observer 的所有时间相加成 E2E：QSA parent、indexer、device event、Host span和其他 stream 可能重叠或嵌套。只有直接可归因的完整 parent 才作为组件候选门。

## 5. 候选选择

P0 后的决策是：

- **A 先研究 QSA**：两块 profile 均把 QSA attention/indexer 排在最前。成熟 `npu_lightning_indexer` 被审查，但其 520-group Top512 使用域超出当前短上下文资格；因此选取数学上精确的 full-set short-context 分支。
- **B 只捕获 resident NPU MoE 子调用**：这是现有依赖图中可以明确留在 NPU、形状固定并在消费前完成的区域。它很小，所以组件成功也不预示大请求收益。
- **不选择 PLE overlap**：暴露等待约 0.13 ms，且已有 ready/consumed 语义；当前证据不支持再打开复杂异步项目。
- **不继续 GDN**：Round14a 已晋级 GDN，当前 profile 中它不再是首要热点。

## 6. P0 控制故障的证据边界

- v331 在模型启动前被继承预算 stage gate 拒绝，未产生模型测量。
- v333 完成四请求，但缺少五桶 producer，因此整体失败。
- v335 修复 producer 后完成测量；原 supervisor 仍因 profiler 名称、训练进程识别和已删除容器复查问题失败。严格 salvage 只接纳可独立校验的测量，原 FAIL 不改写。
- v338 在相同合同下完整通过，作为第二个 fresh 块。

机器可读原始结果见 [P0_PROFILE_RESULT.json](../../../artifacts/round15/P0_PROFILE_RESULT.json)，协议见 [P0_PROFILE_PROTOCOL.json](../../../artifacts/round15/P0_PROFILE_PROTOCOL.json)。
