# Round15 A 线：QSA 成熟算子研究

## 1. 为什么选择 QSA

P0 的两个 fresh profile 块都显示，W2 Decode 的 QSA attention 每次采样调用约 6.1 ms，indexer 约 2.8–3.0 ms，高于 GDN、resident MoE、CPU result wait 和 PLE exposed wait。A 线因此冻结为 QSA index-selection 家族，不同时修改 GDN、MoE、PLE 或调度。

候选默认关闭，Prefill 保留现有 vectorized provider，未资格形状在执行前回到 eager reference。

## 2. 成熟路径审查

donor vLLM-Ascend 中存在 `npu_lightning_indexer` 路径，但本地当前窄配置有一个关键语义事实：在已资格的短上下文范围内，completed causal groups 数量不超过 Top-K 容量，选中集合就是完整 causal prefix。`npu_lightning_indexer` 真正有意义的 520-group、Top512 形状超出现有 <=256-token 资格。

因此 A 的请求候选不是强行移植新算子，而是一个精确的成熟化 fast path：

```text
FULL_SET_CAUSAL_SHORT_CONTEXT
```

它在 selected set 等于完整 causal set 时直接构造有序逻辑索引，跳过没有选择作用的 ranking；保持 per-query causal、GQA、KV/index 身份、padding 和 downstream attention 不变。长 Decode、Prefill、多 query、不合格 dtype/geometry 全部在修改状态前 fallback。

## 3. 组件合同

v339 和 v340 是两个独立 fresh-admission 组件进程，覆盖 layer 3/23/47 和 context 32/64/128/256，共 12 个完整 parent 场景/块。每个场景要求：

- selected set 与 eager reference 完全相同；
- downstream relative L2 不超过 `0.001`；
- 重复 hash 一致；
- 真实输入变化能改变输出；
- 20 轮复用无逐轮资源增长；
- source、候选身份、allocator 和清理通过；
- 每个完整 index-selection parent 的候选/基线中位数不超过 `0.85`。

两块 24 行全部通过，实际 candidate/baseline 比值约为 `0.170–0.198`。组件状态是 `PASS_ROUND15_A1_QSA_TWO_BLOCK_COMPONENT_GATE`。

该大幅局部收益只说明 index-selection parent 变快，不等于整个 QSA attention、完整模型层或请求按同样比例变快。

## 4. 请求接入

请求进程使用不变的 S0，唯一变量是 `FULL_SET_CAUSAL_SHORT_CONTEXT`。v341 为 baseline，v342 为 candidate；每边：

- 5 类 canary；
- 1 个多步 Decode；
- 1 个取消控制；
- 1 个同进程 reuse；
- W1、W2 各 1 warmup＋2 measurement；
- 合计 14 generation attempts、1 abort control；
- profiler OFF。

候选在全部 12 个 QSA 层真实命中；两臂 14 组输出 token 完全一致，有限 logprob、取消和复用全部通过。whole-card HBM 峰值为 35048 MiB 对 35047 MiB，owned HBM 都是 25754 MiB，没有通过资源变化换取速度。

## 5. 请求结果

| Workload | 指标 | v341 baseline | v342 candidate | 变化 |
|---|---:|---:|---:|---:|
| W1 128/16 | TTFT | 2.8650 s | 2.7607 s | 改善 3.64% |
| W1 128/16 | TPOT | 0.28208 s | 0.27155 s | 改善 3.73% |
| W1 128/16 | E2E | 7.0964 s | 6.8340 s | 改善 3.70% |
| W1 128/16 | 串行吞吐 | 2.2547 tok/s | 2.3412 tok/s | 改善 3.84% |
| W2 128/128 | TTFT | 2.7720 s | 2.8621 s | 退化 3.25% |
| W2 128/128 | TPOT | 0.26651 s | 0.26762 s | 退化 0.41% |
| W2 128/128 | E2E | 36.6190 s | 36.8494 s | 退化 0.63% |
| W2 128/128 | 串行吞吐 | 3.4954 tok/s | 3.4736 tok/s | 退化 0.63% |

W1 呈现约 3.6–3.8% 的方向性改善，但 Decode-heavy W2 没有收益。组件中减少的是 QSA 的一个局部选择步骤；W2 的完整关键路径仍包括 attention、48 层主干、GDN、PLE 和异构 Expert，局部收益被其余成本和运行波动吸收。

## 6. 去留结论

```text
ROUND15_A_DIRECTIONAL_NO_REQUEST_GAIN_KEEP_DEFAULT_OFF
```

A 被判定为“组件正确且快，但请求不晋级”。这不是兼容性失败，也不是成熟算子普遍无效；它只说明在本地当前短上下文、BF16/TP1、H129 窄配置中，该 full-set 快路径没有改善主观察点 W2。

由于用户缩减了请求实验，本数据不构成四进程 ABBA 或统计稳定性证明。即使 W1 方向为正，也不能据此打开默认开关。

机器可读证据：

- [A1_COMPONENT_RESULT.json](../../../artifacts/round15/A1_COMPONENT_RESULT.json)
- [A_PROVIDER_AND_CANDIDATE_STATUS.json](../../../artifacts/round15/A_PROVIDER_AND_CANDIDATE_STATUS.json)
- [A_REQUEST_DIRECTIONAL_RESULT.json](../../../artifacts/round15/A_REQUEST_DIRECTIONAL_RESULT.json)
- [A 组件实现](../../../artifacts/round15/round15_a_qsa_candidate.py)
- [A comparator](../../../artifacts/round15/compare_round15_a_model.py)
