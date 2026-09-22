# Round6.2 阻塞分析：Deferred exposure、真实 decode 与 batch

日期：2026-09-15。状态：**研究结论，待审阅；不是新一轮执行结果。**

本次只阅读源码、既有路由元数据和官方资料，没有启动新模型实验，
没有读取 QD2 跨模式 logit 差异，也没有访问 HD2 模型结果。

## 1. 结论先行

当前不是“CPU 完全不工作”，也不是“已证明 Deferred 数值不正确”。
准确状态是：成功采集并审计的 QD2 中，仅343个位置具备实际 Deferred
暴露，未达到冻结要求500。工具在数值比较前正确停止。

最值得优先验证的机制是：**官方按整个 forward 的 expert-ID 并集保护，
而测试工具把逐步生成改成了每一步完整前缀 prefill 重放。**
两者组合既容易压低 Deferred 暴露，又重复计算大量前缀。

因此优先修复测试执行方式和样本设计，而不是为了达到500去改官方
selector、增加延迟 expert、换高频 expert、降低门槛。
batch 可以研究，但它同时影响 Deferred 选择语义，不能只当作无害加速参数。

## 2. 已证明的事实与尚未证明的部分

| 项目 | 证据状态 | 结论 |
| --- | --- | --- |
| capture 启动污染 | A3_VERIFIED | 旧运行多出同一 HEALTH_CHECK 的两个 forward，旧运行永久 INVALID |
| 修正后的完整采集 | A3_VERIFIED | 8192 positions、16384 forwards、自带/独立矩阵/事务审计全部通过 |
| exposure 不足 | A3_VERIFIED + LOCAL_RECOUNT |343<500，未计算跨模式数值 |
| selector 按全 forward expert-ID 并集保护 | CODE_INSPECTED | 本地与官方所查版本一致 |
| 当前矩阵是重复完整前缀 prefill | CODE_INSPECTED | 每个位置独立请求，input_ids=prompt+baseline[:t]，max_new_tokens=1 |
| 前缀越长时 exposure 更少 | LOCAL_RECOUNT | 相关性存在；不能单凭分桶证明全部因果 |
| 真实 KV-cache decode 能否达到500 | NOT_RUN | 必须用独立 pilot 实测，不能预先宣布 |
| batch=2/4 是否安全、是否更快 | NOT_RUN | 需要独立边界、确定性、质量与性能验证 |

停止点：`5e708dc63d07afe4d79d26f0bdbb97d95ad3aaba`。
旧 ADR：`../round6_2/ADR_QD2_INSUFFICIENT_EXPOSURE.md`，不得倒改 PASS。

## 3. 为什么“max_deferred=1”没有保证每个位置都有 Deferred

本地 `kt-kernel/python/experts_base.py:625`：

1. 对 `[M, topK]` 的每一行选出最高分 protected_k 个 expert。
2. 将这些 expert IDs 展平，写入全局 `protected_flag[num_experts]`。
3. 任何行上属于该保护集合的 expert 都被立即执行。
4. 剩余候选还必须属于 CPU-owned experts，才形成实际 CPU Deferred。

这里的 M 是当前 MoE forward 的 token 行数，不等于 HTTP 请求数。
单请求长 prefill 的 M 也很大。D1 的 topK=6、max_deferred=1，
意味着 protected_k=5；并不意味着每个 token 必定延迟一个 CPU expert。
L7 只有 {16,38,41,50} 四个 CPU expert，L8 为 terminal，不能继续 defer。

例子：token A 的低分候选 expert16，本来可能延迟；但 token B 的高分
保护列表中也有 expert16，则两个 token 上的 expert16 都改为立即执行。
因此，增加同一次 forward 的 token 行，在保持既有路由/权重不变的
条件下，只能扩大或保持保护集合；不能给既有行凭空增加 Deferred。
真实合批还可能改变 GEMM 数值与路由，因此实际结果须实测。

已有本地测试 `test_prefill_selection_uses_batch_union_expert_protection`
明确验证了这种机制。它不是刚刚发现的实现缺陷，而是此前对
qualification workload 暴露率的预期没有充分考虑它。

## 4. 官方对照

本次联网查询的官方 HEAD：
`0dce4c9b097710951d1b3e232f50b973ff0a5505`。
所读 `experts_base.py` 原始 SHA256：
`2aa8f6e0fc225c592ad198eb21779a19c4e29cca1cc598eb149c54a2bd836720`。
这是研究快照，不是要合并/升级的目标；当前生产谱系保持不动。

官方同样使用 top-k 后对 expert IDs 做 scatter/gather 的全 forward 保护，
并带有 qlen 超过 C++ buffer 容量时的 fail-fast。不能把 selector 改成
逐 token 独立强制延迟，然后称为“照搬官方修复”。
[官方固定版本源码](https://github.com/kvcache-ai/ktransformers/blob/0dce4c9b097710951d1b3e232f50b973ff0a5505/kt-kernel/python/experts_base.py)

官方 README 提供 Deferred 预算、CPU 线程、prefill 阈值和并发配置说明，
也展示 LLAMAFILE 服务配置；这些是调优入口，不是本项目
DeepSeek-V2-Lite/Ascend/BF16/D1 批量资格通过的证据。
[官方 KT-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)

已检查的官方通用实现包含 Ascend callback-worker、framework copy 和
显式等待逻辑，但本次不迁移 callback/Graph/runtime 架构，以免和采样
及 batch 风险混合。官方 GitHub API 本次有 rate limit，未通过 API 重新
解析当前 HEAD 的 SGLang gitlink；不能虚构新的 child SHA。下一轮如需
进一步对照，先冻结可复现的 parent/child 快照，仅做语义差异审计。

## 5. 既有数据说明了什么

输入：成功 retry1 的 manifest.json.gz；raw SHA
`637f7d730cccdb037bd395d1fc0489598ef9eb31e186f6cb9fff83719d9e5e54`。
仅聚合路由元数据，没有计算数值差异。

| 目标前缀长度 | 位置数 | 暴露位置 | 暴露率 |
| --- | ---: | ---: | ---: |
|8–32|2421|191|7.89%|
|33–64|4096|125|3.05%|
|65–92|1675|27|1.61%|
|全部|8192|343|4.19%|

生成位置0–15/16–31/32–47/48–63，每组2048位置，暴露分别
161/66/67/49。prompt 内容与路由分布也会影响这些比例，不能将全部下降
精确归因于一个因素；应通过固定已捕获 operands 的 selector 审计确认。

两层合计 CPU route event 计数510194，而实际 Deferred route record440。
这是每个请求前缀重复计算后的事件总数，不能当成独立 token 数。
同样，343不是模型质量分数，更不是数值不一致的数量。

## 6. 为什么运行慢

当前采集客户端每个目标位置发一次HTTP请求，重算完整前缀，并额外
产生一个 serving lookahead。8192目标共重算373440个目标前缀 token 行，
平均每次45.586行，另有8192个lookahead。

同样128个prompt若使用一次prefill+63个逐步decode，目标计算行数约为
1803+128×63=9867，额外lookahead单独计数。373440/9867≈37.8，
**仅是理论 token-row 工作量比，不是预计37.8倍实际加速。**
decode kernel、KV访问、CPU执行、同步、I/O与调度成本不同。

其他明确开销：

- 每请求前后全目录 rglob，随已保存 capture 数增长，累计近似二次扫描。
- 每位置保存完整多层张量，执行 D2H、序列化、磁盘写入和再次 torch.load。
- batch=1、HTTP串行；CPU池仅16线程且固定亲和，可能有host瓶颈。
- 完整capture本身有观察开销，不能用它的NPU利用率判断生产上限。

尚缺分段计时，不应把所有低利用率都归因于 batch，也不应承诺扩大
batch 就线性提速。先测模型时间、D2H、dump、读取/审计、HTTP等待各项。

## 7. batch 的直接回答

**可以纳入下一轮，建议从2、4做独立pilot，不直接上8/16。**

| 参数/概念 | 实际作用 | 不能混淆 |
| --- | --- | --- |
|客户端并发数|同时提交多少请求|不保证同一次NPU forward的batch |
|max-running-requests|调度并发上限|不等于固定batch |
|实际decode batch|同次forward的序列数|会改变保护集合与跨请求依赖 |
|prefill token行数|多个请求展开后的总token数|必须受CPU/NPU buffer容量约束 |
|max-total-tokens|KV容量预算|不是简单的吞吐开关 |

当前telemetry有单request断言、`logits[-1]`读取、shape `[1,V]`与全前缀
positions假设。直接提高并发将导致关联错误或漏记，甚至产生假通过。
必须新增 per-request row spans、logit row map、absolute positions、实际
batch membership、padding有效位以及各自完整history hash。

固定batch内的prompt共享保护集合，不再独立。若128prompt按4合批，
只有32个独立batch block，而不是128。沿用现有95% coverage和每族
alpha=.05/3时，32或64个block没有满足要求的order-statistic rank；
至少80个block才存在max-rank解，128个block对应rank127。
这是本地既有统计函数计算，不是新的批准contract。

建议 batch1 完成主数值资格；batch2/4另做固定分组的吞吐与正确性资格。
如要求正式batch4统计声明，按128独立batch block设计则需512个prompt，
而且分组、batch边界、执行模式必须在Q/H比较前冻结。

## 8. 应修与不应修

应修：真实decode teacher forcing、capture事务边界、batch关联、I/O效率、
先pilot再冻结样本量、与实际执行配置一致的统计单位。

不应修：为了500改selector为逐token延迟；budget1改2；改D1 placement；
根据Q失败挑高hit prompt；用batch降低暴露后宣称数值更好；把HD当pilot；
把旧Round6.2改成PASS。

正式执行方案见 `ROUND6_3_REVIEW_PLAN.md`。
