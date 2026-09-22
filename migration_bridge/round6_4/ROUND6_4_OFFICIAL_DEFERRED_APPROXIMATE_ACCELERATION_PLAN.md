# Round6.4 — Official Deferred Approximate Acceleration

## 基于封存 TP1 基线，接受官方跨层近似的完整执行方案

日期：2026-09-15  
状态：`PLAN_FOR_REVIEW`，本文件不是测试通过记录，也未启动新一轮 A3 任务。  
目标：DeepSeek-V2-Lite / Kunpeng / Ascend NPU0 / LLAMAFILE / sglang-kt。

> 本轮认可官方跨层 Deferred 是一种近似加速，不要求它在数学上等价于 Deferred OFF。
> 工程成功定义为：机制实现正确、运行稳定、质量代价符合事先约定、真实工作负载有收益。
> 不再把额外的全候选排序不变量，当成官方算法必须满足的条件。

## 1. 目标纠偏与历史边界

用户原始目标不是发明新的数值等价理论，而是在已封存的 CPU/NPU Hybrid 上引入官方近似加速。
前几轮把跨 backend 浮点兼容要求延伸到 Deferred 跨层近似，并额外要求任意候选对排序保持，
偏离了这个目标。本轮明确纠正这一点，不把验证工具的过强要求误称为生产功能缺陷。

历史结果保持原样：Round6.2 曝光不足；Round6.3 按当时冻结协议 REJECTED。
不改历史报告、旧 tag、旧 evaluator，也不把旧 QD3 改为 PASS。
新轮次使用独立的 `APPROXIMATE_ACCELERATION_CONTRACT.json`，不再沿用“先拟合 candidate
logit 阈值，再允许质量测试”的入口顺序。

本轮应回答四个工程问题：

1. 我们是否忠实执行官方“L 层贡献延迟至 L+1”的机制？
2. 是否保留 CPU/NPU ownership、权重、路由、buffer、stream 的正确性？
3. 对目标工作负载，任务质量损失是否可接受？
4. 在相同 batch、资源和真实曝光下，ON 是否比 OFF 更快？

## 2. 官方机制与本地实现范围

官方参考冻结版本：

- KT：`0dce4c9b097710951d1b3e232f50b973ff0a5505`。
- 对应已审计 SGLang：`3424f35d0e60f03b9e75d9b90941f8f4cb944524`。
- [官方专家实现](https://github.com/kvcache-ai/ktransformers/blob/0dce4c9b097710951d1b3e232f50b973ff0a5505/kt-kernel/python/experts_base.py)。
- [官方 Deferred race 修复 #1646](https://github.com/kvcache-ai/ktransformers/pull/1646)：相关修复已在本地谱系，不重复搬运。

其核心是：L 层分出低优先级 CPU expert，结果写入下一 slot，在 L+1 CPU 输出处 incremental merge。
L+1 的前序 attention/normalization 已经使用缺少这份贡献的输入，因此这是计算图近似，
不是仅仅改变同层浮点加法顺序。后续 KV 状态也可以不同。

迁移原则：沿用已经接入的官方机制，不重写 coordinator，不整包覆盖已经验证的 Ascend 适配。
只在语义 diff 找到具体缺失时进行最小 port；不可将升级动态 main、换模型、换量化混入本轮。
官方“功能存在”不是本配置“质量及收益已通过”的证明，后两者仍须实测。

## 3. 双基线、分支和已封存资产

### 3.1 不可变 reference

| 对象 | 冻结身份 |
|---|---|
| Round4A tag target | `096a6ef860a01dc0a3c8f2cc5618a396425ee09b` |
| Round5 tag | `round5-upstream-aligned-tp1-a3-verified` |
| Round5 tag target（本次本地解析） | `be5b48621db32fb894402e3e886060dea263cdc9` |
| Round5 SGLang | `412219b4b05a888808f0b88d2963a8b6164a0fda` |
| 当前调查结束提交 | `ba3494d9aedcc0e5623157832134e1e32a8ea44a` |

“基于封存版本”指保留可重复的 reference oracle 和清楚的谱系，不是删除 Round6 已完成的系统修复。
建议独立 reference checkout 固定 Round5 tag；执行候选分支从上述调查结束提交创建
`codex/round6-4-official-deferred-approximation`，逐项列出相对 Round5 的生产 diff。
如审计发现无关污染，再从 Round5 新建最小 port 候选，不能默默替换本次来源。

开工记录 parent/child HEAD、branch、dirty state、SO/wheel SHA 和容器实际加载路径。
已有 `third_party/llama.cpp` 修改归用户所有，不 add、reset 或丢弃。
A3 现有 checkout+overlay 不得冒充未修改的本地 commit。

### 3.2 固定模型与 D1

| 对象 | 冻结值 |
|---|---|
| 模型 | DeepSeek-V2-Lite |
| revision | `604d5664dddd88a0433dbae533b7fe9472482de0` |
| tokenizer SHA | `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669` |
| D1 producer | L7：CPU `{16,38,41,50}` |
| D1 terminal | L8：CPU `{8,25,33,55}`，禁止向后 defer |
| D1 placement SHA | `23edad71e52c05b1dc3a227ea6e27cacad0b4db6a18c4419647efae00530d39c` |
| CPU GGUF SHA | `4b42799ec7f143569692133aadd37f089ff0c18508411e5fd37911b62c8e323f` |
| 配置 | TP1，NPU0，CPU0–15，16 workers，LLAMAFILE，NPU BF16 |
| OFF / ON | 同 placement，deferred budget 分别 0 / 1 |

保留 BF16-visible expert boundaries、FP32 routed partial merge、framework Tensor copy_、
terminal guard、consumed-pending 清理及 CPU 空路由 no-op guard。

## 4. 明确取消和保留的验收要求

| 项目 | 本轮性质 | 原因 |
|---|---|---|
| ON/OFF 全 token 相同 | 不要求，报告差异 | 跨层近似可以改变输出 |
| ON/OFF stable top1 必须 100% 相同 | 不作为独立发布门禁 | 不以旧 logit reserve 定义近似算法可接受性 |
| 任意非 top1 候选排序不变 | 取消硬门禁，只作诊断 | 不是官方近似机制的保证 |
| top32 外 token / ambiguity membership | 只作诊断 | 不因 candidate universe 截断而误判系统错误 |
| 固定 500 exposure 才能进入质量测试 | 取消 | 不再使用原 prompt-tail 估计器资格链 |
| ON vs OFF 总 logits 的固定最大误差 | 不新增硬阈值 | 以质量、oracle、运行安全区分问题 |
| same-path 确定性 | 保留 | 相同配置与批次日程必须可重复 |
| routing/ownership/权重/共享专家 exactly once | 保留 | 近似不允许漏算、重算或混入别的模型 |
| CPU/NPU expert 同 operands 安全检查 | 保留已有回归 | 不能比较两条已分叉路径的不同 hidden 再归因 kernel |
| NaN/Inf、死锁、跨请求污染 | 硬失败 | 与近似无关 |
| 独立下游质量与真实性能 | 主要接受条件 | 近似加速必须体现质量—成本权衡 |

注意：不同 ON/OFF 历史上的 router IDs 可以不同。
exact routing 检查是在每一路径内部，核对该路径实际 router 输出与送入专家的 ID/weight，
或在固定同 operands 的局部实验中比较；不能要求所有后续跨路径路由始终相同。

CPU-not-hit exact 只用于**整个请求和因果历史从未有 CPU/deferred 贡献差异**的控制。
当前 token 没命中但 prefix 曾经 defer，仍可存在真实差异，不误判污染。

## 5. 本轮三个对照对象

- R：封存 Round5 非 Deferred reference，用于保护旧能力和最小回归。
- A：当前候选代码 + D1 + Deferred OFF，是 Deferred 开关 A/B 的直接基线。
- B：同一候选代码 + 同一 D1 + Deferred ON。

主要质量/性能比较 **B 对 A**，不能把不同 placement 的历史 All-NPU 分数当作即时对照。
R 对 A 的比较必须选共同支持的 placement/profile，并应用原非 Deferred 回归标准；
若 reference 不支持当前测试布局，应另用原封存 P1/P2/P3 compact profile，明确范围。
不把 D1 的新布局称为 Round5 已验证布局。

## 6. 旧证据复用与避免重新跑一切

可复用为已完成工程证据：

- observer OFF/reference/full、full/compact parity；实际 next-input 与采集关联；
- 7390/8192 曝光和 QD3 描述性数值分布；
- 两个失败历史的 production/private oracle 7800/7800 common tensors、130/130 logits exact；
- 旧底层回归作为预期值，而非新代码已通过的证明。

只在相关代码、配置或采集方式变化时重跑相应控制，不无条件重做三组 128×64 全 tensor capture。
旧 QD3 只能作为已观察的工程/校准数据；旧拒绝只在旧协议下解释。
HD3/FD3 若经日志、文件和访问审计证实从未生成模型输出，可作为本轮未观察验证集复用原字节，
明确是“继承未观察数据”，不是新造一份名字不同的 held-out。
如发现已生成或读过模型结果，转为开发集，另冻结新集，不隐瞒。

## 7. 执行总图

```text
批准新目标与质量容忍提案
  → WP0 provenance / 新契约冻结
  → WP1 封存回归 + D1 系统 smoke
  → WP2 batch1 runtime / oracle / 安全
  → WP3 早期无采集性能 pilot + batch2/4 工程资格
  → WP4 未观察数值画像与自由生成诊断
  → WP5 独立质量 A/B（不等待 logit 等价门禁）
  → WP6 正式性能 A/B
  → WP7 稳定性与最终回归
  → WP8 ADR / opt-in 功能发布 / 证据冻结
```

系统硬失败阻止后续该配置测试。数值排序变化本身不阻止质量测试。
性能短测没有收益时，仍可给出“实现正确、暂无加速收益”，不要无限调参直到变快。
本轮主要范围是 D1 batch1 及经验证的 batch2/4；不自动扩 D2/D3、多 block 或模型。

## 8. WP0 — 契约、输入和来源冻结

产物：`00_BASELINE_AND_SCOPE.md`、`01_APPROXIMATION_ADR.md`、
`APPROXIMATE_ACCELERATION_CONTRACT.json`、`corpora/`、`run_manifest.json`。

步骤：

1. 核对 tags、parent/child、模型、tokenizer、GGUF、placement、库路径和软件版本。
2. 对比当前 Deferred 与官方冻结源码，保留最小必要本地适配，列出每个差异。
3. 记录本次明确改变的是验收目标，不是把旧失败抹掉。
4. 冻结所有测试输入、题目 ID、prompt 模板、答案提取器、token budget、seed 和质量容忍值。
5. 下载数据若使用 Hugging Face，采用 `hf download --revision <完整SHA>` 并离线校验；
   A3 正式 run 不依赖网络，凭据不写入文档或日志。

## 9. WP1 — 保护原基线与最小系统测试

优先复用已有测试，不重新发明 Hybrid 层：

- R2A CPU expert：F32 GGUF、1408、BF16 输入语义、FP32 输出契约。
- R2B copy_/submit/sync、callback 生命周期。
- R2C ownership/mapping/routing once/shared once。
- 封存 P1/P2/P3 compact 非 Deferred 回归。
- D1 OFF/ON：prefill、decode、两层消费、terminal、全请求 no-hit。
- 同 operands CPU/NPU expert 继续原 relative-L2 ≤1e-2 安全门禁；不能用于整个跨层路径输出。

验收：无未解释的新回归。此阶段无需跑质量长矩阵。

## 10. WP2 — D1 runtime 与 oracle

### 确定性

- S6 原冻结 12 prompts，每个 10 次 64-token greedy，同一进程。
- 每个 prompt 检查 8/16/32 与 64 前缀一致。
- 新进程重跑 12×64，与前一进程相同配置输出 exact。
- 正式确定性使用 observer OFF；debug observer 单独验证，不用于性能计时。

### 系统和 buffer

- 每个 CPU-owned 路由 immediate XOR deferred；routing weight 恰好一次。
- producer L7、consumer L8；消费顺序正确、pending 完整 drain、terminal 不外溢。
- 真正 CPU/deferred forward 次数、每 expert 命中、worker 完成计数必须记录。
- ≥1000 个真实 CPU-expert forwards 的 lifecycle 回归，不以 padding 或 token 总数替代。
- 全请求 no-hit 使用已冻结控制，纳入 lookahead；若不成立如实报告而非挑方便样本。

### Oracle

复用两个失败历史的已验证 oracle；另在预先指定的 4 个工程请求上检查异步 production
与独立 buffer explicit replay。要求相同算术语义的共同 stage/输出一致。
oracle 是排除实现缺陷的工具，不要求 B 与 A 相同。

## 11. WP3 — 尽早验证 batch2/4，但不偷换加速来源

顺序：batch1 runtime PASS → batch2 短测 → batch4 短测。
不等大型 teacher-forced 数值分析结束才尝试 batch；不直接开启 batch8/16。

每个 batch：

1. 固定 cohort 和请求进入顺序，先 16 个工程请求、每个 64 tokens。
2. 检查实际运行 batch，不以 HTTP 并发数冒充 model batch。
3. 验证 row→RID、KV、buffer capacity、terminal、expert partition、padding/no-op、请求退出后的 reuse。
4. 固定 schedule 重复 10 次；两次进程重启验证相同 schedule 的确定性。
5. 记录真实 deferred routes、曝光比例。官方跨行保护并集可能抑制 deferred，这是算法行为。
6. 相同 batch 下分别测 OFF 与 ON。batch4 ON 比 batch1 OFF 快，不能被归为 Deferred 收益。

不要求 batch1 与 batch4 每个 token exact；batch 大小会改变保护并集与算术路径。
但实际每个 batch profile 都必须进行自己的质量验证，不能借用 batch1 结果直接宣称 READY。
正式质量默认仍跑 batch1；只有 batch2/4 自己通过对应质量协议，才可将其作为合格加速配置。
若批量路径尚不支持可靠的逐 RID teacher-force，就不做该模式的 teacher-force 声明，
先用自由生成和直接质量评测；不为完整报告字段大规模改调度器。

允许准备样本和离线统计并行，A3 NPU0 上模型进程顺序执行，不同 profile 不同时争用。

## 12. WP4 — 数值画像与自由生成：诊断而非替代质量

旧 QD3 结果直接归档为开发画像，不重新拟合2.125或另找一个“能过”的 reserve。
经未观察审计的 HD3：128×64；FD3：8×64，可用轻量采集生成新的描述性报告。

Matched-history 模式记录 full-logit abs/relative-L2、KL 或 JSD（稳定 FP64 log-softmax）、
top1 agreement、top32 变化、margin、当前/历史曝光分层。
以 prompt 为汇总单位，报告 p50/p95/p99/max；不将 correlated token 当作独立样本。
自由生成记录首次分叉、长度、重复率、输出格式、答案正确性，不要求首次分叉一定 near-tie。

保留 NaN/Inf、oracle 不一致、全请求 no-hit 污染等系统硬失败。
单纯 finite logit drift 或非 top1 排序变化只触发报告/归因，不禁止 WP5。
若出现输出坍塌，应核查并由质量/安全结果决定，而非事后添加新的数值阈值。

## 13. WP5 — 独立质量 A/B（本轮主要验收）

### 13.1 两档证据，不夸大小样本

**历史兼容 smoke**：原 C-Eval128 + GSM8K-MC128，原 manifest
`c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce`。
仅比较同环境 A/B，保留题目级结果；历史数据不声称是新 held-out。
低分 MC smoke 不能单独证明真实推理质量无损。

**主质量测试（推荐）**：

- GSM8K 官方 test 全集，执行时确认题数/版本；采用原生自由回答、固定答案提取，
  不用人为生成选项代替推理。计划规模约1319题，max_new_tokens=512。
- CMMLU test 固定分层抽取2048题，按学科比例、seed6404一次冻结；
  单字母选择题自由生成，max_new_tokens=32，不将正确答案提示给模型。
- 固定0-shot模板，不开 speculative/grammar workaround；两路径完全同输入、预算、解析器。
- GSM8K train 的32题和 CMMLU dev 只用于预先检查模板/解析器；不得据 test B 的分数改模板。
- EOS 正常停止；输出截断、无可解析答案计错误并单独报告，不能删样本。
- 不执行模型生成代码。代码/JSON 小集可静态解析，不能绕过沙箱运行不可信程序。

数据源：[GSM8K](https://github.com/openai/grade-school-math)、
[CMMLU](https://github.com/haonan-li/CMMLU)。执行前固定完整 source revision、许可、题目 IDs、
字节 SHA 和与已有 Q/H/quality 的重叠清单。已使用题目保留为历史重叠，不冒充未观察部分；
主报告分别列重叠/未观察子集，重要协议改动不得利用 B 的结果做选择。

### 13.2 质量容忍是产品选择，不从已看到的失败倒推

**本计划提请审阅的默认值：每个主 benchmark 允许最大绝对准确率损失2个百分点。**
这是近似加速的工程容忍提案，不是官方承诺、不是从 QD3 两处失败推导，也不是已经获批的数值。
批准本方案后，在模型 B 主质量输出产生前写入新契约。之后不得看结果临时改为3%/5%。

对同题配对结果 Xi = correct_ON − correct_OFF，估计 Δ=mean(Xi)。
主判断使用配对二项差的有效置信区间方法（paired score/exact 方法，先以合成数据验证实现）；
两 benchmark 各使用单侧97.5%下界，控制两个主终点的家族错误率。
配对 bootstrap20000次、seed6404仅作辅助图表，不能用零 discordance 的退化[0,0]
bootstrap区间声称总体零风险。

状态：

- `QUALITY_ACCEPTED_WITHIN_MARGIN`：两个 benchmark 的 Δ 下界均大于−0.02。
- `QUALITY_REGRESSION`：有明确超出允许损失的证据，或质量/解析发生严重异常。
- `QUALITY_INCONCLUSIVE`：区间跨过允许边界，证据不足；不冒充无损，也不自动把模型判坏。

同时报告 n、A/B原始正确数、A对B错/B对A错、invalid、截断、绝对及相对差和按学科分层结果。
“p>0.05 没发现显著差异”不等于非劣通过。
事先做基于假设 discordance 比例的功效表，说明1319/2048可能仍不足以证明2pp；
不保证样本数必然通过，不按中途 p 值任意追加题目。若 INCONCLUSIVE，本轮可以完成
工程实验报告但不能给 quality-qualified 标签，另行固定扩展样本或业务容忍决定。

### 13.3 Batch 的质量

batch2/4 先在预设小子集确认无明显异常；要对某 batch 发正式标签，必须跑相同主质量集，
以相同 batch OFF 为直接对照，报告 cohort 构造及实际 batch 分布。
多个 batch 的整体“至少一个通过”声明需预先处理多重比较；默认 batch1 为唯一主终点，
其他 batch 独立标实验状态，不从中择优替换主终点。

## 14. WP6 — 正式性能：必须真正加速

模型/placement/精度/CPU affinity/worker/资源限制不变，仅切换 Deferred 和经验证的 batch。
禁用 tensor dump、teacher-force、逐 forward 磁盘记录；预热不纳入计时。

- 在独立性能集固定短/中/长输入、固定64/256输出预算（在测试前定义形状）。
- 每 profile 使用固定 ABBA 顺序，至少5个成对测量 block；报告全结果，不挑最好一轮。
- 同 batch 比较 ON/OFF，报告端到端吞吐、TTFT、decode tok/s、P50/P95请求耗时、RSS/HBM。
- 固定长度吞吐测量单独标记 ignore_eos；真实 EOS 自由生成耗时作为另一指标。
- 用独立短 instrumentation run 记录 deferred route/overlap，不能把采集速度当生产速度。
- 看得到 CPU expert 工作和独立 NPU 工作的重叠；不为了通过偷偷改成 sequential。

性能结论：

- `SPEEDUP_VERIFIED`：主性能 workload 的成对速度比置信下界>1，且重复 block 稳定；
  其他形状的负收益如实列出。无需擅自要求>10%/20%。
- `NO_MEASURABLE_SPEEDUP`：无明确收益，保持 opt-in实验功能，不能宣传加速完成。
- `RESOURCE_CONFOUNDED`：共享集群干扰使比较不可归因，仅报告测量值，不宣布赢或输。

batch 变大但 actual deferred exposure 减少时，分开报告 batching 收益与 Deferred 收益。
不以“GPU利用率更高”替代端到端加速证据。

## 15. WP7 — Stability / lifecycle / retained regression

主 batch1：6 prompts×64 tokens×3 cycles×2 campaigns=2304tokens。
第二 campaign 独立启动，固定请求次序；ON自身重复确定性，不要求匹配OFF轨迹。
每个拟发布 batch 单独完成同类稳定性及混合请求长度/完成顺序测试。

监控 RSS/HBM 起点、峰值、平台和结束，区分缓存增长与无界增长；记录全部异常和超时。
必须 no crash/deadlock/NaN/Inf、无跨RID污染、无未消费pending、无持续无界内存增长。
最终重跑受影响的R2A/R2B/R2C、Round3与原非Deferred compact回归。
代码若未变，可引用同一已固定代码的本轮结果，不为“最终commit”重复全部昂贵矩阵。

## 16. 错误处置与安全边界

系统bug：保留首次失败，官方源码核对→最小fix→定向回归→仅重跑受影响证据。
新证据使用新目录和源SHA，旧失败不覆盖。
质量/性能未达标：如实分类，不自动换低频expert、降低defer budget或换benchmark。
实验调参若确有价值，作为新profile预注册并重新评估，不回填正式profile。

A3仅NPU0、本项目容器；不安装宿主库，不动业务容器/其他人进程。
显存不足、OOM、超时、设备错误只停止精确确认归本次启动的进程组。
正常运行不重启；监控不重复提交。检查日志不得输出密码、token。
本轮不涉及TP2/HCCL、Graph、Dynamic Placement、MTP、Speculative、KML、W8A8/MXFP4或Qwen迁移。

## 17. 时间预算和减负措施

不再重复旧QD3三组长采集；只重跑改动相关controls。旧结果的离线统计不占NPU。
参考近期2×64 production+oracle含两次启动约4.4分钟，不能据此线性保证长质量集耗时。
开工先用16个独立工程请求测每配置TTFT、实际输出长度、tok/s并冻结预算。

估时公式：T≈启动时间+Σ(prefill时间+输出token数/decode速度)+离线评估。
仅GSM8K若1319题、A/B两模式、平均256token，约675328输出token；
假设10tok/s约18.8小时，3tok/s约62.5小时，均未计prefill和中文任务。
这是条件估算而非已测速度。不能承诺大型质量验证“几小时一定结束”。

执行优先级：工程smoke→早期速度pilot→质量smoke→主质量→正式性能/稳定性。
报告每阶段预算、已完成数、ETA和实际batch。自动接续只在执行批准后启用，
无变化静默，失败/完成更新；本次仅写计划，现有heartbeat继续PAUSED。

## 18. 新契约结构（执行时填实值，不能把占位当冻结）

```json
{
  "version": "round6.4-official-deferred-approximation-v1",
  "status": "DRAFT",
  "semantics": "official_cross_layer_deferred_approximation",
  "cross_mode_token_exact_required": false,
  "arbitrary_candidate_order_required": false,
  "logit_metrics_role": "diagnostic",
  "same_path_determinism_required": true,
  "routing_ownership_lifetime_required": true,
  "all_finite_required": true,
  "quality_margin_absolute_proposed": 0.02,
  "quality_margin_approved": false,
  "quality_method": "paired_noninferiority_two_primary_benchmarks",
  "speedup_claim_requires_paired_measurement": true,
  "primary_profile": "D1_batch1",
  "batch2_4": "separately_qualified",
  "default_deferred_enabled": false
}
```

正式冻结还需实际parent/child、环境、语料/模板/解析器/样本列表/顺序SHA、统计实现SHA。
不再生成沿用Round6.3版本名的candidate，不沿用其derive状态作为入口。

## 19. 最终状态与交付范围

分开标记，禁止一个泛化PASS掩盖缺失项：

| 状态 | 条件 |
|---|---|
| `DEFERRED_IMPLEMENTATION_READY` | oracle、system、determinism、lifecycle通过 |
| `DEFERRED_QUALITY_QUALIFIED` | 独立质量在批准margin内 |
| `DEFERRED_SPEEDUP_VERIFIED` | 相同batch/资源下真实成对加速证据 |
| `DEFERRED_APPROX_ACCELERATION_READY` | 上述三者均满足，且列明profile/工作负载 |
| `IMPLEMENTED_NO_SPEEDUP` | 系统和质量可接受，但没测到收益 |
| `QUALITY_INCONCLUSIVE / QUALITY_REGRESSION` | 证据不足 / 质量不可接受，分别记录 |

默认开关仍OFF；通过后以明确opt-in启动命令发布，不无提示改变原封存用户行为。
只对实际通过的模型/精度/placement/batch声明资格，不外推所有模型、D2/D3或高并发。

## 20. 文档、提交与冻结

```text
migration_bridge/round6_4/
  ROUND6_4_OFFICIAL_DEFERRED_APPROXIMATE_ACCELERATION_PLAN.md
  00_BASELINE_AND_SCOPE.md
  01_APPROXIMATION_ADR.md
  02_UPSTREAM_AND_LOCAL_DIFF.md
  03_SYSTEM_RUNTIME_AND_ORACLE.md
  04_BATCH2_4_QUALIFICATION.md
  05_NUMERICAL_AND_FREE_GENERATION_DIAGNOSTICS.md
  06_QUALITY_PROTOCOL_AND_RESULTS.md
  07_PERFORMANCE_AB.md
  08_STABILITY_AND_REGRESSION.md
  09_FINAL_DECISION.md
  10_GIT_AND_ARTIFACT_PROVENANCE.md
  APPROXIMATE_ACCELERATION_CONTRACT.json
  tools/  corpora/  evidence/
  ROUND6_4_FINAL_SUMMARY.md
```

提交按docs/protocol、test/tool、productionfix、evidence分离。只提交本轮拥有的文件。
child若有改动先发布可获取ref，再提交parent gitlink；远端推送/PR按执行授权处理，
不因为写计划就push或merge main。旧tag不移动。
只有三项资格均完成才考虑新tag `round6-4-dsv2lite-deferred-approx-a3-verified`，
tag message列完整SHAs、质量margin、benchmark分数和实际速度收益；失败不创建READYtag。

## 21. 审阅要点

本方案的核心选择是接受官方近似，而非重做严格cross-mode等价测试。
审阅时主要确认：

1. 默认质量容忍提案“每benchmark最多2个百分点绝对损失”是否符合业务需求；
   该项未批准前不是正式门禁。允许在开测前改成别的明确目标，不按结果修改。
2. 主目标先限定D1 batch1，batch2/4独立验证，不自动扩大层数/专家数。
3. 主质量使用原生GSM8K推理和CMMLU；历史256题只作smoke，不作为唯一质量证据。
4. 没有实际速度收益时接受“实现完成但不宣称加速成功”的真实结论。

**执行原则：不再要求官方近似算法证明它不是近似；要求它证明近似值得采用。**
