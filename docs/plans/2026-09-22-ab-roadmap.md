# 09｜从当前证据走向 A/B 完整主线：路线、验收与回传要求

> 编制日期：2026-09-22。本文讨论项目后续方向，**不是要求下一轮同时完成所有事项**。  
> 当前事实以 [S1]/[S2] 为准；新增里程碑、测试建议与晋级门槛是本次评审提出的设计。

## 1. 不改北极星，只调整最近的一步

项目目标继续是：**Kunpeng Expert Engine + Static/Dynamic Placement + Hybrid/Layerwise Prefill，与 Qwen3.8 的 GDN/QSA/PLE Ascend 语义和必要算子优化结合。**

当前最合理的近程顺序是：

```text
现有真实 Expert fixture / provider 地图 / PARTIAL_ATTRIBUTION
  → 核对 prepare 的来源、阶段与数值协议
  → 一个 Expert shadow 准备成本分解实验
  → 有证据才做一个最小计算/布局/搬运改进
  → 在相同配置下补组件和完整模型回归
  → 逐项补齐 A/B 覆盖矩阵，再开展质量与正式性能资格
```

KV/HiCache 继续后置；它们只有在明确的上下文容量目标下才应成为独立扩展。现在也不应为了“更像算法项目”强行重写 GDN/QSA，更不能把 Static/Dynamic 和 Hybrid 的未展开证据永久隐藏在总体 PASS 之下。[S2][S3 §0、§15]

## 2. 最近一轮如何分配注意力

### 主实验：A 的 Expert shadow preparation

它有明确长边界和可复用组件入口，是 [S2] 唯一选定的后续提案。详细执行范围见文档 08。

### 配套工作：B 的覆盖和参考账本，不增加新设备主分支

利用已存在的 `COMPONENT_AND_KERNEL_MAP.md` 与归因报告，整理 GDN/QSA/PLE/GR 在 Prefill、首个 decode、后续 decode 中分别属于：源码已定位、真实调用已观察、独立 reference 已验证、尚未运行。[S1:L8，L19，L25]

同时核查“OK 只返回换行”的请求契约。它是一个应明确处理的质量边界，但不应在缺少完整请求的情况下升级成“整个模型错误”。如确认合理设置下存在数值/语义回归，则优先正确性，不让性能候选晋级。

**这两项配套工作先读现成证据，不自动追加全模型请求。**

## 3. A 主线的最低验收矩阵

下表中“本轮证据”是摘要所支持的范围；“后续验收”是建议，不是已完成事实。

| ID | 能力 | 本轮证据 | 后续最小验收方向 |
|---|---|---|---|
| A01 | CPU Expert 真执行 | MXFP4 CPU partial 与真实回放 | 锁定 CPUInfer/native worker；区分 queue、packing/解量化、GEMM、合并；输入/权重/线程/NUMA 可追溯 |
| A02 | 混合路由正确性 | 一个 layer0 decode 样本 | 在原资源内补不同路由/空分支的覆盖索引；CPU/NPU 分工与 reference 一致 |
| A03 | Prefill microbatch | 摘要未给专项回放结果 | 一个合法大小的真实/明确派生批量用例，边界不超过冻结 workspace；不得与 decode 混称 |
| A04 | Static identity | 实际 Q32 map 与 replay 对齐 | 策略名称、预算、统计来源、每层 mask/hash 和 deterministic 规则；不是只数 resident 数量 |
| A05 | Static 价值 | 无公平性能对照 | 在同一质量/预算条件下比较冻结配置，确认成本变化；不默认全局 Frequency 最优 |
| A06 | Dynamic 正确性 | 历史机制；本轮未提供专项事务运行细节 | 统计→迁移→publication→decode 的一条完整证据；mask/map/source/generation 一致；failure/cancel 收口 |
| A07 | Dynamic 收益 | 未完成 | 分开迁移成本、后续节省和质量变化；使用相同请求条件，不能直接借用官方性能 |
| A08 | Hybrid Prefill | 不能由混合 decode 代替 | 锁定触发条件、CPU/NPU 分支和 token coverage；有同语义 reference |
| A09 | Layerwise Prefill | prepare 次数和累计成本 | shadow source、materialization、布局、copy、retire 分解；同一合法 lifespan 内的复用关系 |
| A10 | Hybrid/Layerwise 选择 | 摘要未给策略比较 | 后续限定两种已支持模式的比较，固定 workload/精度；不要同时 sweep mask、bank 与 batch |

来源依据：[S1:L11–L25，L35][S2][S3 §5.4、§6.2][S4 §2–5]。

“本轮没给证据”不等于“功能不存在”。已有旧测试应标明来源、配置和是否与当前基线兼容；不兼容或缺失才补实验。先做覆盖表，可以避免多跑昂贵完整模型。

### 3.1 Static/Dynamic 后续尤其要防止精度混淆

当前一侧是 MXFP4 CPU，一侧是 BF16 NPU。改变放置可能改变某个 Expert 实际使用的数值路径。不能一边切换 map/精度一边用“输出不 exact”证明调度错误，也不能只因更快就忽略质量变化。[S1:L11，L23][S5 §4.6]

因此需要分开：固定有效算术时的系统执行对照；放置变化导致数值路径变化时的混合精度参考与质量检查。组件性能比较不是完整模型质量资格。

### 3.2 Dynamic 的价值最终要算净收益

在可比条件下，后续 decode 累计节省只有超过迁移及同步开销，才有该请求上的净性能收益。这是成本比较原则，不是当前测量结果；现有材料没有足够数据给出盈亏平衡 token 数。

在没有该证据前，应写“Dynamic 功能/事务已验证到某范围”，不要写“Dynamic 提升速度”。

## 4. B 主线的最低验收矩阵

| ID | 路径 | 下一步应从已有材料恢复的合同 | 之后可能需要的受限验证 |
|---|---|---|---|
| B01 | 模型身份 | 精确 config/tokenizer/checkpoint revision、层类型、实际类/入口 | receipt 适用范围与可信链；不能拿模型简称当唯一身份 |
| B02 | GDN Prefill | 实际 provider、chunk/state 输入输出、历史所有权 | 同语义 reference；分块前后连续性，受支持边界 |
| B03 | GDN Decode | 实际递归更新路径、state epoch 与复用 | 合法连续步状态与参考；cancel/reuse 不串状态 |
| B04 | QSA/indexer | index、选择、KV 与 attention 的实际调用和短输入分支 | 已支持的边界输入；未触发 sparse 分支要标 NOT_EXERCISED |
| B05 | PLE | row identity、gather/H2D/consumer、history 和 slot | 既有 exact row/history 证据可追溯；只有暴露等待足够才立项调度变化 |
| B06 | GR/Router/Shared Expert | 不把这些都漏为未说明的 Other | 分支和基础数值/reference，按热点需要补计时 |
| B07 | Ascend correctness | 模块 reference、两组 OFF/ON、真实 fixture 分开保存 | 同路径输出/状态、已知答案 canary；不把回归一致当质量 |
| B08 | 成本归因 | parent/child、采样层、时钟、request/phase 范围 | 对确定热点提高覆盖；保留残余，没必要先做一张虚假的 100% 图 |
| B09 | kernel 必要性 | 当前 eager/provider 身份与已知缺口 | 精确 shape/dtype/layout、fallback 原因、独立 reference、可消除成本与资源预算 |

来源依据：[S1:L8–L27][S2:L11–L14][S3 §5.5、§9.2]。

**eager provider 不是“没有调用任何设备 kernel”的同义词，也不是“必须重写”的充分证据。** 当前来源仅说已识别实际 eager providers；未提供具体算子时序和完整成本，不能代替热点立项。

QSA 的工作区和历史正确性属于 B 主线；研究通用跨请求 prefix tree、持久 KV 存储则是另一支线。前者不能因“KV 后置”被忽略，后者不能以“模型需要历史”获得自动扩权。

## 5. 何时允许进入真正的改代码优化

建议用以下四道门，而不是规定“下一轮必须产出一个新 kernel”。

### 门一：对象明确

必须有真实函数/算子、调用阶段、输入输出和 source 身份，不是“CPU 慢”“QSA 新”或“搬了很多 GB”。

### 门二：正确性可独立检查

已有固定的 reference、fixture、容差、边界与回退。变化对象不改数值时要求相应身份/内容 exact；浮点路径有差异时沿用既定规则，不事后放宽。

### 门三：节省的是可影响请求的成本

不必要求全模型每个周期全部解释，但必须证明所改成本真实、可消除且不只是被完全隐藏的工作。组件速度、累计 bytes 与请求节省分别记录。

### 门四：改动和资源有界

一次只改变一个主要因素；新增 buffer、生命周期、同步边界或配置均需单独说明。行为候选首先默认关闭，有清晰回退；晋级前补未 profile 的对照与必要完整模型回归。[S3 §9–10]

若准备分解后最应改的是一个重复 materialization 或布局转换，它同样属于 A/B 的必要优化，不比“写一个 GEMM kernel”低级。技术价值来自解决真实限制，不来自代码层级的名字。

## 6. 三个后续阶段，而非一个无限大的 sprint

| 阶段 | 主要结果 | 可接受的收口 | 不应塞入 |
|---|---|---|---|
| M1：准备成本分解 | 文档 08 的同源、同分支、受限组件结果 | 可解释的子成本/残余 + 一个候选或具体 NO_CHANGE | 新缓存/新 kernel/完整模型 sweep |
| M2：单一候选实现 | 一个 materialization、布局、复制或明确 kernel 改动 | 数值正确、资源不变/有批准、组件改善；完整模型证据另按预算补 | 同时调 Static/Dynamic、bank、PLE 与 batch |
| M3：A/B 窄配置资格 | 覆盖矩阵 + 已知答案/质量 + 可比未 profile 性能 | 对已限定模型/精度/模式/输入范围的正式结论 | 自动扩大到 128K、HiCache、多卡、多模态 |

M2 的具体对象由 M1 决定，不在这份评审里预先宣布。M3 也不要求把所有历史后端做笛卡尔积；先固定一个可解释主配置，明确其他配置的继承证据或未资格状态。

## 7. 质量 canary 的建议处理顺序

先恢复原短请求完整参数和预期：它究竟是只为获得一条 token 的路径探针，还是要求模型完整回答 OK？如果生成上限、模板或 stop 条件使期待不成立，就应修正测试契约并保留旧结果的原名，而不是偷偷把预期改成换行。[S1:L27]

下一次有独立完整模型预算时，再使用合理输出预算和预先确定的答案规则做最小语义 canary，并与冻结默认路径对照。一个 canary 仍不足以完成 CMMLU，但能够避免长期只验证“两个版本一样输出错误”。

若出现可重复的同输入、同参考语义分歧，先暂停优化候选晋级，沿首次分歧模块定位。没有这种证据前，不从一条换行直接指认 GDN/QSA/PLE 或低 bit bank 为根因。

## 8. 建议的回传包：少传材料，但传对材料

下次本地执行者应在现成证据基础上输出以下摘要。并非要求上传模型、私有 prompt 或全部原始激活。

| 内容 | 至少回答 |
|---|---|
| A/B 覆盖表 | 每项是本轮实际执行、历史继承、只读源码、未运行，还是证据未提供 |
| provider/source 合同 | 准备边界的真实文件/符号、model/layer/bank/layout/map，Prefill↔replay 关系 |
| 准备分解结果 | 父项、子项、未归因、时钟、bytes 口径、alias/分配和 lifetime |
| 数值与工作负载身份 | A/B fixture 各自来源、精度/reference、误差/容差来源、输出差异 |
| 观察 OFF/ON 差异 | 调用次数、结果、异常、资源；是否有额外同步/持久分配 |
| 唯一候选卡 | 改什么、为什么、预期省哪段成本、哪些证据仍缺、不改什么 |
| 质量/未知清单 | OK canary 契约、未解释首次 decode parent、GDN/QSA 未覆盖分支 |
| 状态与清理 | 新预算实际消耗、失败/停止、训练活动、owned 回收与身份哈希 |

给每个结果附精确 run ID、来源文件及 hash。保护源码/观测/replay 不同 snapshot 的身份；“模型全量 receipt + 当前元数据”继续按原方法标注，不能改写成“本轮整个 checkpoint SHA 全量复验”。[S1:L7，L57]

## 9. 供本次评审使用的状态描述

以下不是执行方状态文件的新值，也不更名其现有字段：

```text
方向判断：符合 A/B 主线
本轮主要新增：真实 Expert replay + provider 观测 + Expert prepare 线索
A 覆盖：局部新增验证；Static/Dynamic/Hybrid 需逐项查现成证据
B 覆盖：局部后端/运行证据；独立语义与全模型成本未齐
执行方当前归因：PARTIAL_ATTRIBUTION
执行方当前决定：NO_BEHAVIOR_CHANGE_SELECTED
T5：NOT_ENTERED
下一提案：EXPERT_SHADOW_PREPARATION_COST_DECOMPOSITION（未启动）
产品质量 / 正式性能 / 长上下文 / 部署资格：继续 PENDING
```

**最终建议：沿 A 的 Expert 准备链路向下定位一个真实成本，同时用明确覆盖表守住 B 的正确性与后续算子门槛。不要回到 KV 扩展，也不要为了证明“在做核心”而无依据重写算子。**

## 来源索引

- **[S1]** [本轮主线交付摘要](sources/ROUND9_MAINLINE_DELIVERY_ZH.md)。本轮事实的主要来源；其链接的远端原始证据未包含在本次上传中。
- **[S2]** [本轮窄实验决策](sources/NARROW_EXPERIMENT_DECISION.md)。保留 `NOT_ENTERED`、`NO_BEHAVIOR_CHANGE_SELECTED` 和下一轮提案的原始含义。
- **[S3]** [上一轮实施方案](sources/05_Round9下一轮方案_KT主线与Qwen38异构推理.md)。用于对照任务和约束，不当作执行结果。
- **[S4]** [官方 KT/SGLang 对照与能力边界](sources/02_官方KT_SGLang对照与能力边界.md)。作为历史材料，不代表本次重新核验官方最新版。
- **[S5]** [功能补齐的设计重点](sources/04_功能补齐的设计重点-1.md)。用于保留 identity、生命周期、计时与资源约束。

`[S1:L7–L12]` 等表示来源文件中的实际行号；`[S3 §6.2]` 表示来源章节。本文的“建议”均是待执行设计，不是新增测试结果或设备运行授权。

## 2026-09-22 并行执行范围修订

用户授权与08并行执行。此次只进行已有本地证据的A01–A10/B01–B09覆盖审计与canary合同恢复，不追加HTTP/模型/组件启动。绑定revision、bank、容量、placement、prefill、PLE和snapshot身份；历史证据不自动等于当前配置资格。M2行为优化未进入；M3在独立预算下只要求选定主配置必要依赖闭合，不等待全矩阵全部完成。
