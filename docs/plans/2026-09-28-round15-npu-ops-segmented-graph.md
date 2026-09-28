# Round15｜成熟 NPU 算子接入与 KT 分段图执行

> **目标**：保留现有 KT 异构架构，分别减少 NPU 计算/中间数据成本和 Host 下发成本，取得真实请求加速。A 线接入成熟算子；B 线研究 NPU 分段图，不把 CPU Expert 塞进整图。本文是给 Codex 的执行任务书，不代表已启动远端任务。

## 0. 起点、范围与不变项

**S0 从 Round14a 已推送提交 `e3a443ee7bb3e075b6ca3bfe19709b2a183c95fb` 恢复**，同时核验实际运行快照、依赖与未提交差异；Git HEAD 不替代完整运行身份。[L1]

固定：Qwen3.8 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`；C128（NUMA 2/3/4/7）；MXFP4 CPU bank；Uniform Q32 BF16/NZ29 resident；已晋级 GDN、QSA Vectorized Prefill＋DIRECT_BF16_ROW Decode；PLE ASYNC；chunk32；H129；TP1、并发1；输入/输出各不超过128 tokens。W2的有效历史可接近256，不能把输入上限误当KV历史容量。

**不同时迁移 W8A8、MTP、front-loading、KT多卡、并发或长上下文；不重开 NZ update/R2/R3。** 保留既有 fallback，不扩大默认资格。旧英文失败与待审项继续登记；本轮不做全面质量清理，但任何新增数值/状态回归都必须阻止晋级。

新 vLLM-Ascend 报告的 BF16 TP8、MTP OFF 约19.76 output tok/s，是不同工作负载、资源和后端的参考，不是本轮单卡速度承诺，也不能证明 Graph 是旧部署慢的唯一根因。[L2]

## 1. 先完成一次公共取证，不另起大规模基准项目

- [ ] 核对 S0 与新部署实际镜像 digest、torch/torch_npu/CANN、已加载自定义库、QSA/GDN/MoE provider、图配置及运行分支。优先读取已有代码和 trace，不重新启动 TP8。
- [ ] 形成 `PROVIDER_MAP.md`：**本地调用点 → 可借鉴实现 → dtype/layout/state 合同 → 依赖与许可 → BF16/TP1可用性 → fallback**。不得只抄启动开关或函数名。
- [ ] 两个 fresh S0 短 profile 块均测 W1/W2，记录设备计算、Host 下发空隙、CPU结果等待、调用次数与中间copy；不加逐算子同步，不叠加父子span。
- [ ] 在设备实验前跑完 import、op注册、模型构造和全部 source/binding gate 的离线预检。候选登记成独立实现身份，旧实现保留为参考，禁止删除hash校验。

公开文档区分编译融合与 ACLGraph 捕获/重放；具体支持取决于本地版本。新框架中有效的图模式或后端，不能直接当作当前 KT 已具备的能力。[O1–O3]

## 2. A 线｜成熟 NPU 算子接入，先保持图关闭

**目标：用可复用、数学兼容的 NPU 实现替换一个真实热点，减少完整调用成本。**

### A1. 按当前证据选择一个算子族

| 候选 | 优先调查内容 | 必须保留的语义 |
|---|---|---|
| QSA | 融合 indexer、稀疏展开、attention，减少小算子和临时读写 | per-query causal、signed ranking、Top-K、GQA、KV/index身份 |
| GDN及周边 | 已有core外的gate/norm/projection融合，或更合适的相同core | **本地sigmoid，不是swish**；conv/recurrent state、norm权重 |
| NPU resident MoE/主干 | Q32 grouped计算、残差/norm/投影融合 | resident子集、空分支、router权重、CPU/NPU合并 |

优先复用现有 SGLang NPU、torch_npu/CANN，或抽取新 vLLM-Ascend 的底层原语；必要时允许最小 Ascend C/Triton-Ascend adapter/kernel。**最多一个算子族、两种实现、一个请求候选。** 依赖不可分离、只支持INT8或TP8时，明确记录不适用，继续检查另一个候选，不强行移植整套vLLM。

### A2. 接入与验收

1. 默认OFF，旧provider独立保留；只改选中的算子族，Graph保持OFF，CPU bank/路由/placement不变。
2. 用真实权重与多层fixture，覆盖Prefill/Decode、尾块、causal/GQA、state reset、请求切换、取消/复用及不支持shape的原路径fallback。
3. reference与候选均按生产 `inference_mode` 运行；原有FP64/FP32 oracle保留。使用预先登记的模块误差门，不统一放宽成一个大容差；不能用候选输出作参考。
4. 至少20轮复用无逐轮增长，检查allocated/reserved及真实持有者。完整parent计时包含layout转换、metadata、临时buffer、copy、同步与归还，不只计kernel。
5. 两个独立组件配对均改善至少15%才进入请求。之后两进程smoke，再做 `S0 → SA → SA → S0` 四进程ABBA。

**A请求主门在实现前二选一**：
- Decode型：两配对块 W2 TPOT均降低≥10%，串行output吞吐均提高≥10%。
- Prefill型：两配对块 W1 TTFT均降低≥15%，W1 E2E均降低≥10%。

未达到门槛但完成实测为 `NO_PROMOTION`；兼容性或数值未过而没有性能数据为 `BLOCKED/REJECTED`，不能写“无收益”。

## 3. B 线｜KT 的 NPU 分段图执行

**目标：减少Decode下发与算子间空隙；不是捕获整个Python模型，也不是跨层延迟CPU结果。** 首版只支持Decode batch1；Prefill保持既有路径。

### B1. 按真实依赖选边界

以下是依赖示意，不要求机械建三张图：

```text
NPU前段：当前token的模型计算/路由
        ↓ 输入与路由D2H完成后才允许CPU读取
Host：提交CPUInfer ─────→ Kunpeng执行
        │                    │
NPU：独立resident等工作       │
        │                    ↓ CPU结果完成、H2D完成
        └──────────→ NPU合并/后段 → 后续依赖层
```

CPUInfer、PLE的Host gather、动态Python路由控制、HTTP/scheduler与取消判定留在图外。现有确有独立性的NPU工作才允许重叠；不提前执行依赖CPU结果的后续层，不引入deferred。

### B2. 最小实现要求

- [ ] 先在一个真实层捕获一个稳定NPU区域，再扩大覆盖；优先使用当前可用的 `torch_npu`/ACLGraph 原语，接口由本地版本核验。[O2–O3]
- [ ] 固定图输入/输出buffer及state存储地址；每次重放更新真实token、位置、route/index、有效长度和request绑定。**不能把capture时的路由或KV长度冻成常量。**
- [ ] 动态数量使用有界容量与正确mask，或图外选择已验证bucket；选中K或形状变化不能改变稀疏语义。未支持分支在执行前退回eager。
- [ ] 图缓存键包含layer/segment、实现与权重身份、dtype/layout、batch/capacity和存储绑定版本；request epoch是每次更新的数据，不能导致每请求新增一套图。初始每层最多两段，整模型最多96个活跃图实例，多bucket不能突破总帽。
- [ ] warmup/capture使用独立scratch状态，捕获后恢复确定初态，不让预热推进生产GDN/KV/PLE历史。记录capture/replay次数、覆盖率、fallback、图池与stream资源。
- [ ] 输入ready、CPU结果ready、H2D完成和NPU消费依赖都保留。取消后停止新增replay，排空在途工作再退休buffer，禁止late commit及旧结果写入新请求。
- [ ] **replay中途失败不能直接对同一token再跑eager**：可能已修改state；须有已验证的回滚，或终止请求、drain并重载。只有执行前不支持的shape允许无副作用fallback。
- [ ] 先测纯capture/replay；需要编译融合时独立标记为B的第二实现，不把融合收益全归给图重放。不同时改CPU调度策略。

### B3. B 不以 A 成功为前置

A/B只读分析和本地测试可并行，设备串行。A已有合格SA时，B对照 **SA eager vs SA＋graph**；否则使用 **S0 eager vs S0＋graph**。同一次B对照的算子身份必须完全一致。

20轮跨请求、连续Decode、reset、取消/复用、图销毁验证通过；两块**完整Hybrid层**成本均降低≥10%后，进行两进程smoke和四进程 `E → G → G → E` ABBA。两块W2 TPOT均降低≥10%、串行吞吐均提高≥10%才晋级。只看到replay成功、Host提交变快或单图更快，不算请求优化完成。

图开销/填充成本抵消收益则保留eager；输入合同无法安全表达则记录具体图断点，不无限碎片化捕获。

## 4. 统一请求测试与最终组合

| Workload | 实际输入/输出token | 用途 |
|---|---|---|
| W0 | 64/2 | 极短请求保护 |
| W1 | 128/16 | Prefill、TTFT |
| W2 | 128/128 | **Decode、主吞吐** |
| W4 | 64/32 | 中短生成保护 |

恢复冻结prompt IDs、模板、采样和stop规则；性能请求必须达到实际输出数，不能用提前EOS获得提速。每个ABBA进程每个workload **1次warmup＋3次measurement，共16次generation**；计时阶段profiler OFF，另存首次编译/capture时间。

每个候选必须通过冻结canary、有限logprob、多步输出参考、state连续性及cancel/reuse。需要容许浮点重排的范围必须在测量前登记，不能看结果后修改判分。每个配对块的非主指标TTFT/TPOT/E2E不允许退化>5%；噪声疑问只能用修复/确认储备重新形成完整配对，不能删除慢样本。

报告绝对E2E、TTFT、TPOT、完整连续窗口output tok/s和HBM/Host峰值；输出吞吐按实际输出数/窗口计算，不使用 `1/TPOT` 替代。收益不跨轮相乘，八卡19.76 tok/s不作为单卡资格线。

A、B均晋级时，再做 **S0 → SAB → SAB → S0**；组合失败，回退已有独立资格的单线实现。只有一线晋级且最终身份完全相同，可复用它的完整请求资格，不机械重跑同代码标签。**首要目标：最终两块W2吞吐提高≥10%，或在预先选择的Prefill分支上达到A的请求主门；不预先承诺固定tok/s。**

## 5. 有界自动推进、资源与预算

Codex接手并启动本目标后按 `P0 → A/B → 最终组合 → 冻结` 推进。测试先行；局部bug自行诊断修复；候选负结果回退后继续另一线，不逐步等待批准。当前文档生成不恢复历史timer，也不启动设备。

| 阶段 | 模型进程上限 | generation上限 | abort control上限 |
|---|---:|---:|---:|
| P0取证/profile | 2 | 8 | 0 |
| A：smoke＋ABBA | 6 | 80 | 2 |
| B：smoke＋ABBA | 6 | 80 | 2 |
| 最终组合（需要时） | 4 | 64 | 2 |
| 修复/确认储备 | 2 | 16 | 2 |
| **合计** | **20** | **248** | **8** |

组件进程总帽20（A8、B8、储备4）。单组件≤1,800秒；单模型含编译捕获≤21,600秒；单请求≤1,800秒；总NPU device-hours≤64。失败启动和有修改修复验证计数，预热/取消/reuse请求计入generation；无修改盲重试为0。同根因最多两次有修改设备修复，预算储备优先修复阻塞smoke的问题，保留完整ABBA额度；不够则记录缺口。

只使用重新准入的物理12，C128拓扑不变；两轮60秒CPU/NUMA采样、每节点≥48 GiB启动门和运行headroom继续执行。HBM、Host及外部训练身份使用Round14a最新冻结合同，不继承旧的更宽上限。新增算子workspace＋graph池＋桥接buffer**合计≤4 GiB且必须满足全部更严格总帽**；超额先减图覆盖或fallback，不提高整卡利用率开关硬顶。

权限不足的counter记UNKNOWN，不改宿主权限；不更新共享驱动、不干扰其他服务或运行并行加载/编译。资源不够时只读等待，最多6次fresh失败周期；用完预算或遇到未恢复的基础正确性问题安全收口。每块退出→guard ACK→归档/hash→无FD/mmap引用审计→登记文件定向DONTNEED→fincore，禁止全局drop_caches。

本轮允许本地隔离分支和可回退提交；不自动push、PR、merge、tag、reset或clean，不删除原dirty工作和失败证据。

## 6. 交付与终态

交付：`PROVIDER_MAP.md`、`GRAPH_CONTRACT.md`、A/B独立patch及manifest、组件/请求比较JSON与CSV、图replay/fallback/内存记录、最终配置、预算、资源退休和hash清单。已有目录结构优先复用，P0登记真实文件与符号后再改，不凭计划猜代码路径。

- **`ROUND15_REQUEST_GAIN_QUALIFIED`**：至少一线真实请求晋级，A/B都有明确去留，最终配置通过。
- **`ROUND15_COMPLETE_NO_PROMOTION`**：候选完成所需实验，或有充分的“不适用/无安全图边界”证据，保留S0；性能目标未达成。
- **`ROUND15_PARTIAL_WITH_GAPS`**：必需集成、数值、资源或请求配对没完成；独立成功子结果保留。

所有新阶段初始化为 `NOT_RUN`，不预填成功。**实现未接通不等于没有收益；A无候选不阻塞B；任何失败都不能靠放宽精度、移除依赖或跳过取消来换取速度。**

## 7. 依据与适用边界

- **[L1]** 用户《无标题文档 (2).pdf》§2–6及Round14a结果：当前混合精度、已晋级provider与窄资格。
- **[L2]** 用户《无标题文档 (3).pdf》§2–6：新TP8部署、1024/256负载、Decode图配置及性能；未独立证明各优化贡献。
- **[O1]** 官方Qwen3.8-Flash-Next实验教程：`https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-Flash-Next.html`
- **[O2]** 官方Graph说明，区分编译优化与运行时捕获：`https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html`
- **[O3]** torch_npu图接口实现：`https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/npu/graphs.py`

公开页面于2026-09-28核对；latest/main仅作技术入口，Codex必须固定实际版本和库hash。本轮所有预算和晋级门是新的工程设计，不是附件已经取得的结果。
