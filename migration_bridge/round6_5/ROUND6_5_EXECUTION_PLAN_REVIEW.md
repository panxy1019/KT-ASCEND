# Round 6.5 — Official-Style Static Heavy-Offload Alignment

日期：2026-09-16。版本：review-v1。状态：**AWAITING_USER_REVIEW，尚未执行**。

本文件是执行前方案，不是实验结果。此次仅核查本地历史证据、官方固定快照并编写计划；未启动服务器模型、扩容容器、生成新权重、修改生产代码或开启 Round7。

## 1. 决策摘要

建议批准 Round6.5，然后再决定是否进入 Dynamic Placement。

主线为：Static heavy offload → Static Frequency → 冻结 Round7 baseline。
Deferred 仅对一个冻结 heavy profile 做一次有上限的 relevance probe，不恢复长期性能研发。

当前32个物理核足以作为第一阶段资格验证的起点，**尚无证据证明足以达到 heavy-offload 的最佳性能，也没有证据支持立即买机器或扩到128/256核**。先做受控16/32 worker对照；仅在拓扑、资源和伸缩收益支持时，额外试48或64核。CPU数量不等于可用内存带宽。

最终最低成功要求：至少 R32 在全部 MoE 层稳定运行，满足静态语义和冻结数值检查，真实节省 routed-weight HBM，并形成 host/route/performance 证据。R8不要求快，R4/R0不进入默认计划。

## 2. 已知事实与仍需验证的事项

### 已归档事实

- Round6.4 final：`c74931235c60d2e899e3acd9fc9277b06962920a`。
- Round6.4 evidence：`c14053c5090a5d022cd8580ef7bbd0b08f557f43`。
- Round6.4：IMPLEMENTED_NO_SPEEDUP，默认 Deferred OFF；不得修改历史结论。
- 历史迁移记录：110.129.0.22，Kunpeng920，256核/4socket，约1TiB RAM；项目cpuset0–31，均在NUMA0。**这不是本轮实时资源余量测量**。
- 910B上16/32worker compact bridge通过，但32worker全层heavy profile尚未验证。
- 新CPU扩展SHA：`0f3aac9009931b18853c9b8fd6f6bba8b02d6ea396e79b2029bf9ce092049895`。
- 旧质量与正式性能来自旧平台，不能替代目标910B上的新实验。

依据：`../round6_4/21_SERVER_MIGRATION.md`、`../round6_4/ROUND6_4_FINAL_SUMMARY.md`。

### 本轮开跑前必须回答

1. 新服务器当时真正可用RAM、NUMA-local RAM、cpuset、CPU quota、HBM、其他任务争用是多少？
2. 当前GGUF是否只有D1/P3相关层？是否覆盖所有要移到CPU的expert？不能由文件大小推断覆盖。
3. 当前loader是否真正只在NPU物理分配resident experts，而不是全量NPU权重加逻辑mask？
4. 32 worker之外的核是否与NUMA0同节点？NPU0与CPU/内存的局部性如何？
5. 当前静态路径能否在所有MoE层保持BF16边界、FP32合并和framework-managed copy_？

## 3. 官方参考及不可照搬项

固定参考SHA：`0dce4c9b097710951d1b3e232f50b973ff0a5505`，只读参考，不merge/rebase。

- [官方专家调度教程](https://github.com/kvcache-ai/ktransformers/blob/0dce4c9b097710951d1b3e232f50b973ff0a5505/doc/en/kt-kernel/experts-sched-Tutorial.md)：提供uniform/frequency/front-loading/random与dynamic update；因此Static Frequency应作为Dynamic的比较基线。官方uniform首先表示跨层预算分布，不自动等于本项目指定的层内均匀ID算法。
- [官方Ascend教程](https://github.com/kvcache-ai/ktransformers/blob/0dce4c9b097710951d1b3e232f50b973ff0a5505/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md)：32/256 NPU resident，其余CPU；使用W8A8/MXFP4，与本项目BF16/F32不同。其实测配置包含单NUMA、32 CPUInfer worker，且指出decode受host带宽影响。这支持先测32worker，不证明本项目32worker最优。官方Graph及其他开关不带入本轮。

用户附录中的 `Official_KTransformers_Offload_Philosophy_vs_Current_KT_ASCEND.md` 未在本次可见附件中取得正文/原始文件；其给定SHA仅登记为USER_SUPPLIED_UNVERIFIED，不能标为已逐字核验。执行前若取得原件再核对；本方案依据用户粘贴任务书和上述已读取官方文档即可成立。

## 4. 范围和继承合同

固定：DeepSeek-V2-Lite revision `604d5664dddd88a0433dbae533b7fe9472482de0`；TP1、batch1、NPU0、LLAMAFILE F32/BF16-valued weights、NPU BF16/unquantized。

主实验：Deferred OFF、Dynamic OFF、Graph OFF、MTP OFF、Speculative OFF。保留production overlap，不以sequential-control替代正式路径。

不做新量化kernel、NUMA线程池重构、上游大合并、新模型、TP2、并发服务。Round7只输出baseline，不在本轮自动实施migration。

### 静态模式与Deferred必须分开

- 静态placement保持模型语义，不享受跨层Deferred近似豁免。
- 系统mapping/ownership/routing/scaling/shared、buffer/stream/lifetime必须正确；同路径token重复与重启exact。
- same-operands expert relative-L2 ≤0.01，全部finite。
- 静态full-model沿用冻结pairwise合同：B_pair=2.1875、K=32；matched history，stable exact，ambiguity membership，overflow=0。不是所有跨backend token强制bitwise exact。
- 若heavy静态超出旧合同：停止升级并记录STATIC_NUMERICAL_SCALE_BLOCKER，不能以2pp质量筛查代替该门禁，也不重新拟合。
- 仅WP9 Deferred ON=1是近似计算；采用新的heavy-profile质量筛查与性能规则，不套用静态full-logit包络来否定已获准近似语义。

## 5. WP0 — Provenance与可复现工作区

批准后从上述Round6.4 final创建 `codex/round6-5-official-style-offload-alignment`；先保留现有 `third_party/llama.cpp` 用户修改，必要时独立worktree，不reset或混commit。

冻结：Round5真实tag及target、Round6.4 SHA、实际运行parent/child与overlay逐文件SHA、容器image/ID、CANN/ops/torch/torch_npu/compiler、SO SHA、模型/tokenizer文件清单、旧与新GGUF清单、所有命令和env。

本地HEAD不能冒充目标机实际加载源码。若运行用overlay，manifest显式列出，不称clean源码运行。

新专用容器建议 `kt-r65-static-offload`，复用已验证image及只读模型资产，避免重复导入镜像；旧容器和Round6.4原始数据保留。不得改宿主库、drop_caches、关swap、改全局NUMA策略或操作业务容器。

产物：00_PROVENANCE.md、provenance.json、source_manifest.json。

## 6. WP1 — 容量、完整权重和CPU核数预检

### 6.1 资源清单

只读收集lscpu拓扑、NUMA node/距离、NPU拓扑、MemAvailable、每node内存、swap、/proc/vmstat、cgroup memory.current/max/events/stat、cpuset.cpus/mems、cpu.max/stat、磁盘空间与页缓存。

预估loader峰值包括：HF读取映射、F32专家数据、CPU loader额外copy、转换临时数据、线程scratch、pinned buffers、框架与服务进程。mmap虚拟长度不能直接当RSS；共享file cache也不能与RSS重复相加。

建议预注册资源保护：峰值预算不超过专用容器memory.max的80%；始终保留host可用内存≥max(32GiB,10%总RAM)，并尊重共享机器实际可分配份额。仅满足整机空闲不足以证明NUMA-local容量足够。超预算先标RESOURCE_LIMIT，不冒称runtime bug。

OOM、设备错误、内存压力或无进展超时只回收本项目run ID所属进程组。其他任务不停、不杀、不迁移。

### 6.2 全层CPU权重资产是新增前置gate

从固定config与实际模型模块推导全部MoE layer IDs，预期1–26但必须实证，禁止把总层数当MoE层数。

逐layer/expert审计gate/up/down的存在、shape、dtype、logical ID及BF16-valued hash。
若旧GGUF仅覆盖少量层，生成**新的Round6.5全层artifact**或固定分层GGUF集合：

1. 使用同一HF revision和原BF16→F32语义，非重新量化；
2. 新文件、新SHA、新转换manifest，不覆盖旧P3/D1文件；
3. 所有expert权重做逻辑值hash对照，实际kernel按层与形状取确定性样本做数值检验；
4. 优先一份全层backing pool用于所有ratio，避免每个ratio重复转换；需记录loader是否映射/复制未执行expert。

容量公式（以config实证H=2048、I=1408、26MoE层、每层64expert为条件）：

`每expert F32权重 = 3×2048×1408×4 = 33 MiB`。

全pool约53.625GiB；R32/R16/R8的CPU-owned逻辑权重分别约26.813/40.219/46.922GiB。它们**不是总RSS或加载峰值预测**，不含共享专家/其他参数、copy、padding、缓存和scratch。

### 6.3 32核的具体决策

默认主线先32worker、现有单NUMA affinity。先辨明physical cores、worker、容器配额、torch/BLAS辅助线程，避免32worker叠加多个全核线程池。

增加一个不混入ratio sweep的有界CPU sizing实验：在单层R32通过后，以固定captured expert输入与同一NUMA内存位置，对16/32worker做各3次测量；全层R32通过后再用同一短服务负载确认端到端趋势。

只有满足资源可用、16→32仍明显改善且无内存/NUMA异常，才试一个额外档位：同NUMA有48核则48，否则同NUMA64核。是否同NUMA以实际topology为准，不能直接假定CPU0–63同node。

- 扩容实用门槛（建议批准）：新档位端到端decode中位数≥32worker的1.10倍，3次方向一致、TTFT无>10%退化、确定性/数值不回退，才值得作为后续新基线。
- 若没有同NUMA额外核心，不自动跨socket扩；跨NUMA算单独实验，不能既换内存策略又把收益归因核数。
- 若扩容档位被选中，全部正式ratio结果必须统一该档位重测/首次测量；不能拿R32的32worker对R8的64worker画同一受控曲线。
- 本方案不请求采购新机器。需要的首先是局部性和带宽证据，而非更多总核数。

## 7. WP2 — 确定性mask和测量协议冻结

| Profile | 每层NPU/CPU experts | resident ratio |
|---|---:|---:|
| R64 |64/0|100%|
| R32 |32/32|50%|
| R16 |16/48|25%|
| R8 |8/56|12.5%|

Uniform定义：每MoE层相同budget；层内固定resident IDs `{j×(64/R): j=0..R-1}`，R64为全体。这样R8⊂R16⊂R32⊂R64，无结果驱动选ID。它是本地明确命名的uniform_v1，不声称与官方层内ID实现逐字相同。

每层：64位mask、resident数正确、无重复、CPU/NPU互斥且并集全体；物理NPU权重dim0=R。未启用offload的层保持R64。R64同时是纯NPU oracle，不伪造CPU coverage。

冻结JSON canonical serialization、mask SHA、layer IDs、mapping dtype、placeholder及zero-weight规则；unit tests覆盖非法预算/重复ID/错层/错hash。

### 工作负载（批准后在任何结果前冻结）

- NUM：原冻结Q2/H2各前4个prompt×64 matched-history位置，总512位置；仅compact资格，不能称新的独立held-out推导。
- DET：原validation中英文/中文/数值/结构四个预定prompt，各64生成，3repeat，8/16/32prefix；全层候选最终提升到10repeat。
- PERF：固定192输入token/64输出token和1024输入/64输出两档；固定prompt与token ID、greedy、EOS处理、缓存状态、KV容量和warmup；性能模式可固定长度，但不得挪用该EOS设置到quality。
- SCHED_DEV与SCHED_HOLDOUT各16新prompt，英/中/数值/结构各4；输入/输出预算、文本、IDs和SHA在路由分析前冻结。与quality数据去重。HOLDOUT不用于改hotness算法。

## 8. WP3 → WP5 — 有界分级矩阵

全部使用同一目标910B上的新R64 oracle，不能拼接旧A3 logits。

| 阶段 | Profiles | 最小运行/门禁 |
|---|---|---|
| 单层L17 |R32→R16→R8|每档4prompt×3repeat×64；NUM512位置；kernel路由控制和权重身份；32个无CPU路由控制样本|
| 四层1/9/17/26 |R32→R16→R8|上述DET；独立restart各4prompt；NUM512位置；4层coverage与生命周期|
| 全MoE |R64→R32→R16→R8|DET+restart+NUM；两档PERF各2warmup/3timed；全层路由、HBM/host采集|
| 最终选中static baseline |一个ratio|4prompt×10repeat、prefix、clean restart；1000真实expert-forward生命周期；两个独立3prompt×64×3cycle服务campaign|

每级任何system、同路径或静态数值失败停止后续升级。R32失败是最低目标阻塞；R16/R8失败不抹掉已通过R32，但保留失败后停止自动推进，不能把未完成全ratio写成Pareto COMPLETE。

同路径gate在同config内exact。重启差异、非finite、ownership/mapping不正确、无解释capture forward均失败。所有lookahead/startup项显式分类。目标只保存必要摘要与首失败张量，禁止默认再做数万个全logit PT重采集。

### CPU-not-hit在heavy比例的特殊性

R8自然fully-NPU token可能非常少甚至不存在。因此分别报告：

1. 实际服务中所有TopK均NPU-owned的eligible行：最多取32，少于32全取；0则NOT_OBSERVED，不伪造PASS。
2. 独立kernel control：真实hidden输入、明确强制全部TopK为NPU-owned的控制route，验证CPU partial exact zero/merge exact；标CONTROLLED，不冒充自然routing证据。

不得为了凑无CPU命中而改正式mask、删prompt或提高resident ratio。

## 9. WP6 — HBM / Host / Routing / Performance

### HBM必须排除KV预分配混淆

固定相同KV token容量、batch、context与cache策略，不能只固定mem-fraction（释放权重后框架可能把空间全吃成KV）。
同时记录device总占用、进程allocated/reserved/peak、routed-weight实际字节、KVpool字节和workspace。before load / after load / warmup / steady / peak分开。

在上述shape前提下，all-MoE BF16 routed-weight逻辑大小R64约26.813GiB；R32/R16/R8分别13.406/6.703/3.352GiB。attention/shared/KV和workspace不同比例缩减。实测与逻辑预测不符先查全量残留权重、临时副本和allocator，不直接说offload无效。

### Host指标

RSS/PSS、RssAnon/RssFile、cgroup anon/file、major/minor faults、swap增量、内存PSI、numa_maps/numastat与page-cache估算。冷加载与warm steady分开，不在共享宿主drop_caches；无法保证冷cache则标NOT_CONTROLLED。

### Routing定义（prefill/decode分层）

- CPU route events：每个(token,layer,topk slot)的CPU-owned事件；同时报每layer-token和每generated token总量。
- route-weight mass：CPU权重和及其占全部路由权重的比例，保留是否含scaling的明确字段。
- token touches CPU / fully-NPU：既报layer-token比例，又报全模型一个token是否任一层进入CPU。两者不可混用。
- 每层/每expert命中数、finite、dense/shared明确不计入routed denominator。

性能timed段关闭重型capture和逐token磁盘日志；独立诊断pass采集路由/CPU wait，与timed pass做配置和输出一致性检查。
TTFT、内部prefill、decode、端到端latency分开；CPU wait是关键路径host等待，不把重叠的所有layer计时简单累加成latency。

### 带宽结论的证据等级

CPU wait高/NPU低利用率不等于已证明内存带宽饱和，也可能是线程调度、page fault、NUMA远端或同步。
优先现有可用内存控制器counter；无权限则标BANDWIDTH_LIMIT_SUSPECTED，只给逻辑weight bytes/sec代理，不能称实测DRAM GB/s。禁止为采样更改宿主perf权限/安装驱动。

输出三组曲线/表：HBM-vs-throughput、ratio-vs-route pressure、ratio-vs-wait/带宽证据，并标记受其他任务污染的时间段。

## 10. WP7 — Static Frequency（不是Dynamic）

先只用uniform的运行数据选择一个ratio；优先满足稳定/容量/压力的R16或R8，必要时R32。不用quality输出选ratio或expert。

在DEV的R64固定history路由上统计每层expert命中频次，按count降序选R个，平局按global logical ID升序。每层仍固定R，不同时引入跨层预算优化。权重mass可报告但不另试一组规则挑赢家。

先离线把uniform/frequency masks投影到**相同HOLDOUT路由trace**，避免自由生成历史改变导致假收益。然后用真实frequency静态配置验证物理resident布局、运行/数值/资源；offline mask projection不能替代真实加载。

预注册locality gate：CPU route-event相对下降≥10%，或全模型fully-NPU token比例绝对增加≥5pp。两个层级指标都报告，但不临时切换成功指标。原值为0时相对变化记N/A。

不满足则LOCALITY_WEAK；不会自动进入Dynamic迁移。可以提出mechanism-only或新模型路线，但本轮不实施新模型。

## 11. WP8 — 冻结Round7 baseline

round7_baseline.json包含ratio、uniform/frequency mask SHA、DEV/HOLDOUT SHA、CPU workers/affinity/mems、NPU/内存限制、weight/code/contract hashes、静态数值、coverage、route pressure和performance。

是否允许Round7必须单独写：

- 静态system/numerical/capacity全通过，frequency locality有信号：BASELINE_FROZEN，建议Dynamic shadow。
- 带宽限制有直接证据且调度改善不足：建议ARM低比特CPU expert，不自动启动新kernel。
- 只有带宽疑似证据：结论保留不确定性，不过度归因。
- 静态数值或runtime失败：Round7 NOT_STARTED。

未来shadow门槛保留用户建议：相对Static Frequency，CPU事件再降≥5%或fully-NPU +3pp才进入migration。此处仅记入下一轮entry条件。

## 12. WP9 — 一次Bounded Deferred Relevance Probe

仅在一个heavy static frequency profile完全冻结后运行，Dynamic OFF，H0 Deferred OFF，H1 ON=1。映射/专家不变，证明actual Deferred routes>0，报告CPU和Deferred route事件及weight mass。没有实际Deferred工作则NOT_EXERCISED，不宣称speedup。

性能：3个ABBA block，12个独立进程；沿用冻结192input/256output、2warmup+1timed/request；同host/核数/placement，重capture OFF。以每block对数加速比计算geomean与Student-t(df=2) 95%CI。geomean>1且CI下界>1才SPEEDUP_DEMONSTRATED。不满足即NOT_DEMONSTRATED，不追加样本找显著性。

质量：新host/新heavy placement重新跑冻结GSM8K32+CMMLU256，H0/H1共576请求；prompt/evaluator/max token/EOS策略不变，invalid/truncation全部保留且按原规则计分。每benchmark observed delta≥-2pp为relevance screen；paired bootstrap仅诊断，不称population非劣。

质量与性能均通过才REOPEN_CANDIDATE，也不等于production READY；Round7仍Deferred OFF。
无加速则Deferred性能工作SHELVED_UNTIL_NEW_MODEL_OR_BACKEND。质量失败标QUALITY_SCREEN_FAIL，不能仅以性能决定候选。
若预算不足，明确NOT_RUN，不能把未测质量或争用下的计时记作完整negative。

## 13. 自动执行、停止与预算

批准后脚本可自动按前置gate推进；每个run保存状态机、PID归属、命令、hash、起止时间、结果和资源曲线。重复启动先检查已有run，不打断健康实验。

资源争用允许共存但不操作别人。runtime结果与性能有效性分开：如果NPU有其他任务、CPU quota throttle或外部内存压力明显，标PERFORMANCE_CONFOUNDED；正式ABBA不据此下最终速度结论。暂停自己的队列等条件改善，不杀其他任务。

预检时冻结请求超时、心跳和资源阈值；以单层初测校准剩余ETA，再冻结后续deadline，不遇到慢结果就偷偷减token。

时间预算不按旧D1速度承诺：

`预计总时长 = 模型启动次数×实测加载时间 + 各阶段生成token/实测速度 + prefill/采集/转换/审计`。

本轮可能最大的开销是全层权重转换/加载和576条质量请求，而不是mask脚本。分级矩阵失败即停，避免启动无效的后续阶段。32核是否扩容先由sizing小实验决定，不把所有ratio×所有核数×所有policy做笛卡尔积。

## 14. 最终状态及交付

允许结果：

- HEAVY_OFFLOAD_ALIGNMENT=PASS；PARETO_COMPLETE（实际完成全部要求档位时）；ROUND7_BASELINE=FROZEN。
- PASS_WITH_CPU_BANDWIDTH_LIMIT：必须具备直接或充分交叉验证证据，明确NEXT建议，不冒称硬件counter存在。
- BLOCKED_HEAVY_OFFLOAD_RUNTIME / STATIC_NUMERICAL_SCALE_BLOCKER / RESOURCE_LIMIT：保留已完成档位，不抹掉Round5/6.4历史。
- 部分完成写PARTIAL，不把未运行档位填PASS。

文档按用户清单输出00–10及ROUND6_5_FINAL_SUMMARY.md；机器文件包括provenance、ratio_profiles/results、routing_metrics、pareto_summary、frequency_results、round7_baseline、deferred_probe。额外保留weight_inventory、cpu_sizing和measurement_protocol及完整性报告。

Commit按scope/profile工具/分级证据/Pareto/frequency/Deferred/ADR分开。仅实际有production缺陷才独立fix commit并重跑受影响静态回归。没有本轮最终资格不得创建READY tag。

## 15. 本次审阅建议批准项

1. 先做Round6.5静态heavy-offload；不直接开始Dynamic。
2. 32worker作为起点；同NUMA资源允许且sizing有收益时，至多追加48或64一个档位，不做NUMA架构重构。
3. 批准必要的全层F32 GGUF新资产构建，但严格保持原始weight identity、旧artifact不可覆盖。
4. 批准上面的compact分级矩阵与原静态pairwise数值gate，不把Deferred近似合同套给static。
5. 一个最终profile做一次3-block ABBA和GSM32+CMMLU256 relevance screen；不恢复大规模质量campaign。
6. 最终只给出Round7冻结baseline/推荐路线；Round7在线migration另阶段执行。

**审阅通过前：仅保留此计划，不启动任何Round6.5模型测试或CPU扩容。**
