# 08｜下一轮任务书：Expert Shadow 准备成本分解

> **状态：PROPOSED / NOT_STARTED。** 本文没有授权新的设备请求、容器启动或计算行为修改。  
> **执行对象：** 能读取本地 checkout、v62/v63 证据、v20 replay snapshot 与原共享凭据的执行者。  
> **目标：** 解释一个真实来源的 Expert shadow prepare 边界，把成本定位到可验证的子阶段，并给出“进入哪一个优化”或“为何不优化”的明确结论。  
> **架构：** 复用现有真实 Expert 回放、默认关闭的 provider 观测和 32-slot 资源契约；不新建 Serving/Profiler 框架。  
> **技术栈：** 当前冻结的 Kunpeng/Ascend、CPUInfer、MXFP4 bank、BF16 NZ29 路径；精确版本、真实接口和容差从本地 manifest 恢复。  
> **依据：** [S2] 已选定但尚未执行的 `EXPERT_SHADOW_PREPARATION_COST_DECOMPOSITION`，以及 [S3 §6–9]、[S5 §3]。

## 1. 为什么这一轮只做它

已知线索是正常 Prefill 请求的 prepare provider 有较大的累计 Host 观察时长，但它混合了 BF16 来源读取、张量/布局处理和设备复制等待。当前没有证据支持优先改 PLE launch/wait、CPU GEMM 或 GDN/QSA kernel。[S1:L20，L25，L40][S2:L5–L16]

因此，本轮只回答：

> **同一真实 Expert 准备调用，时间分别消耗在权重 materialization、ND/NZ parent 构建、复制提交/完成和旧资源 retire 的什么位置？这些成本对应哪些张量身份和寿命？**

这是一项 A 主线的组件归因实验，而非优化实现 sprint。B 主线在本轮保留来源、状态和数值参考约束，不另开 GDN/QSA/PLE 性能分支。

## 2. 绝对不扩大的范围

不改变 resident mask、Static/Frequency 算法、Dynamic 事务、CPU/NPU 权重分工、bank、模型数学、PLE ring 或 launch/wait。不开启 shadow reuse、changed-slot copying、额外 next-layer staging、跨层权重保留。

不做 KV/HiCache/allocator 新接入、不做 FP8 KV、不做 Graph/MTP/TP2、多请求并发、长上下文或全 NPU 对照。不要把本轮实验称为“已优化”；观测关闭时原调用次数、返回值、异常和资源所有权必须保留。[S2][S3 §3]

**本轮不在完整模型上继续加跑一般性 profile。** 已关闭的 2 次模型启动、3 次设备剖析预算不能沿用；上一轮未用完的 HTTP/组件额度也不自动结转。[S1:L42][S2:L25]

## 3. E0：先恢复证据与真实接口，不启动物理测试

### 3.1 必读本地现成材料

先读取 [S1] 所列的 `KT_QWEN_ATTRIBUTION_REPORT.md`、`COMPONENT_AND_KERNEL_MAP.md`、`PROBLEMS_AND_RESOLUTIONS.md`、`NEXT_ROUND_STATUS.json`，及真实 Expert replay、正常 Prefill trace、资源摘要和对应 manifest。

报告的远端证据入口为 `/mnt/data1/kt-round9/evidence/20260922-kt-mainline-v1`。这是来源提供的路径，不是本次编制者已经访问过的目录。[S1:L46–L57]

分别记录 v61 基线、v63 观测构建、修正实际 map 后的 v20/replay snapshot。后两者不能合并成一个源码状态；核对差异是否仅属于已说明的 replay 分支，而不是猜它们等价。[S1:L35，L57]

### 3.2 必须交付的 provider contract

下面是要查明的字段，不是假定仓库存在的函数签名。先完成真实文件/符号定位，再写 instrumentation patch 清单。

| 字段 | 必须说明 |
|---|---|
| provider 文件、符号、构建 hash | 正常 Prefill 60.2026 s/264.2664 s 对应的是哪个边界 |
| caller 与执行条件 | 哪个 Prefill/Layerwise 分支调用；不是只测 decode resident 分支 |
| canonical BF16 source | 文件/张量/转换来源、模型和 layer 身份；是否与 CPU MXFP4 bank 分离 |
| materialization 输出 | shape、dtype、stride、device、storage identity、是否新分配 |
| ND/NZ parent | 真正的布局参数；是 view/alias、format conversion，还是新 parent storage |
| copy 与完成点 | 提交返回点、existing stream/event、最后完成证据；是否在 provider 返回前完成 |
| retire | 旧 reader、旧 slot、旧 source 的归还条件；是否在 parent 区间内 |
| shadow key | model revision、layer、precision/layout、expert set、source/bank 与 slot generation |
| 数值 oracle | 对应哪种执行精度与分支；exact/容差协议的来源 |
| 资源契约 | 32-slot reservation 的实际对象、Host/NUMA/设备准入和持久 baseline |

**阻塞条件：** 无法找到真实 provider、无法锁定参考/来源、map 身份不一致、资源凭据缺失时，停止物理阶段，交付 `BLOCKED_WITH_EVIDENCE`，不编造接口。

### 3.3 两道容易遗漏的门槛

**门槛一：decode fixture 与 Prefill prepare 的路径可比性。** 现有捕获是 layer0 一个真实 decode 行，而热点是 Prefill prepare。先证明组件调用的是被观测的 prepare provider，或明确它与该 provider 的真实映射关系。若只执行 resident decode，没有触发 shadow 准备，则实验不能回答所选问题。

**门槛二：精度/分支与 reference 可比性。** 捕获结果来自 MXFP4 CPU + BF16 NPU 混合执行。若 Prefill shadow 测试使用了不同的有效权重精度或分工，不能强求它与该 decode 输出 bit-exact，也不能把差异视为新回归。必须使用同一路径的既定 BF16/混合精度 reference；缺少参考时先只交付路径证据，不给数值 PASS。[S1:L11，L23][S3 §6.3]

## 4. E1：冻结两个受限 workload 身份

### 4.1 原始真实组 A

优先使用 [S2] 指定的 captured layer0 top10 route、真实输入、同 revision/MXFP4 bank/NZ29 布局、实际 Q32 resident map。保持原 fixture 不变；hash、行来源、routing weights 和参考输出各自索引。

明确 `resident_mask_id`、`routed_expert_set_id`、`scheduler_chunk_id`、`expert_microbatch_id`、`shadow_lifetime_id`。**Q32、top10、32-token 和 32-slot 不能合并成一个 chunk_size。**

### 4.2 一个 missing-expert transition B

先从已有正常 trace 中找同 layer、同来源/布局、现有容量能够容纳的缺失 Expert 转换。目标是观察当前实现如何处理它，不是实现增量复制。

如果已有 trace 没有这样的转换，不能改写原真实 fixture 然后继续称“真实捕获”。仅在批准的组件输入范围内建立一个显式标记 `DERIVED_COMPONENT_CASE` 的派生用例，列出改动的路由/输入、原来源和独立 reference；不得改生产 resident map。

若无法得到合法派生用例或同语义 reference，则报告 `MISSING_EXPERT_TRANSITION_NOT_COVERED`，不靠额外完整模型请求补齐。A、B 各自有参考，不能拿 A 的捕获输出验证 B。

### 4.3 同样流程、两种观测模式

两次组件运行分别采用现有观测 OFF 和新增分解观测 ON。使用相同 A/B 输入、原 prepare/consume/retire 次序、同资源配置。除观察外不改变计算、复用和等待策略。

如果原实现每次都完整 prepare，就照实测完整 prepare；如果它原本有安全保留，就记录其原有条件。不为了实验“看见复用”新增保留，也不人为插入重复 prepare 来制造热点。

## 5. E2：只增加能回答问题的最小子边界

建议归因类别如下。它们是观测分类，不是要求创建同名新函数，也不预设真实执行顺序。

| 分类 | 记录什么 | 不能误称什么 |
|---|---|---|
| canonical BF16 materialization | 来源开始可访问到本次可用张量形成；逻辑 bytes、真实分配、page/cache 状态是否可知 | 未观察到实际 I/O 时不叫磁盘带宽 |
| ND/NZ parent construction | view/alias/新 storage、format/layout 转换、padding、转换前后身份 | 不能把包装 view 的时间当完整格式转换 |
| copy enqueue | Host 发起 copy 的调用边界、目标 slot、提交 bytes | 不等于 DMA 完成或传输带宽 |
| copy completion / dependency wait | 既有 event 的设备区间；消费者等待；完成时 source/slot 可用性 | 不把 wait 与其等待的设备工作重复相加 |
| retire / release | 最后读者、安全归还、baseline 变化；与准备 parent 的关系 | 已返回 Python 函数不等于所有权可归还 |
| residual / unresolved | parent 内尚未解释的同域间隙及原因 | 不能用 Other 偷偷表示已经解释清楚 |

metadata 序列化、大张量 hash 与完整 tensor 抽取放在受限核验阶段，不塞进每次计时热路径。设备事件统一在安全 drain 后读取，不为每个子阶段加全局 synchronize。[S3 §7]

### 5.1 Host 与 Device 分开记

Host 单调时钟回答调用、阻塞、资源转换；Device event 回答对应 stream 上的区间。两者没有校准关系时，只建立 correlation ID 和先后依赖，不画精确跨流重叠，也不直接相减。

某个设备复制若在 prepare 返回后才完成，应独立记录 tail。不能把其完成点强行塞入较短 parent，也不能为了方便记账增加等待改变原行为。

### 5.2 分解是否闭合的判据

对于同一 Host parent `P`，将能明确映射的同域 child 区间限制在 P 内，计算其区间并集覆盖长度。只在同一时钟和合法父子关系下使用：

```text
T_parent   = end(P) - start(P)
T_covered  = duration(union(valid_host_child_intervals within P))
T_residual = T_parent - T_covered
```

重叠 child 先取并集；父/子不相加。设备时长另列，不放入这个 Host 覆盖公式。缺少 end、时钟不明、出现无法解释的负间隙或跨父边界时，要判 invalid/incomplete，而不是取绝对值或补零。

真实未覆盖间隙可大于零；应说明是已知未包围逻辑、观察器开销，还是仍未知。允许误差必须依据时钟分辨率、事件记录开销和范围定义事先记录，不能看到结果后临时用一个百分比把差额抹平。

**验收不是“把每一毫秒硬填到某一项”，而是“每一项有边界，无法解释的差额公开，结论不越过覆盖范围”。** 若缺口仍决定不了哪个环节值得改，实验状态继续 `PARTIAL_DECOMPOSITION`。

## 6. E3：先验证观察器，再进行获准的组件运行

### 6.1 离线/冻结依赖层测试

| ID | 具体场景 | 预期结果 |
|---|---|---|
| O01 | 观测 OFF | 不增加实际 prepare/copy 调用，不改变返回值/异常，不增加持久设备分配 |
| O02 | 首个 event 记录失败 | 底层模型调用仍按原语义发生；关闭后续计时并降级证据 |
| O03 | 底层 provider 抛异常，观察器也失败 | 保留原异常，不能被观察错误覆盖 |
| O04 | 未完成 preallocate、只读 slotted adapter | 保留已修复的绑定顺序/proxy，不实例 monkey-patch |
| O05 | begin/end 缺失、空 trace、容量溢出 | 状态降级或拒绝分解，不输出成功占比 |
| O06 | 父子嵌套与两个相交 child | 并集不重复计数，未覆盖区间准确保留 |
| O07 | Host/Device 无关联或 child 越界 | 不计算跨流 overlap，不伪造 parent 闭合 |
| O08 | model/layer/map/source/layout 不匹配 | 在设备工作提交前拒绝错误 fixture |
| O09 | 相同 expert ID、不同 layer/source | 不命中同一 shadow 身份 |
| O10 | 取消/drain 的既有所有权逻辑 | 不增提交，不提前回收 source/slot；现有 baseline 恢复 |

先使用现有测试入口定位新增测试文件和 exact 命令；本任务书不假设仓库已有某个 pytest 路径。回归出现断言失败则先停止，不用增加设备请求碰运气。

### 6.2 提议的最小物理预算：仅在新一轮明确获准后

| 项目 | 建议上限与解释 |
|---|---|
| 完整模型启动 | 0 |
| 新 HTTP 请求 | 0 |
| 新完整模型设备 profile | 0 |
| 组件进程启动 | 最多 2 次：OFF 参考、ON 分解各一次 |
| 每个进程的组件 lifecycle | 1 次，含 A 与一个合法 missing-expert transition；共最多 2 个 lifecycle |
| 同时活跃测试 | 1，串行执行 |
| 新驻留/staging | 0；沿用原 32-slot reservation |
| 自动失败重试 | 0 |
| 预热、额外重复、重型 profiler | 不隐含批准；确需增加时单独列预算 |

以上是本评审建议的**新预算方案**，不是 [S2] 已给出的授权。一次 OFF/ON 不能产生正式速度提升结论；这一小预算只用于路径、数值、归因和生命周期证据。

若原 watchdog/资源配额无法容纳一次合法组件调用，交付阻塞记录，不擅自延长超时。组件停止与资源回收必须沿用既有安全路径，不能为了“及时结束”中断在途权重状态。

## 7. 共享安全和构建边界继续保留

以下数值从上一轮 [S3 §3] 沿用，启动前仍需重新准入，不因本次回放显存低而降低门槛。

| 项目 | 约束 |
|---|---:|
| 物理卡 | 15，核对容器内映射 |
| 训练任务 HBM | ≤15,360 MiB |
| Round9 allocator | ≤37,376 MiB |
| Round9 owned process | ≤38,400 MiB |
| 整卡 HBM | ≤56,832 MiB |
| 最低保留空间 | ≥8,704 MiB |
| MXFP4 Host 准入 | ≥48 GiB / NUMA-node |

训练边界使用原批准的 `/mzy-train` 完整容器 ID 与 cgroup，不把同名新容器自动视为已批准；PID 轮换按原合同处理。身份不明、未知外部进程、监控失效或资源不满足时 fail-closed，只停止/不启动 Round9 自有任务。

不修改训练，不全局 `drop_caches`，不放宽资源门禁，不主动制造 Host 冷缓存；文件级 DONTNEED 只沿用此前批准的自有文件清单和前置条件，不推广成常规 benchmark 手段。若实际执行过已授权清理，将其记入 run 的 cache/NUMA 条件。[S1:L36][S3 §3]

不热改 v63，不覆盖 v20/replay snapshot。先确认真实 provider 文件是否属于冻结核心：若只有修改被冻结的核心文件才能插入子边界，则需单独记录最小 diff 并取得相应允许；不能声称“只是观测”就隐瞒字节变化。禁止复制一份 prepare 算法绕过这个约束。

保留 PLE Async OFF、prefix cache OFF 和原其他 feature 设置。不 push、不合并 main、不打 tag、不恢复 Round7 定时任务；`2026-09-20-04-PAUSED.md` 不修改、不暂存。

## 8. E4：验收条件与唯一决策

### 8.1 本轮分解实验的通过条件

必须同时具备：真实来源/分支对应关系；明确且冻结的数值协议；OFF/ON 未改变原计算；A/B 生命周期及资源正确；parent/child/残余口径可信；子成本有对应张量/存储身份。

原始 capture exact 只用在同一 capture 语义；不同精度分支用各自冻结 reference。不能为了通过分解测试让 MXFP4 与 BF16 强行逐位相等，也不能临时放宽已有误差门槛。

通过只意味着 `COMPONENT_DECOMPOSITION_VALIDATED`（本方案建议的新状态名）。它不升级 `PARTIAL_ATTRIBUTION` 为全模型归因，不升级产品/性能资格，也不证明改变策略有收益。

### 8.2 数据如何决定下一项动作

| 经分解后的证据 | 下一项可以提出的窄改动 | 仍不能直接做 |
|---|---|---|
| materialization 占明显可消除的同步成本，且同身份有效寿命内有真实重复 | 限定同 layer/source 的 materialization 消除或复用实验 | 全模型常驻 BF16、跨层缓存 |
| ND/NZ 实际新分配/转换重复，且可证明表示等价 | 一个布局/parent 构建改进实验 | 同时改 GEMM、routing 和 bank |
| 设备 copy 完成成本确实暴露，字节与时间同边界 | 一项有依赖与预算证明的复制/搬运实验立项 | 仅因 Host 时间长便宣称链路带宽瓶颈 |
| 相邻集合存在合法 slot 保留且 identity/lifetime 成立 | 单独立项 changed-slot；先冻结可保留集合 | 未证实缓存命中就实现增量复制 |
| retire 或其他 reader 等待主导 | 针对一个明确 reader/dependency 的生命周期调查 | 提前释放、绕过 fence/event |
| 组件不复现完整模型准备现象，或残余仍主导 | 给出不匹配原因与最小缺证；不选优化 | 以组件小样本外推全模型吞吐 |

“明显可消除”的判断必须来自同边界数据和依赖，不从总 bytes 或一次父函数时长直接推断。下一轮行为修改仍需要独立计划与测试预算。[S2:L24–L25]

## 9. 强制交付：本轮不能再以“建议继续泛泛 profile”结束

以下是**建议新增的文档/记录名称**，不是本次已生成或远端已存在的结果：

| 交付 | 必须包含 |
|---|---|
| `SHADOW_PREPARE_CONTRACT.md` | 真实文件/符号、来源、Prefill↔replay 对应、精度/参考、32-slot 和寿命 |
| `SHADOW_PREPARE_DECOMPOSITION.md` | parent/child/未覆盖时间、Host/Device 分列、bytes 类型、allocation/alias 与 first-use 条件 |
| `shadow-decomposition-events.jsonl` | 有界事件、run/parent/phase/correlation ID、时钟域及错误状态 |
| `shadow-decomposition-result.json` | 数值、覆盖、OFF/ON、A/B、资源与状态；未测项目不填 0 |
| `NEXT_SINGLE_CHANGE_DECISION.md` | 一个有门槛的后续候选，或具体到接口/缺证的 NO_CHANGE |
| patch / test / identity 清单 | 区分基线、观测构建、组件 snapshot；记录最终资源归还 |

源代码改动清单必须来自 E0 的真实定位。本方案不写虚构仓库路径或可复制执行的假命令。

### 9.1 建议状态初态

以下 JSON 是计划状态，不覆盖执行方现有 `NEXT_ROUND_STATUS.json`：

```json
{
  "plan": "EXPERT_SHADOW_PREPARATION_COST_DECOMPOSITION",
  "plan_state": "PROPOSED",
  "execution_authorization": "NOT_RECORDED",
  "baseline_and_provider_contract": "NOT_RUN",
  "prefill_replay_path_compatibility": "NOT_RUN",
  "reference_compatibility": "NOT_RUN",
  "component_off_on": "NOT_RUN",
  "missing_expert_transition": "NOT_RUN",
  "parent_child_reconciliation": "NOT_RUN",
  "numeric_validation": "NOT_RUN",
  "ownership_cleanup": "NOT_RUN",
  "behavior_change": "NOT_AUTHORIZED",
  "inherited_reported_attribution": "PARTIAL_ATTRIBUTION",
  "PRODUCT_QUALITY": "PENDING",
  "PERFORMANCE_QUALIFICATION": "PENDING"
}
```

## 10. 可交给本地执行模型的任务摘要

> 从本轮交付的 v63 观测构建和修正真实 map 后的 replay snapshot 分别恢复身份，先读取现成组件图、归因报告、问题记录、正常 Prefill trace 与 v20 回放证据。不要从旧 v61 直接启动新的全模型扫测，也不要沿用已经关闭的物理预算。
>
> 唯一主实验是 Expert shadow preparation 分解。首先确认 layer0 decode fixture 是否能够驱动被观测的 Prefill prepare provider，并核对 BF16 shadow 来源与 MXFP4 CPU/bank、有效精度和参考协议的关系。不能以 decode resident 调用代替 Prefill shadow，也不能把不同精度输出强行作 exact 比较。
>
> 在原 32-slot reservation 内，观察 canonical BF16 materialization、ND/NZ parent 构建、copy enqueue/completion/wait 和 retire；保持原算术、source、resident map、资源与等待次序。一个真实专家组和一个合法 missing-expert transition 分别有明确身份；派生用例不能冒充真实捕获。观测默认关闭，不重写 ring、不新增 staging、不缓存跨层 shadow、不改 kernel。
>
> 先做离线错误/身份/父子区间/观察器回归，再在新授权后按最多两次组件启动进行 OFF/ON 对照。本方案不安排新的 HTTP 或完整模型启动。时钟未校准不算跨流 overlap；Host 记账 bytes 不假称 DMA；父子与等待区间不重复计数。
>
> 保持卡 15、训练完整容器 ID、所有 HBM 和 48 GiB/NUMA 准入约束。身份/资源/参考不明则停止物理阶段，报告具体阻塞。结束后交付子成本、shadow 寿命、数值与资源证据，以及唯一下一改动决策；没有可支持的优化就写清为什么。保留产品与性能资格 PENDING，不 push/merge/tag，不修改训练或用户未跟踪文件。

## 来源索引

- **[S1]** [本轮主线交付摘要](sources/ROUND9_MAINLINE_DELIVERY_ZH.md)。本轮事实的主要来源；其链接的远端原始证据未包含在本次上传中。
- **[S2]** [本轮窄实验决策](sources/NARROW_EXPERIMENT_DECISION.md)。保留 `NOT_ENTERED`、`NO_BEHAVIOR_CHANGE_SELECTED` 和下一轮提案的原始含义。
- **[S3]** [上一轮实施方案](sources/05_Round9下一轮方案_KT主线与Qwen38异构推理.md)。用于对照任务和约束，不当作执行结果。
- **[S4]** [官方 KT/SGLang 对照与能力边界](sources/02_官方KT_SGLang对照与能力边界.md)。作为历史材料，不代表本次重新核验官方最新版。
- **[S5]** [功能补齐的设计重点](sources/04_功能补齐的设计重点-1.md)。用于保留 identity、生命周期、计时与资源约束。

`[S1:L7–L12]` 等表示来源文件中的实际行号；`[S3 §6.2]` 表示来源章节。本文的“建议”均是待执行设计，不是新增测试结果或设备运行授权。

## 11. 2026-09-22 执行修订（本节优先，用户已批准修订后执行）

本轮授权08 E0–E4及09已有证据审计并行。09不增加设备请求，不开启M2优化实现或M3资格测试。子智能体统一gpt-5.6-sol medium；物理进程由主执行者串行管理。

1. 共享系统上限不变。组件实际限制：allocator4%且不超过既有v20实际上限；owned4096MiB；Host64GiB、memory-swap64GiB；cpuset沿原160物理核、quota136、NUMA2/3/6/7。workspace512MiB；每进程1200s、guard1250s；最多2组件启动（OFF/ON各1），自动重试0。trace最多2048events、8MiB；0模型启动/HTTP/完整模型profile。任何无法满足之处先记录阻塞，不扩大预算。
2. 32-slot private shadow约300MiB，最大ND projection约200MiB，另计模板与workspace。零新增staging指不增加原路径外的缓冲区；原分配和同步全部保留。
3. 实际parent为TorchKTEPAdapter.copy_experts。分解source read/host tensor+finite/cat、ND allocation、blocking copy调用、index_copy、既有synchronize、staging retire。compute后的shadow release另列，不塞入parent。只新增默认OFF观测与派生snapshot，不热改v63/v20，不复制prepare算法。用户本次修订后执行授权包含这些已指明边界的最小观测diff。
4. A为真实decode捕获输入/路由驱动的原BF16 shadow provider，不称真实prefill捕获。B为明确派生用例：替换一个路由expert，保留输入与权重；A/B分别使用原BF16 shadow数学reference，相对L2<=0.01。旧混合精度capture output只保留来源，不充当BF16 oracle。组件全生命周期一次构造，A、B各prepare/compute/drain/release一次，不新增预热/重复。
5. 必要身份核验→原模板构造→shadow preallocate→输入构造和baseline→A→B→CPU reference/identity核验→cleanup。每项初始化可能的cache影响写入结果。OFF/ON不作为速度提升或观察开销百分比。
6. PLE本轮NOT_EXERCISED；全局默认OFF不变，历史v63显式ASYNC配置不改写。保留prefixOFF及生产resident map。
7. 结果分维度记录execution、A numeric、B transition、decomposition、ownership、next decision；无B时不能标整体validated。Host union仅在合法parent内；device独立事件时长不作DMA或跨流overlap。所有未知为null/UNKNOWN。
8. 09输出配置身份绑定的A/B覆盖矩阵及canary契约；旧兼容证据继承不重复跑。M3只面向窄主配置，非全矩阵阻塞条件。未push/merge/tag、未恢复Round7定时任务。
