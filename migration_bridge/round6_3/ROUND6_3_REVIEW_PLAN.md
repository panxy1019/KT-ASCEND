# Round6.3 审阅方案：真实 Decode Deferred 资格与受控 Batch 扩展

日期：2026-09-15。
状态：**DRAFT_FOR_REVIEW — 本文不是执行授权记录，不代表任何新gate通过。**

## 0. 推荐决定

建议批准两条有先后门禁的工作线：

1. **R6.3-A：先修验证工具，完成 batch1 的真实 KV-cache teacher-forced
   decode qualification。** 保持官方 Deferred 选择机制与D1 placement。
2. **R6.3-B：受控 batch2→4 pilot 与资格验证。** batch8只在2/4有可测
   收益且已通过对应安全门禁后另行决定；本计划不直接批准batch8/16长矩阵。

先减少重复prefill、capture/I/O开销，再用batch提升吞吐。不要同时改
selector、线程池、NUMA、模型和量化。当前Round6.2保持
BLOCKED_INSUFFICIENT_EXPOSURE，绝不倒改为PASS。

依据及官方链接见 `ROUND6_2_BLOCKER_ANALYSIS.md`。

## 1. 冻结范围

| 项目 | 冻结值 |
| --- | --- |
|模型|DeepSeek-V2-Lite，604d5664dddd88a0433dbae533b7fe9472482de0|
|硬件|A3，NPU0 only，专用disposable container|
|并行|TP1；不引入HCCL、多NPU|
|CPU/NPU|LLAMAFILE F32 GGUF / Ascend BF16 unquantized|
|D1 producer|L7，CPU {16,38,41,50}|
|D1 terminal|L8，CPU {8,25,33,55}，不允许继续defer|
|对照|defer0 vs defer1，其他model semantics相同|
|生产路径|现有overlap/framework tensor copy；不能用sequential替代|
|特性|Graph/Deferred之外的Dynamic/MTP/Speculative/量化扩展均OFF|
|旧状态|Round6.2不足500的结论永久保留|

当前基础提交：`5e708dc63d07afe4d79d26f0bdbb97d95ad3aaba`。
建议执行分支：`codex/round6-3-decode-exposure-qualification`，审批后创建。
本地SGLang `0a667fee2afd4b2d65fa6e4bff13a714f1407f37`；A3仍是既有
checkout+hashed overlay，开工必须重新记录实际部署，不能将本地HEAD
冒充A3执行SHA。不改已有tag，不rebase官方main。

新容器/新output roots优先；若复用专用容器，明确记录extension与源码
hash，并保持原始证据只读。sudo权限不用于修改宿主系统库或其他任务。

## 2. Gate图与执行顺序

```text
审批
→ provenance/source audit
→ 离线路由选择机制复核
→ test-only stateful teacher-forcing + transaction capture
→ A3 capture/observer parity + negative controls
→ 独立 pilot：exposure + 成本 + batch2/4可行性
→ 冻结新 QD3/HD3/FD3、统计方法、样本量、模式
→ QD3 batch1完整性与exposure门禁
→ QD3数值推导 → candidate commit
→ D1 runtime regression
→ HD3 one-shot
→ FD3 free generation → downstream quality → ADR-A
→ 若A通过：batch2/4独立门禁与受控吞吐验证
→ 最终文档与资格范围冻结
```

任何hard gate失败，停止其下游。B失败不倒推A失败；也不能用B的结果
降低A/B数值门槛。pilot与正式qualification必须使用不同数据与结果目录。

## 3. WP0 — Provenance与事故隔离

执行git status/HEAD/childSHA/源码SO哈希/环境/placement/GGUF/tokenizer复核。
保留原始Round6.2全部日志、旧失败、retry成功capture、343 gate输出。
禁止拼接旧run位置生成新PASS manifest。

固定官方参考快照 `0dce4c9b097710951d1b3e232f50b973ff0a5505`。
仅审计selector、buffer容量、callback/submit、批处理能力，不整体合并。
如未来确需port，先给出具体文件/语义差异和最小patch；生产修复独立commit。

产物：`00_BASE_AND_SCOPE.md`、`01_SOURCE_SEMANTIC_MATRIX.md`。
Exit：provenance明确，旧gate不可变，所有新增运行可追溯。

## 4. WP1 — 离线验证 exposure 机制，不改算法

只读现有QD2 captured router IDs/weights/immediate/deferred IDs：

- 记录M、每行top5保护IDs、全forward保护并集大小、CPU-owned候选数。
- 精确核对实际分割是否符合官方 union-protection；相同score ties使用
  实际设备产生的选择结果，不用CPU不同tie顺序误判NPU。
- 分解零exposure：无CPU候选 / 低分候选为NPU-owned /
  CPU低分候选被其他行保护 / terminal禁止defer。
- 做固定operands的反事实诊断：同一行单独选择与原prefill一起选择。
  这不是改变生产selector，也不是模型decode数值等价证据。
- 量化“跨行保护”解释了多少未延迟CPU路由；区别概率机制与已验证因果。

此阶段不得读取QD2/HD2跨模式logit差异来选新参数。
Exit：selector逻辑核对PASS；若实际mask与机制不符，先定位bug，不扩语料。

## 5. WP2 — 真正有状态、同历史的teacher forcing

### 5.1 核心协议

每个prompt一次prefill，然后63个实际decode目标位置，目标总数仍64。
defer0生成baseline token history后，freeze原始bytes与SHA。
defer0 replay和defer1 replay都输入完全相同的baseline token序列。

**两条路径各自维护独立KV cache。** 不能把All-NPU/defer0的KV复制给
defer1；模型中间执行不同，缓存必须由本路径的历史计算生成。
“token history相同”不意味着两个KV byte相同。

### 5.2 实现约束

优先使用现有SGLang runner/scheduler API可支持的step机制；没有合适
稳定API时，允许最小default-OFF、test-only forced-token adapter。

adapter在真实logits与natural top1记录后，依据预冻结的
`(request_id, step) -> baseline token`表，统一指定下一步消费的token。
必须在scheduler/KV/output_ids状态推进的正确边界生效。
候选代码位置是本地ModelRunner.sample和scheduler结果处理之间，
具体位置须源码验证后再选，不能只改返回给HTTP客户端的字符串。

禁止：

- 修改logit值/logit bias或mask让baseline token人为成为top1；
- 只对已知失败token做tie-break；
- 让Hybrid自行free-run后声称是matched-history；
- 在prefill里一次算所有未来token并当作decode；
- 用不同mode共享KV缓存；
- test开关污染普通serving请求。

分别记录raw自然top1、forced next-input token、两者不同标记。
原先“API output必须是logit argmax”的断言不能原样用于forced token；
新的断言应验证natural top1来自原始logits，forced token精确等于manifest。
普通free-generation仍保留原argmax检查。

### 5.3 事务schema

必须有：run/session UUID、forward sequence、request ID、目标step、mode、
prefill/decode、原始input IDs、absolute positions、seq_lens、history hash、
本路径cache lineage标识、natural logits/logprob、强制token、实际消费token、
CPU/deferred routes、protected IDs、pending/consume/drain、source provenance。

目标位置t的exposure包含该路径本请求此前prefill/decode已发生的Deferred
与当前步骤；这是因果历史暴露定义，不只是当前forward是否delay。
同时独立报告current event count、累计exposure及**至少有一次暴露的prompt
数**，防止累计标记看起来很多却由很少prompt贡献。
padding、lookahead和取消请求不得计入64个qualification目标。

### 5.4 验证门禁

- new adapter OFF与旧production路径在同进程/配置上exact一致；
- forcing baseline自身历史，与无forcing baseline生成结果对齐；
- 注入统一synthetic强制序列，证明实际下一步input/position/KV进度正确；
- 两模式所有步骤history hash一致，cache却分别维护；
- 20请求/代表性64步：observer OFF/ON无扰动，事务完整；
- 故意错RID、错step、缺forward、重复forward、错误token、越界/终端pending
  的negative controls全部fail-fast；
- 与正常production实际decode的锚点比较，不能拿full-prefix prefill作唯一oracle。

若稳定的production-like step注入不可实现，STOP并给ADR；不悄悄切回旧重放。

## 6. WP3 — 提升采集效率，先保持batch1

1. 使用一次prefill+连续decode，去掉64次重复prefix computation与HTTP往返。
2. append-only capture index/atomic completion事件代替每步全树rglob；
   首请求前empty-session检查、全run最终unknown-RID审计仍保留。
3. 正式长矩阵保留每位置全词表原始logits、路由与完整关联元数据。
   完整隐藏激活保存于预声明diagnostic subset/有限ring buffer，不能根据
   数值结果事后选择对自己有利的证据；出现错误即冻结相关现场。
4. full observer与compact observer先做byte/metadata parity，记录无法等价的
   字段，不把删掉的检查假装通过。
5. 分段计时：模型forward、D2H、dump/压缩、HTTP、读取/审计、CPU等待。

不改CPU线程/NUMA，不用unsafe ACL pointer copy换速度，不跳过全logit有限性。
Exit：相同pilot输入下证据等价，开销有实测，不承诺预定加速倍数。

## 7. WP4 — 独立pilot及样本量决策

用新增R63-PILOT（避免与历史Q/H/S/T/D/FD重复），建议32个不同任务，
各64步；不作为QD3或HD3。所有prompt必须完整保留，包括零exposure样本。

先运行batch1真实decode，记录：

- current-only、累计exposure、exposed prompt数及领域分层；
- 路由候选→保护→实际defer→consumer drain的完整漏斗；
- 完整capture/compact耗时、RSS/HBM/CPU/NPU使用时间序列；
- prefilling与decode分开统计，比较QD2重复prefill的机制差别。

500仍是正式Q/H exposure最低门槛，不因pilot低命中而降低。
正式规模从128/256/512 prompts候选中，在查看任何QD3/HD3数值前确定。
采用按prompt block的保守pilot不确定性估计与预算，预登记一次选择规则；
不能只用500/观察比例的点估计保证一定够。选择后freeze两套完整语料，
没有途中“缺多少再补多少”。若pilot无法支持预算内充足exposure，先STOP。

真实decode仍不足时，形成新的workload-design ADR；不能自动换expert、
增加defer预算或用路由筛选后的stress集冒充自然分布qualification。
若额外设exposure-rich stress集，必须与自然集单列，并有独立holdout与
限定的条件性声明，需审阅其协议，不能隐式替代本计划自然集。

## 8. WP5 — 新冻结数据与统计单位

新名称QD3 / HD3 / FD3；旧QD2只做机制诊断，旧HD2封存，不拿来试参数。
每套冻结prompt text、token IDs、baseline histories、order、seed、协议、SHA。
QD3/HD3必须跨集合、跨历史集合去重并审阅模板同源性，不以随机shuffle
证明任务独立性。推导与held-out明确隔离。

batch1主体保持已有V2的risk definitions：contraction、full-logit maxabs、
relativeL2，K32+natural serving anchor，stable exact、ambiguity membership、
outside-U zero、stable pair-order、NO_EXPOSURE byte-exact等hard gates。

继续95% prompt-block coverage、alpha=.05/3、5% prompt-block tail budget；
新N对应rank必须由固定公式自动计算并预登记：例如N128→127，N256→251，
N512→497。不能把rank127复制到任何新规模；不能用HD3调rank或乘系数。
置信度仍以独立exchangeable blocks假设为条件，不宣称普遍生产保证。

既有500只是coverage gate，不替代独立样本量和数值质量检查。
分别报告暴露/未暴露的误差分布；若对暴露子群提出正式统计声明，需要
单独预登记其有效N、条件分布、multiple-testing，不从总体声明自动推导。

## 9. WP6–WP9 — batch1正式资格

### QD3

全量输入/事务审计PASS→路由exposure>=500→才读取跨模式数值。
自然top1、候选pairwise数值必须是相同baseline history的实际decode输出。
任何hard gate失败停止；通过后freeze candidate JSON+code+corpus SHA并commit。

### D1 runtime

沿用原S6：12prompts×10次64token，加8/16/32prefix checks；fresh restart。
还必须完成terminal/no-pending、exact-once、whole-request CPU-not-hit、
explicit replay与1000代表性forward/lifetime测试，不把工具unit代替A3 gate。

### HD3

仅candidate已commit且runtime PASS后运行一次。保持已冻结reserves、
statistics、history生成规则；先完整性和exposure，再hard gates与tail gates。
数值失败拒绝，不重拟合；infra retry需事故记录且不变输入。

### FD3 / quality / ADR-A

FD3至少8个新prompt×64，真实free-generation首次分叉必须满足冻结
stable/ambiguity规则，不能把分叉后的不同history差异当intrinsic error。
质量保持原manifest/evaluator：C-Eval128、GSM8K MC128，paired bootstrap
20000、seed6202、95% CI，invalid/nonfinite不得豁免。
无显著回退不是严格equivalence/noninferiority证明。
所有前置PASS才批准batch1；否则ADR写BLOCKED的具体gate。

## 10. R6.3-B — batch2/4受控扩展

### 10.1 先pilot，不直接改正式矩阵

独立pilot按B1→B2→B4顺序，固定请求分组和step调度。
必须记录实际forward batch，而不是只记录客户端并发或max-running-requests。
不以低exposure的batch配置宣称Deferred更准确；每种资格配置都要满足
对应exposure与有效样本门禁。收益不足时保留batch1或batch2，不能硬上4。

### 10.2 batch-aware capture与状态

给每个request建立input span、logits row、seq_len、absolute position、
history hash、forced token、route owner；batch层记录所有成员与保护并集。
检查请求重排、不同长度prefill、padding、结束/取消、batch shrink/grow、
buffer slot reuse、跨请求pending误串。Graph与mixed-chunk保持OFF，
先固定组成的lockstep batch，再讨论动态continuous batching。

CPU侧global class buffer和pending按layer共享，不能假设B>1已隔离。
任意ownership/history/row mapping错直接FAIL。

### 10.3 shape与资源预算

必须验证sum(extend_lens)与decode行数均不越过CPU C++/Python buffer容量。
当前chunked_prefill_size=512的限制不能因为B增大而静默突破。
max-total-tokens是所有活跃KV的总预算；context limit是每请求预算。
若需扩大KV/预分配，先A3资源探测与manifest记录，仅改专用容器配置；
它是新的batch profile，不是原batch1完全相同环境。
NPU OOM/timeout/设备错误只终止本项目新进程，保持其他任务不动。

### 10.4 数值和统计

基线与Deferred必须使用相同batch成员、顺序、token history和实际batch边界。
不要求B1与B4所有logits bitwise一致；要求各配置自身确定性以及在冻结
数值契约下的相同配置defer0/defer1比较。

优先用A已批准的reserves对batch扩展作严格兼容性门禁，不为B2/B4单独
扩大阈值。失败则该batch profile BLOCKED，不损害batch1已批准状态。

固定同一batch内prompt有相互作用，统计单位用整组完整轨迹block，
不能把128prompt/B4=32groups伪装成128独立样本。
若保留128独立groups的正式验证规模，则B2需要256prompt，B4需要512prompt。
全部group在输出前冻结；不能在HD阶段更换batch partners。
不同B的声明各自标注置信范围；如要一个同时覆盖多个B的联合置信声明，
需提前把多profile也纳入alpha预算，不能继续只除3。

批量正式资格需要独立、预冻结的grouped validation dataset，不直接
重组看过结果的HD3来宣称新held-out。它可以更慢更大，应先核算必要性。
若只做短pilot，最终只能标记BATCH_PILOT_VERIFIED，不标正式质量READY。

### 10.5 吞吐acceptance

分别报告aggregate tokens/s、per-request latency p50/p95、TTFT、
CPU pool wait、D2H/H2D、RSS/HBM、实际batch分布、Deferred exposure、
其他NPU任务负载。先warmup，再至少3次固定workload重复。
比较相同模式同等输出token工作量，不用prefill token或lookahead虚增吞吐。
高NPU利用率不是PASS标准；吞吐提升但p95、exposure或quality恶化需显式报告。

## 11. 上游借鉴与保留边界

借鉴并保留：官方union protection、shape fail-fast、framework copy、
callback/submission边界、batch buffer思路。
本地保护：L7→L8 terminal规则、BF16-visible语义、FP32 partial merge、
source identity、capture boundary、同路径determinism。

不自动引入：最新upstream全栈、Graph、Dynamic、CPU backend替换、
KML/NUMA、worker redesign。官方README提到batch相关参数不等于本场景已验证。

## 12. 预计时间与预算控制

以下是工程估计，不是保证；需pilot后更新：

| 工作 | 初步预算 |
| --- | --- |
|源码/路由mechanism与协议审计|半天左右|
|stateful teacher-forcing/capture实现、negative tests|1–2工作日|
|A3 parity/pilot、I/O与batch可行性|半天至1工作日|
|QD3/runtime/HD3/quality|按pilot实测计算，可能数小时至1–2天|
|batch2/4正式统计与质量资格|独立追加预算，不能仅按pilot时间推断|

长run前必须打印预计请求数、有效position数、实际token-row量、磁盘预算、
基于pilot的P50/P95时长与timeout；超预算先STOP，不盲跑夜间矩阵。
保留低token监控选项；用户要求持续显示时保持当前对话简短进度更新。

## 13. 交付与commit纪律

建议目录：`migration_bridge/round6_3/`，包括：

- base/source matrix、routing mechanism、teacher-force protocol；
- capture schema、parity/negative tests、pilot report；
- new corpus/statistical preregistration及SHA；
- QD3/candidate/runtime/HD3/FD3/quality/ADR-A；
- batch2/4 profile、group statistics、resource/performance报告；
- final gate matrix、真实执行provenance、失败记录。

commits按test-only capture、teacher-force、效率、corpus freeze、统计、
证据分别提交；真实production bug另开fix commit。无须修复的生产文件不改。
不混入继承的llama.cpp dirty修改。无READY则不发READY tag；push/merge按
用户后续授权，不将方案草稿冒充已执行结果。

## 14. 审阅要点

推荐批准：

1. 先做真实stateful decode与测量工具修正，不改官方selector。
2. 正式QD3/HD3继续保留500 exposure minimum与现有统计方法原则；
   新N、history/exposure定义和实现hash必须在正式输出前freeze。
3. batch2/4允许pilot；batch1保留anchor，不直接用大batch替换主资格。
4. batch正式声明单独满足组级有效样本与相同mode数值门禁，不扩大reserves。
5. 本轮计划需要审阅后才执行；当前没有新模型run或生产patch。
