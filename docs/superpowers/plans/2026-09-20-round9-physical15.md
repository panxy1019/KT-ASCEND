# KT-ASCEND Round9 Physical15 Execution Plan

> **For agentic workers:** 使用 `superpowers:executing-plans` 在本任务中逐项执行；未获用户方案审阅通过前，不启动开发、转换或测试进程。复选框表示尚未完成的工作，不是已有通过证据。

**状态：APPROVED_IN_PROGRESS，2026-09-20。** 用户已审阅并明确要求开始执行；以下门禁仍须逐项实证。

**Goal:** 在 110.123.0.3 的物理 NPU ID 15 上，按独立身份、资源、源码语义、算子、系统与质量门禁，完成 Round9 异构工作集运行时开发与分配置资格认定。

**Architecture:** SGLang KTEP 统一协调 CPU/NPU；先建立同步 PLE、固定 Q64 的无新增权重量化参考，再单独叠加 CPU 量化、C128、PLE 异步与调度。训练共用环境逐级准入，性能资格与共享正确性证据分开。

**Tech Stack:** AArch64/Kunpeng、Ascend910/A3、CANN/torch_npu、SGLang、KT-Kernel/LLAMAFILE、GGML、Python/C++；实际版本、镜像与编译参数在阶段 0 冻结。

**Spec:** `migration_bridge/round9/EXECUTION_PLAN.md`、`05_CPU_PRECISION_REVIEW_AND_REVISED_PLAN.md`、`06_C128_EXPERIMENT_PLAN.md`、`17_CARD7_SHARED_TEST_ASSESSMENT.md`，以及两份 2026-09-20 handoff。本文承接最新用户指示：目标改为 .3/物理15，先审阅方案再执行。旧文中的 .22-only、强制全池 F32、heartbeat active 均不再适用。

## 1. 全局约束与已有证据

- 唯一测试目标：110.123.0.3，Card7/chip1/physical15，PCI `0000:83:00.0`、`/dev/davinci15`。隔离后进程逻辑 `npu:0` 必须实证映射到物理15；不隐式改用14或多卡。
- 模型：`Qwen/Qwen3.8-Flash-Next`，revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`。本机已发布目录 `/home/admin/models/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540` 只读使用。
- 目的端身份回执已 PASS：144 files、131 shards、1658 tensors；receipt SHA256 `7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239`。这是传输身份资格，模型执行仍为 NOT_RUN。启动前核验回执/manifest/文件清单；出现更改证据才针对性重新完整校验。
- Text-only、NPU TP1、batch1、max-inflight1；基线 Dynamic/Deferred/Graph/MTP/speculative OFF。48层、每层512 routed experts、Top10、hidden2560、intermediate640，36 GDN + 12 QSA，GR四分支。
- Q64 表示每层64个 NPU resident expert，完整 CPU backing 覆盖每层512个。它是工程参考，不是 All-NPU 独立真值。
- 默认先 C64；C128 是独立资格候选。有限 F32 canonical 用于算子对照；首个全模型优先 BF16 storage/F32 compute，不默认生成450GiB全量 F32 bank。
- 不恢复旧 round7 定时任务；不恢复 Round8 已取消的性能测试；不继承 DSV2/Qwen3.5 的资格或分数；不 push、不合并 main、不打 tag。
- 不修改/进入训练容器作为运行捷径，不停止训练，不修改宿主全局包、设备共享模式或驱动。已有训练没有 cpuset/内存上限，因此给自身设限不等于获得独占资源。
- 使用当前独立 Round9 分支；保护 `kt-main-integration` 的脏工作树。每个语义变更独立本地提交；使用受测提交/原生二进制身份解释结果。

## 2. 资源预算与共享准入

以下是建议申请与自身控制预算，均不是当前已分配资源，也不是实测模型峰值。

| 阶段 | CPU | Host RAM | 物理15 HBM | 说明 |
|---|---|---|---|---|
| 源码审计、构建 | 最多16物理核 | 32–64GiB | 0 | 限制并行构建与磁盘I/O |
| 格式/局部 CPU probe | 先单池，再64 expert + 8辅助物理核 | 64–128GiB | 0 | 仅按需取真实权重，不复制全池 |
| NPU 最小共存 probe | 单进程、单请求 | ≤64GiB | 先小分配，逐步到2–4GiB | 工程上限；非驱动硬隔离 |
| 真实算子/module probe | C64 | ≤128GiB | 满足准入后到8GiB | 每次启动前复查，总量包含运行时开销 |
| Q64/BF16 全模型 | 64 expert + 8辅助物理核 | 目标预算640GiB，准入后自身上限768GiB | 由测量决定 | routed resident payload 已28.125GiB，不能直接宣布整模型可共用 |
| MXFP4 / Q8_0 全模型 | 同C64 | 稳态初估384 / 512GiB | 同样独立测量 | 转换/加载峰值可高于稳态；CPU量化不减少NPU BF16 resident payload |
| C128候选 | 128 expert + 8辅助物理核 | 相同格式同预算优先 | 与C64同配额 | 四子池分片，禁止四份完整bank |

磁盘：源码与转换输出使用 `/mnt/data1/kt-round9` 独立目录，规划最多约1.2TiB总工作空间；原始source继续只读引用 `/home/admin/models`，不重复拷贝360GB源文件。保留 BF16 bank、当前一个低比特候选及有界临时文件，生成下一候选前复核剩余空间；不自动删除用户或其他任务数据。

NUMA：本机320物理核/640逻辑核、8节点；选择每物理核一个SMT线程，运行前依据实际负载冻结cpuset。C64=2×32 workers，C128=4×32；辅助核独立。两NUMA节点合计约500GiB，不能将640/768GiB全进程严格绑在这两节点；expert按分片first-touch，PLE/source页与其他内存使用单独记录的允许节点。NPU PCI locality当前未知，不能照搬 .22 核号或宣称最优亲和性。

HBM准入采用 `本进程测得/保守估计峰值 + 训练观测峰值 + 预留余量 <= 64GiB`。建议预留至少8–12GiB，且需理解训练阶段与可能增长；短时约55GiB空闲不足以证明可用预算。先至少10分钟低频只读采样并覆盖可辨识的训练阶段；若不能界定训练峰值，仅准入小额probe，不启动全模型。每级观测成功也不能保证训练未来不增长。

Docker host memory limit不限制HBM；可见设备限制不限制算力。需同时记录torch allocated/reserved和驱动进程/设备总HBM，包含直接ACL分配；监测缺失、设备身份不明或准入失效时不升档。停止自身进程只是回收措施，不是硬隔离保证。

## 3. 执行依赖与产物契约

主链：`0资源/运行时 → 1源码语义 → 2格式与C64 → 3模块 → 4全模型参考 → 5低比特 → 6同精度C128 → 7 PLE → 8调度/Q32 → 9长上下文 → 10性能/收尾`。

阶段1与阶段0的只读核对可交错；所有NPU探针必须经过阶段0。低比特或C128失败不阻断已通过BF16/C64上的独立PLE与静态长上下文工作；Dynamic失败则保留Static。任何旁路都明确记为原目标未通过，不伪装整轮完成。

新增证据放 `migration_bridge/round9/evidence/physical15/<run_id>/`；大权重/原始tensor留在服务器工作目录，仓库保存路径与hash，不提交大二进制。统一 JSON 顶层字段：`run_id, status, source_revision, git_commit, runtime_manifest_sha256, profile_id, device_physical_id, shared_device, prerequisites, resource_budget, observations, artifact_hashes, failures`。status取 PASS/FAIL/BLOCKED/NOT_RUN；性能另列资格，避免一个PASS混淆全部门禁。

计划新增小工具与测试（均尚未创建）：

- `migration_bridge/round9/tools/admission.py`：只读采集与准入判定；对应 `tests/test_admission.py`，覆盖错卡、遥测缺失、HBM超预算、SMT重复、依赖未通过，均拒绝启动。
- `tools/semantic_census.py`：header/权重轴/源代码映射清单；`tests/test_semantic_census.py`覆盖缺失、重复、错误shape和PLE层偏移。
- `tools/profile_manifest.py`：固定profile与协议hash；`tests/test_profile_manifest.py`覆盖缺少身份、改变mask/template/precision后不能复用旧资格。
- 后续算子与系统用例优先复用仓库现有runner，并为Round9建立独立manifest/结果目录；不复制一个不可维护的旧轮次测试系统。

上述 `tools/` 与 `tests/` 均相对 `migration_bridge/round9/`。每个实现任务先加入能真实复现错误的测试、确认失败，再最小修改、运行相关正反例、记录输出并本地提交。阶段0–3的实测结果用于冻结后续具体代码修改清单，不能现在凭空认定尚未导入的SGLang文件已具备目标接口。

## 4. 逐阶段任务与停止条件

### 阶段0 — 设备共享、资源、可重建环境（R9-A）

输入：目的端身份回执、现有 .3 资源清单。输出：`admission.json`、`runtime-manifest.json`、设备映射证据。

- [ ] 核验source/support回执与文件清单，检查空间、NUMA/SMT、cgroup v1、swap增量、训练占用；历史swap非零不直接判失败。
- [ ] 确认物理15、设备文件和进程逻辑设备映射。当前device-share=False、无vNPU；查明实际A3/26.0.rc1支持的独立进程/隔离访问方式，不能以A2文档代替实证。
- [ ] 建立独立运行环境、单owner锁、单活动worker与只管理本任务PID的退出机制。若需改变设备全局模式或影响训练才能运行，记录 BLOCKED_SHARED_ACCESS，停止NPU分支并报告具体条件，继续可独立CPU工作。
- [ ] 冻结精确gitlinks、镜像digest、CANN/driver/torch/torch_npu/编译器、CPU ISA与编译flags、实际import文件及native .so hash。旧镜像仅可作为重建输入，不能继承资格；宿主不升级。
- [ ] 在已支持的访问方式下做最小allocation/compute/synchronize/free，逐级至2–4GiB。验证进程退出后资源释放、错卡拒绝、超预算拒绝和遥测失效停止。

门禁：身份一致、只使用物理15、访问路线可解释、资源预算可满足、退出可回收。失败不进入8GiB及全模型。

### 阶段1 — 模型结构与源码语义（R9-A）

输入：阶段0运行时、冻结checkpoint；输出：`semantic-census.json`与逐模块兼容性矩阵。

- [ ] 重新按header细分main、routed/shared、PLE表/其他PLE参数、MTP、vision；验证1658 tensors/index闭合，不能用“所有含ple的键”替代精确table预算。
- [ ] 对照本地受测源码、官方KT pin `f7607c0f66c643220ff43b822b24483c98c41392` 与已有干净Qwen4 SGLang审计源。记录差异后最小移植，不整体追最新依赖；gitlink与实际import分别核实。
- [ ] 定义packed gate/up轴、down轴、ND/NZ、route weight、shared expert/scaling所有权、CPU/NPU分界；PLE配置[2]与checkpoint layers.1的layer_id+1映射写入测试。
- [ ] 核验Qwen4 config/registration、QSA、GDN、四路GR、PLE可真实import；逐项区分可复用、需移植和未支持，不能只凭模型名称注册成功判通过。

门禁：所有数学语义有明确实现或明确canonical；缺失模块/未知权重轴阻断全模型。结构header通过不等于数值通过。

### 阶段2 — CPU精度与C64最小算子（P0）

输入：真实早/中/末层权重、语义清单；输出：`format-capability.json`、局部差异与分片身份。

主要审计/修改边界：`kt-kernel/operators/moe-tp.hpp`、`operators/llamafile/moe.hpp`、`operators/llamafile/tp_slices.hpp`（后两者相对kt-kernel）。保留已验证的框架copy和NUMA分片，不机械移除guard。

- [ ] 用有界真实专家建立同值F32 canonical和现有production F32对照，覆盖decode、grouped-prefill、Top10、首末logical ID、重复/混合路由、CPU/NPU所有权边界。
- [ ] 先单池，再C64=2×32、intermediate320+320；加入640非256整除的失败用例，检查非F32目前限制与forward_many限制。
- [ ] 验证BF16 storage、F32输入/工作区/累加与FP32 partial merge。实际主机有SVE，不能依赖“无SVE时自动选F32 vec-dot”的源码条件宣称已支持；逐一记录实际类型/选中kernel。
- [ ] 局部探测MXFP4，再在主候选不成立时探测Q8_0；验证32weight块、scale/rounding/packing、down切分和回读。量化内核对同量化权重解码reference，量化损失单独统计。
- [ ] 测试每个子池只持有自身分片、merge仅一次、共享expert与route scaling仅一次；同一路径可重复，invalid logical ID必须拒绝。

门禁：适用的F32局部rel-L2<=0.01，copy/布局/source展开精确身份；BF16与量化各数值容差在看待测结果前以相同运算定义冻结，不能事后放宽。未通过不做全池转换。BF16失败先修实现；若需全F32，重新申请资源准入并记录reference方案改变。

### 阶段3 — 原始模型模块与边界算子

输入：阶段1语义与阶段2专家路径；输出：`module-gates.json`，最大8GiB范围的共享证据。

- [ ] QSA：真实indexer/block选择、padding、causal边界、selected-KV身份、prefill/decode一致性；不得用普通dense attention冒充原QSA。
- [ ] GDN：prefill/decode/chunk、状态更新、reset/cancel、跨请求隔离；GR验证全部四分支和read/write gates，非仅最终shape。
- [ ] PLE同步：ngram ID/history、BOS/EOS、层号、分片/row顺序、重复ID与chunk边界；所有miss必须拿到精确原始行，不以零值或旧行代替。
- [ ] KTEP：完整global logical expert ID，CPU/NPU所有权互斥且并集完整；ND/NZ转换先恢复完整ND再选择，写现有NZ目的tensor保留layout语义。
- [ ] 用同输入/权重的独立reference逐模块对照，覆盖真实层级与路由边界；有限性、状态释放和错误传播必须通过。

门禁：模块数学/状态正确且实际NPU执行身份明确；慢的reference path可以保留，但标为REFERENCE_PATH，不宣称优化完成。

### 阶段4 — Q64/BF16/C64静态全模型参考（M1）

输入：阶段0–3 PASS；输出：冻结 `QWEN38_Q64_BF16_STORAGE_REFERENCE_V1`（实际实现符合后才命名）、完整bank身份、系统与CMMLU参考。

- [ ] 先提交profile：固定64/512 per-layer masks、precision、同步PLE、C64拓扑、模板、thinking/greedy/stop/parser/invalid规则及context/output预算。首轮质量结果前冻结hash。
- [ ] 以有界流式转换覆盖48×512完整CPU backing，核对每层每expert的source/分片/格式；保留BF16原source供NPU promotion。避免整池重复匿名copy和全池pin。
- [ ] 分步加载并测实际host/HBM峰值，含PLE约95GiB、225GiB routed CPU payload、运行时和cache；仅当共享全模型准入成立才继续。共享预算不足记录BLOCKED_RESOURCE，不跳用Q32来冒充Q64参考。
- [ ] 小请求bring-up后运行32 compact；同进程各重复3次、两次clean restart、observer OFF/ON配对、reset/cancel与请求交错隔离测试。结果包含token、有限性、所有权与释放检查。
- [ ] 连续稳定性采用预冻结32 compact序列两轮，逐请求记录实际生成长度、峰值、残留与异常；EOS导致的短输出不伪报固定token总量。
- [ ] 先用独立DEV问题校准模板/答案解析及输出预算（初始候选max_new_tokens256、thinking OFF），冻结后完整CMMLU256一次；保留每题原输出、解析结果和无效输出。协议不适用时在正式计分前修正并重新冻结。

门禁：身份、模块、系统、canary/模板均通过后，参考分数才可用于后续比较。不得用旧模型222/225或174分作为本轮目标；infra失败保留原记录，同样本有限重试一次，未解决则该次测评不合格。

### 阶段5 — CPU低比特产品候选（M2）

输入：原始无新增量化参考、阶段2格式探测；输出：MXFP4或Q8_0的ADOPT/REJECT/BLOCKED决定。

- [ ] 先MXFP4；冻结converter/scale/rounding/packing及全部backing身份，NPU BF16、mask、PLE和其他模块不变。
- [ ] 依次通过同量化权重kernel对照、compact/重复/重启、CMMLU256；总正确数相对阶段4最多少12题，少13题即超过-5pp而失败。所有优化均对同一个原reference，不逐阶段累积容许损失。
- [ ] MXFP4实现或质量不通过时记录原因，唯一优先fallback Q8_0按相同门禁执行；两者失败则保留BF16，低比特目标标未通过。
- [ ] 分开报告容量与速度；共享环境可先决定“容量候选”，速度采用结论待阶段10，不因压缩倍数承诺加速。

### 阶段6 — 同精度C128（补充方案）

输入：已通过精度的C64 profile；输出：独立C128资格与默认拓扑决定。

- [ ] 先单独审计4×32 workers、160×4分片、first-touch、down/scale/tile边界和FP32四路merge；不是把workers参数直接改128。
- [ ] 保持source/precision/mask/PLE相同，局部真实形状→compact/重复/restart→CMMLU256；C64资格不能继承。
- [ ] 记录各NUMA页、跨socket访存、副本与barrier成本；禁止四份完整bank。不同归约顺序不要求跨配置bitwise相同，但各自确定且数值合格。
- [ ] 共享环境仅获得正确性/容量资格；正式同机同资源窗口比较后才选C128为速度默认。无收益或资源不能准入继续C64，不阻断主线。

### 阶段7 — PLE异步与联合预算（M3）

输入：已通过precision/topology的同步PLE profile；输出：同步/异步/联合预算三种同预算对照。

- [ ] 依序实现CPU gather overlap、有界pinned ring+框架H2D/event、最后有界device cache；每步先验证精确row和顺序，再叠下一步。
- [ ] request/token/history、task/slot/event/consumer lease组成完整身份；producer ready后读，consumer complete后复用。
- [ ] 强制测试迟到completion、cancel后slot复用、重复row去重还原、cache miss/淘汰、跨请求污染、copy失败；禁止取消任务写入新请求。
- [ ] 当前token必需数据优先于预取；prefetch仅用已知history。专家miss走CPU，PLE miss取原行，QSA miss取正确selected KV，三者不互换语义。
- [ ] 相同HBM/host预算测cold/warm、useful/wasted bytes和miss尾延迟；系统与质量门禁后再判断是否默认启用，无收益保留同步。

### 阶段8 — Frequency、Dynamic与独立Q32（M4/M5）

输入：冻结precision/PLE/topology；输出：每个调度/容量profile的独立资格。

- [ ] 冻结互不重叠SCHED_DEV/HOLDOUT的文本/token ID/长度/生成协议；禁止用CMMLU答案选expert。先每层固定64个Frequency，不改成全局top-K。
- [ ] DEV最多筛选两套预登记策略，选定后本地提交candidate和mask，再打开HOLDOUT；失败保留结果，不无限追加搜索。
- [ ] Dynamic先只在request quiescent boundary更新，再作为独立候选测试prefill-end；weight/mapping/mask同generation原子发布，完整512 CPU backing始终有效。
- [ ] 注入copy失败/半发布/旧generation读和tie边界；部分失败停止服务该候选，不以半新半旧状态继续。fresh-static必须同placement/input/KV/GDN/history，包含同prefill策略Dynamic-OFF对照。
- [ ] 检查migration-triggered覆盖；常规CMMLU没有触发时增加预冻结触发集，不用零触发证明Dynamic质量。收益扣除全context与迁移总成本。
- [ ] Q32另起Static资格，尽可能使用nested masks；同precision/PLE对同一阶段4参考跑系统与CMMLU，通过后才评Q32 Dynamic。容量缩小不继承Q64资格。

### 阶段9 — 长上下文分级（M6）

输入：已合格静态profile优先；输出：最高实际通过tier与各失败边界。

- [ ] 按4096→16384→65536→131072有效context逐级执行；输入预算=该tier减预冻结输出reserve，实际tokenizer长度写入证据，无截断、无未经验证RoPE变更。
- [ ] 每级先资源预估/准入，再运行固定known-answer检索、跨块结构/代码任务，核对QSA选块/因果、KV/indexer/GDN metadata及输出正确性。
- [ ] 不以QSA选择2048tokens推断历史KV有界；若必须host KV，作为新语义子功能先建立独立参考与门禁，未完成时该tier记BLOCKED。
- [ ] 训练共享预算不够时停在最高合格tier，不抢占训练或隐式换卡；非空输出、短CMMLU不能替代长上下文资格。

### 阶段10 — 正式性能与最终审计（M7）

输入：各正确性/质量合格的已提交profile；输出：Round9独立性能报告、默认配置及真实未完成项。

- [ ] 仅在受控资源窗口做正式性能：物理15无竞争或有可证明隔离，CPU/内存/I/O对照一致。共享计时只记诊断；没有窗口标NOT_QUALIFIED_SHARED_RESOURCE，不声称加速成立。
- [ ] 冻结短协议64 input/32 max output、中协议256/64，使用固定token序列、EOS策略和完整实际token数。每候选一次warm-up；每个工作负载两组ABBA对照，预算不随结果追加。需要改协议时新建协议ID，不混算。
- [ ] 只比较具有单一变因且已合格的候选：BF16/selected-lowbit、C64/C128、sync/async、independent/joint-budget、Static/Frequency/Dynamic、Q64/Q32。先做正确性，再按资源窗口执行这些有界Round9比较；不恢复Round8性能矩阵。
- [ ] timed path observer OFF，使用设备实际完成边界；单独诊断CPU wait、PLE gather/copy、QSA/indexer。记录startup/conversion、TTFT、prefill、decode、E2E、actual allocated/reserved/driver HBM、host anon/file/NUMA与峰值。
- [ ] 最终报告逐profile列身份/算子/系统/质量/容量/性能/长context状态，给出默认、拒绝原因、复现命令、原始证据hash和本地最终commit。未实现、资源阻塞和性能待定均明确保留，不把M1通过写成整轮完成。

## 5. 审阅重点与停止/恢复规则

1. 错把logical0当physical15：阶段0映射负例必须拒绝；只有设备名字相同不够。
2. 训练增长或遥测丢失：阶段0拒绝升档；运行中只停自身worker，保留资源快照。不保证轮询能防止所有瞬时HBM竞争。
3. BF16+SVE/640分片误入旧F32专用路径：阶段2验证实际kernel、类型与320边界；C128另测160，不删除guard绕过问题。
4. 异步cancel后复用导致旧数据污染：阶段3/7测试请求隔离和迟到completion，失败禁止进入系统质量评测。
5. 调度切换权重与mask不同步：阶段8负例触发fail-stop，恢复上一完整已验证generation，不继续服务半发布状态。

数值、finite、身份、状态隔离失败立即停止该候选；保存失败→最小复现→检查受测源码/原生二进制→最小修复→相关正反例→重新运行失败门禁。不得调宽阈值、挑题或丢弃失败掩盖问题。进程超时在各运行manifest中按输入规模预先冻结，超时保留为结果，不无限重试。

普通局部修复与通过后下一阶段在方案获批范围内连续推进；涉及训练停机、全局驱动/共享模式变更、换卡/换模型/TP变化时，提交明确的新资源条件给用户，不擅自扩范围。审批通过不表示上述共享准入条件已经满足。

## 6. 审阅后的首批可交付结果

批准后首先执行阶段0–2，交付：物理15共享可行性结论、可重建运行时manifest、Qwen4语义缺口清单、BF16/C64真实形状局部结果与全模型准入预算。若前置通过则按主链继续，无需每个正常小步骤再次询问。全轮耗时取决于源码移植、共享准入和实测CPU路径，当前不以尚未运行的吞吐给出虚假完成日期。

本方案自审：原计划R9-A、M1–M7及精度/C128补充均有对应阶段；共享设备性能资格、完整backing、原reference总质量护栏、失败旁路与未完成状态均显式保留。编制时尚未执行新增测试；获批后的实际结果见 `migration_bridge/round9/18_PHYSICAL15_EXECUTION_PROGRESS.md`，不能从计划复选框推定资格。
