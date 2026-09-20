# Round9 新窗口执行交接：Qwen3.8 异构工作集运行时

日期：2026-09-20。与 `2026-09-20-01-HISTORY_AND_REUSABLE_COMPONENTS.md` 配套。
本文件是最新决策的执行导航；完整原计划仍在 `migration_bridge/round9/EXECUTION_PLAN.md`。

## 1. 当前目标和真实状态

目标：Qwen/Qwen3.8-Flash-Next 在 Kunpeng–Ascend 上形成正确可测的
expert + PLE + sparse-context 异构工作集运行时。不是只启动服务器，也不是把所有旧优化同时打开。

当前 **R9-A IN_PROGRESS：冻结来源的模型下载中**。尚未完成完整模型身份审计，
未完成Qwen3.8算子/全模型/低比特/PLE/调度/长上下文设备资格。所有相应PASS仍NOT_RUN。
Round8已经完成用户缩减范围后的审计交接，不能重新要求Round8性能PASS才开始Round9。

最新用户要求：

- 保持后台下载；**停止定时任务**，迁移到新Codex窗口。
- 应用heartbeat `round7`（名称“Round9异构工作集推进”）已经PAUSED。
- 下载systemd服务仍ACTIVE，与暂停heartbeat是两件事。
- 暂停后没有代理定时检查/自动软件修复；下载程序自身仍会有界重试与做最终校验。
- 新窗口不要静默恢复旧线程heartbeat；如用户后续要求定时继续，明确由新窗口接管。

## 2. 工作区与来源身份

| 对象 | 身份/位置 |
|---|---|
| 本地工作树 | `/home/admin/Documents/ChatGPT/KT/kt-round8-qwen35-prep` |
| 分支 | `codex/round9-heterogeneous-working-set` |
| GitHub | https://github.com/panxy1019/KT-ASCEND |
| Round8闭环SHA | `9c1f1dae92303c1a13078319ab2aa2b5026fdb9e` |
| 当前模型 | `Qwen/Qwen3.8-Flash-Next` |
| 完整revision | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| 下载manifest | `migration_bridge/round9/manifests/download_source.json` |
| Metadata初始freeze | `migration_bridge/round9/manifests/preparation.json` |
| 最新下载实现commit | `41dfcd8a9`；后续交接文档commit不改变运行代码 |
| 远端执行主机 | `admin@110.129.0.22` |
| 当前SSH跳板 | `admin@110.123.0.3`，不是新的模型部署主机 |
| 远端根目录 | `/home/admin/kt-round9-qwen38` |
| 远端模型目录 | `/home/admin/kt-round9-qwen38/models/de4b8e4d43b917e7706784d8bb445c9af86a3540` |

模型131个safetensors分片，文件字节总量360,000,192,888；manifest144个文件还包含metadata。
不要混淆文件总大小与index中的tensor payload359,999,963,128。

当前source inspection记录的官方KT快照是`f7607c0f66c643220ff43b822b24483c98c41392`。
用户提供过SGLang pin`541ddc37cbc92c60dc748db5ff1a2aad0b069a80`，但本地记录仍为gitlink核验待办。
某猜测路径qwen4_exp.py的404不能证明整个fork不支持模型。需查实际注册和文件路径。
不得无理由merge/rebase整个最新上游。

## 3. 下载接管：先不要动健康进程

### 3.1 活服务与依赖

- `.22`：`kt-round9-range-recovery-v3.service`，单控制器，最多8个分片×每片2连接。
- 本地用户服务：`kt-round9-proxy-tunnel-v1.service`。
- 本地代理：`127.0.0.1:7890`；隧道提供远端`127.0.0.1:17891`。
- 远端HTTPS_PROXY/HTTP_PROXY使用该17891端口；源是官方HF冻结revision。
- **不要让本地代理主机休眠/关机，不要停止上述隧道。换对话窗口不需要停止它。**
- 若改到另一台桌面，仅复制文档不够：代理隧道仍依赖当前桌面。
- 凭据不写入文档。采用已有安全连接/交互授权，禁止把password/token输出到证据。

原v1/v2与其他HF/curl失败unit保留历史，不并行启动；启动多份当前控制器会触发同一stage锁。
systemd-run创建的transient unit停止后可能被回收；需要恢复时先查存在与否，不能假定systemctl start总可用。

### 3.2 最新已知进度（快照，不是未来实时状态）

2026-09-20 00:52 UTC只读核查：52/131完整分片、153.54GB；八个进行中分片约11.21GB，
合计164.75/360.00GB，45.8%；约20分钟平均9.2MB/s、无新增failure JSON、无OOM。
01:05左右交接检查时服务和本地隧道仍ACTIVE，进行中分片继续增长。
新窗口必须重新取值，不复述这些旧数字为实时结果。

### 3.3 只读检查命令

```bash
cd /home/admin/Documents/ChatGPT/KT/kt-round8-qwen35-prep
git status --short
git branch --show-current
git rev-parse HEAD
systemctl --user is-active kt-round9-proxy-tunnel-v1
ssh -J admin@110.123.0.3 admin@110.129.0.22
```

在远端（必要权限用sudo，口令交互，不写进文件）：

```bash
systemctl show kt-round9-range-recovery-v3 -p ActiveState -p MemoryCurrent
cat /home/admin/kt-round9-qwen38/download-state.json
find /home/admin/kt-round9-qwen38/range-recovery-v1 -maxdepth 1 \
  -name 'model-*.progress.json' -exec cat {} +
sudo journalctl -u kt-round9-range-recovery-v3 -n 30 --no-pager
cat /sys/fs/cgroup/system.slice/kt-round9-range-recovery-v3.service/memory.events
```

- `download-state.json`只代表最近一个活动transfer，不代表所有16个child。
- **旧`progress.json`已经过时**，新进度为`model-*.progress.json`。
- `*.verified.json`是单片完整SHA结果；不等于整模型通过。
- 计总进度时完整文件加尚未发布的partial，已发布分片的progress不能重复相加。
- 速率应取总新增有效字节/墙钟时间，而不是将16个连接速度估计相乘。
- cgroup memory.current可能主要是file cache；结合anon/file/oom_events判断，不能把缓存增长直接称泄漏。

### 3.4 下载器行为与已知限制

源码：`migration_bridge/round9/tools/range_recovery.py`、`download_guard.py`、`verify_download.py`。
实际部署副本在远端根目录，修改后核对SHA再部署，不能只改localHEAD。

1. 分段32MiB，每段Range/最终body长度严格检查；tiny tail允许32MiB传输上限以容纳302响应，但最终长度仍exact。
2. fsync后推进该分片offset；旧HF incomplete只复制不覆盖，失败part/header保留。
3. 整片SHA与manifest一致后hardlink no-clobber发布；已有完成文件最后统一全量校验。
4. 每range最多3次；传输失败分片在当前wave结束后按4/2/1并发继续有界尝试。
5. 校验错误不是普通传输重试；若服务失败，保留日志后诊断，不无穷systemctl restart。
6. 完成后自动调用verify_download.py做最终manifest身份审计；状态`DOWNLOAD_IDENTITY_PASS`仍需读取实际审计文件独立确认。
7. guard保留PAUSE_DOWNLOAD、内存/磁盘reserve与资源保护；CPUQuota400%、MemoryMax8G、swap0、idleI/O、磁盘300MB/s。
8. 8×2是用户授权吞吐配置，不是已完成整夜稳定性资格。运行初期约8.5–9.2MB/s；网络变化时ETA应更新。

保留失败链：原HF TLS/RemoteProtocolError长连接中断；ModelScope小样本没证明更快；
短range修复恢复能力。没有证据证明硬件网络问题已经修好。

### 3.5 下载完成后的第一步

不是直接load模型。依次：完整manifest SHA → tokenizer/license等身份闭合 → safetensors header census →
区分主模型/PLE/MTP/vision → architecture源码语义 → 实际资源预算 → 样本算子。
不要重复全量SHA多个并行实例；读取自动审计完成状态后按必要范围复核。

## 4. 决策优先级：精度与C128修订覆盖旧正文

阅读顺序：

1. 最新用户指令与本交接。
2. `migration_bridge/round9/05_CPU_PRECISION_REVIEW_AND_REVISED_PLAN.md`。
3. `migration_bridge/round9/06_C128_EXPERIMENT_PLAN.md`。
4. `migration_bridge/round9/EXECUTION_PLAN.md`其余未冲突内容。

旧EXECUTION_PLAN正文多处仍写Q64/F32，并写heartbeat持续ACTIVE：这些已分别被精度修订和最新暂停指令覆盖。
不能照旧正文自动生成450GiB F32，也不能自动恢复旧heartbeat。

### 4.1 推荐精度路线（尚未设备验证）

- 小规模真实expert F32：作为canonical/diagnostic数学参照，不强制全量。
- 首个无新增量化reference：先探测**BF16权重存储 + ARM F32工作区/计算/累加**。
- 主Low-Bit候选：**MXFP4**；唯一优先fallback：**Q8_0**。
- BF16 storage失败时先定位；有证据需要才回F32并重做资源准入。
- 权重16/4/8bit不代表全部算术或整个模型同精度；NPU仍BF16，PLE不同时量化。
- 正式参考若改为BF16 storage，首轮质量前冻结新referenceID，不继续称已执行Q64/F32。
- 所有后续CMMLU差值相对于同一无新增量化Qwen3.8reference，不允许逐阶段各掉5pp累积。

理论全expert bank payload：F32 450GiB；BF16 225GiB；MXFP4约59.765625GiB；Q8_0约119.53125GiB。
它们不含PLE、临时副本、scale额外布局、allocator与页缓存，不能当实测峰值。
Q64 routed NPU BF16 payload28.125GiB；Q32为14.0625GiB。PLE约95GiB只是先验估计，待真实tensor统计。

### 4.2 C128独立候选

C64维持2×32worker子池设计；Qwen3.8 intermediate640候选320+320。
C128优先研究4×32worker子池、160×4分片、NUMA本地first-touch；不是单线程池改128，不是NPU TP128。
32weight block整除只说明布局可能兼容，不能证明kernel tile、down reduction与prefill安全。
先选定格式并通过C64，再同精度C64/C128对照；新拓扑独立local/system/quality/perf。
当前没有分配128核，没有C128 PASS。无收益保留C64，不阻塞主线。

## 5. 架构前置审计：不能沿用Qwen3.5名字实例化

现有冻结metadata：`Qwen4ExpForConditionalGeneration`、`qwen4_exp`/`qwen4_exp_text`；
48层、hidden2560、512routed experts、Top10、expert/shared intermediate640；36linear+12full；
GR hc_count4/hc_lowrank320；PLE config层号[2]但index落在layers.1，必须查源码编号语义。
96个主expert packed keys，1658总index keys；尚待真实shape/dtype验证。

必须闭合：

| 模块 | 入全模型前要求 |
|---|---|
| Loader | gate_up/down真实axes、expert维、transpose、ND/NZ、排除MTP/vision |
| GDN | 真实prefill→decode连续state，chunk/reset/cancel与请求隔离 |
| QSA | indexer、选择block、因果mask、padding、K/V索引正确；不能替换为不同attention语义 |
| 四分支GR | 所有gate/norm/read-write/residual shape与源码一致，不能丢分支 |
| PLE | n-gram ID/history/BOS/EOS、shard/row/table身份与gather顺序精确 |
| KTEP | 512logical IDs、Top10、shared/scaling owner、CPU/NPU互斥完备 |
| Low-bit | 同一量化权重decode reference、scale/packing、ARM effective vec-dot和partial dtype |

参考路径允许慢但必须语义正确，标REFERENCE_PATH；公开NVIDIA/GB10/x86结果不是Ascend验证。

## 6. 后续严格执行顺序与出口

### A. 源码/资源闭合与独立环境

完成下载身份与tensor census；审计当前已验证runtime、官方KT fork、upstream SGLang三条线；
重新inventory .22的NPU、cpuset、NUMA、CANN/torch_npu、磁盘、host/cgroup及加载峰值。
复用既有构建资产但新环境隔离。资源不足记录BLOCKED_RESOURCE_ADMISSION，不能静默换机器/TP/模型。

### B. 小样本格式/算子能力

早/中/末层，GDN/QSA后MoE，真实decode/grouped-prefill、Top10、首尾expertID、重复/混合routes。
NPU/productionCPU/canonicalCPU三路。同源无损转换exact；局部F32安全门禁按原定义relL2≤0.01。
量化kernel对相同量化权重reference，不把量化损失当kernelbug，也不靠放宽门限跳过错误。
通过后才冻结reference精度、converter与全层转换/身份审计，覆盖完整512experts backing。

### C. 首个Q64 Static全模型reference

TP1、单请求/batch1、text-only、每层64/512NPU；Dynamic/Deferred/Graph/MTP/speculativeOFF。
同步host PLE必须存在，不允许0向量/随机行/完整表搬HBM。QSA/GDN/GR基础语义齐全。
32compact、同路径repeat、重启、observerOFF/ON、state隔离、finite、正常结束与资源回收。
冻结Qwen3.8自己的CMMLU256模板/thinking/token预算/greedy/parser，建立其参考成绩。
Q64是工程reference，不是独立All-NPU oracle；依赖局部与模块canonical证据防止“两条错路径互证”。

### D. Low-Bit与C128选择

先MXFP4，必要时Q8_0；只改CPU routed representation，不同时改PLE/NPU量化/mask。
same-quantized-weight局部正确性 → system → CMMLU对原reference → 容量 → 性能。
选定精度后比较C64/C128；保留原BF16 source用于未来promotion，不能用有损反量化冒充原权重。
低比特仅容量受益也可限定采用；失败明确UNSUPPORTED/IMPLEMENTATION/QUALITY_REJECTED。

### E. PLE旗舰路径

同步精确lookup → CPU gather重叠 → bounded pinned ring/framework H2D/events → bounded device cache。
每次绑定request/token/forward/history/table/row/generation、producer event、consumer lease。
取消后不写新request buffer；producer完成才读，consumer完成才复用。
预取仅用已知历史，不假设未来decode token。冷/热cache、chunk、跨请求、重启、reuse压力均验证。
共同预算比较保持HBM/精度/mask相同：当前token必需请求优先；没有收益则保留简单有界独立plane。

### F. Static Frequency / Dynamic / Q32

先Q64每层固定64slots的frequency，不能静默换成官方全局budget。
冻结独立SCHED_DEV/HOLDOUT，不用CMMLU答案挑experts；候选本地提交后held-out。
Dynamic先quiescent请求边界发布，再研究prefill后更新；weight/mapping/mask/backing同generation。
对fresh-static须相同placement/输入/state，不能忽略已经不同的KV/GDN历史。
记录迁移bytes/time、churn、hit、CPUweightmass、TTFT/E2E净收益，补等prefill策略的Dynamic-OFF对照。
Q32是同precision/PLE配置的独立Static缩容点，嵌套mask优先；通过后才谈Q32Dynamic。

### G. 长上下文与最终收口

4096 →16384 →65536 →131072 tokens，逐档资源与系统/已知答案验证。
记录真实有效输入、输出预留、无截断、选择block的KV身份；QSA选2048token不等于历史KV只需2048。
不能用短CMMLU或非空输出宣称128k。某档失败保留最高已验证档，后续NOT_RUN。
最终交付ROUND9_FINAL_REPORT/STATE，明确每个子目标采用、拒绝、未实现或受限，不能M1完成就全轮COMPLETE。

## 7. 质量、性能与权限不可变项

- CMMLU256 observed delta≥−5pp，相对同一个Qwen3.8reference；最多少12题，少13题失败。
- 不继承DSV2的174或Qwen3.5的222/225作为Qwen3.8质量reference。
- 旧B_pair只保留诊断，不重新作为heavy-offload通用hard gate。
- Dynamic短题未触发须报告trigger率，补冻结触发型覆盖，不用0trigger质量冒充迁移后质量。
- 正式计时同机资源安静、无diagnostic hooks，固定预算，不无限追加样本找收益。
- Deferred OFF且非关键路径；不做Qwen3.6；TP2/HCCL、Graph、MTP、多模态不自动进入核心范围。
- 仅授权.22；.16迁移已经取消，.20/其他机器没有因此被重新授权。
- 不push、不merge main、不打成功tag、不删除用户文件、不停止他人任务、不变更宿主库/全局网络策略。
- 普通故障：保留失败→最小复现→真实import/native审计→官方一手资料→最小修复→负例/回归→重跑。
- 不可恢复资源/设备/未知语义允许诚实BLOCKED，不允许改模型语义或样本制造成功。

## 8. 新窗口第一轮具体检查单

1. 读取两份交接、CURRENT_EXECUTION、EXECUTION_PLAN、05精度、06C128。
2. 检查git status/分支/HEAD，确认仅自己写该工作树；不要另起重复下载。
3. 只读核对.22的v3服务、本地代理隧道、per-file进度和错误/OOM。
4. 未下载完：保留健康任务，可做轻量源码语义审计，不开始全池转换/NPU启动。
5. 下载完：读取并复核全量身份审计，再进入A→B→C，不跳过前置。
6. 将新的实测状态、commit、runtime和证据路径写回CURRENT_EXECUTION；候选在held-out前提交。
7. 旧heartbeat保持PAUSED，直到用户明确要求新窗口恢复/新建定时推进。

推荐本地下载工具回归命令（不启动模型）：

```bash
python3 -m unittest discover -s migration_bridge/round9/tests
git diff --check
```

最后验证结果18tests PASS，只覆盖下载/准备工具，不是Qwen3.8设备资格。
