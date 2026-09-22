# Round9 下一轮实施方案：KT 主线核验与 Qwen3.8 异构推理归因

> **版本：** v1.0 ｜ **编制日期：** 2026-09-22  
> **执行对象：** 能访问本地 Round9 checkout、冻结构建、证据目录和受保护测试环境的开发者或本地模型。  
> **当前性质：** 待执行任务书。生成本文件不等于已经批准或启动新一轮物理卡测试。  
> **核心目标：** 在不破坏 v61 取消闭环和共享门禁的前提下，回到“KT-style Expert 异构执行 + Qwen3.8 Ascend 适配”主线，得到可单独验证的 Expert 组件、真实后端调用图和有边界的性能归因；最多准备或执行一个满足门槛的窄实验。

**架构定位：** SGLang 是请求、模型执行和状态管理的承载层；KT/kt-kernel 与本地适配负责 routed Expert 的 CPU/NPU 异构执行；Qwen 模型定义决定 GDN/QSA/PLE/GR 的语义；Ascend/Kunpeng 后端负责实际算子、数据布局和设备协同。不要把本轮改成通用 KV offload、HiCache 或新 Serving 框架研发。

**技术栈：** 沿用本地已冻结的 KT、KT 集成版 SGLang、torch/torch_npu、CANN、CPUInfer、ARM Expert bank 和 Ascend 算子组合。具体版本以本轮复核后的 manifest 为准，不安装“最新版本”替换基线。

**设计依据：** 用户上传的两份设计文档、用户粘贴的 v58/v61 执行报告，以及用户最新明确的项目主线。官方公开资料仅用于组件边界核验，不能替代本地源码与运行证据。

**阅读导航：** [本轮范围](#scope) · [组件边界](#architecture) · [安全门禁](#guardrails) · [基线复核](#baseline) · [调用图](#component-map) · [Expert 回放](#expert-replay) · [时间轴](#timeline) · [模型归因](#attribution) · [候选决策](#decision) · [PLE 实验](#ple-experiment) · [交付与验收](#deliverables) · [本地执行指令](#execution-brief) · [官方参考](#references)

---

<a id="scope"></a>

## 0. 阅读入口与本轮结论

### 0.1 一句话任务

**先回答“当前 KT/Qwen 到底调用了什么、哪些 Expert 真正在 Kunpeng 计算、Prefill/Decode 时间花在哪里”，再决定“是否值得改算子或移动一次 PLE 调度”。**

本轮不是再做一次泛泛的 profiler 建设，也不是重新证明 HTTP abort 能运行。它需要把已有观测能力用于三个此前没有充分回答的问题：

1. **KT 核心是否可独立核验？** 从真实模型抽取受限输入，绕开 HTTP 和 Scheduler，重放 Expert 计算并核验结果。
2. **Qwen 新架构实际使用什么后端？** 将 GDN、QSA、PLE、GR 和 MoE 对应到本地真实文件、符号、二进制与设备执行路径。
3. **哪个开销值得下一步开发？** 分开 Prefill 与 Decode、CPU 提交与实际执行、累计搬运与关键路径等待，输出证据和明确决策。

### 0.2 本轮建议范围

| 类别 | 本轮处理 | 交付含义 |
|---|---|---|
| KT Expert 主线 | 必做 | 真实组件回放、精度/布局/放置语义核验、CPU/NPU 路径归因 |
| Qwen 架构适配 | 必做 | GDN/QSA/PLE/GR 调用图、状态依赖和后端覆盖表 |
| 有界完整模型剖析 | 通过门禁后执行 | 至少一个窄配置的完成型 Prefill/Decode 观测，不以被取消请求代替正常服务基线 |
| PLE launch/wait 实验 | 条件分支，默认不启用 | 仅在合法窗口、实际等待和计时能力均成立时进入 |
| 新 CPU/NPU kernel | 本轮仅做立项判定 | 输出具体热点、算子契约、参考路径与必要性，不自动开始重写 |
| Context Tier / HiCache / FP8 KV | 后置 | 仅保留依赖边界说明，不实施 allocator 或缓存扩展 |
| Graph / MTP / TP2 / 多并发 / 长上下文资格 | 后置 | 不作为本轮验收项 |

### 0.3 执行顺序

```text
T0 基线、身份、授权与证据复核
  → T1 官方/本地组件及算子调用图
  → T2 真实 KT Expert 组件回放
  → T3 扩展已有有界观测并做离线验证
  → T4 窄配置完整模型归因
  → 决策：无修改 / PLE 窄实验 / 下一轮 kernel 立项 / 证据不足
  → T5 条件性 PLE 实验（可以不执行）
  → T6 回归、冻结、证据与环境收口
```

**成功不要求一定改计算代码。** 但不能只交一张“建议继续 profile”的表：必须交付实际调用路径、可复用组件验证入口、带来源的成本数据和可以执行的下一步决策。

---

## 1. 来源等级、已知事实与待核验事项

### 1.1 本地材料的使用规则

| 标识 | 材料 | 本文件如何使用 |
|---|---|---|
| [L-A] | `02_官方KT_SGLang对照与能力边界.md` | 已读取；保留其四层边界、Frequency 预算区分、Dynamic 事务与能力限制 |
| [L-B] | `04_功能补齐的设计重点-1.md` | 已读取；保留取消契约、PLE 因果约束、shadow identity、资源门禁和资格边界 |
| [R] | 用户在本次对话中粘贴的 v58/v61 执行结果 | 作为用户报告的当前状态；不是编制者独立复现实验的结果 |
| [I] | 用户最新说明的项目目标 | 本轮优先级依据：KT on Ascend + Kunpeng，适配 Qwen3.8，而非以 KV 卸载为主线 |
| [U1]—[U9] | 文末列出的官方仓库、模型卡和文档 | 2026-09-22 网页核对；未重新构建官方版本，也未将移动 main 当作固定依赖 |

[L-A]/[L-B] 内部引用的 `[L1–L3]`、`[U1–U12]` 来自其原文材料包。本次未拿到该材料包中的完整 `00` 文件，**不能声称逐项审计过这些原始证据**。本文件的 `[U1]—[U9]` 是自己的引用编号，与原文编号不互换。

以下文件只在 [R] 中被提及，编制者没有直接读取，必须由本地执行者在 T0 核验：

- `62_CANCEL_CLOSURE_AND_TIMELINE_RESULTS.md`
- `ROUND9_CANCEL_TIMELINE_STATUS.json`
- `CURRENT_EXECUTION.md`
- v61 的原始 JSON/JSONL、冻结构建 receipt 与源码树清单。

### 1.2 沿用的报告状态，不自动升级

```text
P0_CANCEL_CLOSURE_FULL_MODEL = FULL_MODEL_SMOKE
P1_TIMELINE_PHYSICAL_PROFILE = FULL_MODEL_PROFILE
P1_OPTIMIZATION_DECISION = NO_CHANGE_SELECTED

PRODUCT_QUALITY = PENDING
PERFORMANCE_QUALIFICATION = PENDING
```

这表示用户报告已有真实完整模型的取消/复用与时间轴证据，不表示已证明所有请求形状、所有故障交错、正式质量或正式性能。[R]

### 1.3 上一轮证据对本轮的约束

| 报告事实 | 本轮允许据此做什么 | 不能据此做什么 |
|---|---|---|
| v58/v61 均通过取消与同进程复用 | 作为稳定基线，保持相同取消契约 | 宣称任意时刻取消都已完成全面验证 |
| v61 控制消息发送至 Scheduler 接受为 343.934 s | 保留取消响应风险，分析当前 chunk 的实际成本 | 将它当作硬上限、单个 NPU kernel 时间或正常请求吞吐 |
| 2,043 个事件、报告丢失事件为 0 | 复用已有观测器和证据格式 | 认为已有事件覆盖了全部设备工作 |
| 673 次 shadow prepare | 核对该计数实际代表调用、slot 还是 Expert prepare | 直接把它解释成 673 次 DMA 或 673 层 |
| Host 观察累计 187,986,739,200 bytes | 查清字节口径、请求归属与搬运来源 | 宣称 HBM 同时驻留 188 GB，或已经证明 DMA 带宽瓶颈 |
| 完整 shadow identity 重复为 0 | 不默认实施完整 shadow 复用 | 排除一切部分 Expert 交集或未来负载复用 |
| PLE gather 后存在独立窗口，未观察到 H2D/Expert 重叠 | 保留一个窄 PLE 调度候选 | 声称已证明可加速，或立即增加 ring 深度 |
| v58 有训练 NPU 进程，v61 测试时没有 | 区分共享活跃度，分别标记证据 | 把两次数据拼成同等干扰条件下的性能对比 |

**特别说明：** HTTP 到 Scheduler 的长等待在本轮作为已知服务风险保留。取消线性化点后的安全性与用户可接受的停止延迟是两个不同验收维度；本轮不新增抢占，也不承诺秒级取消。

---

<a id="architecture"></a>

## 2. 官方 KT/SGLang 核对后的架构边界

### 2.1 可确认的公开事实

官方 KT 主仓库将原一体化代码放入 `archive/`，当前从 `kt-kernel` 源码树组织推理与微调能力。其推理介绍包括 CPU 优化 MoE、NUMA 感知执行和异构 Expert placement。[U1]

KT-Kernel 文档同时提供独立 Python API 与 SGLang 集成入口；所述集成明确使用 `kvcache-ai` 的 SGLang fork / `sglang-kt`，不能把 upstream `sglang` 包视为可无条件替换的相同依赖。[U2]

官方已有 Ascend + CPU Expert offload 教程，也已有 ARM/KML 源码构建说明。因此，本地价值应按**新增后端能力、具体精度/布局、Qwen 兼容性和经验证的系统行为**描述，不能概括成“官方只有 x86，我们首次支持 ARM/Ascend”。官方 DeepSeek-V4 的案例也不能自动证明本地 Qwen 配置可用。[U2][U3]

### 2.2 复用什么、适配什么

| 组件 | 可复用的部分 | 本轮必须自己核验的边界 |
|---|---|---|
| kt-kernel / CPUInfer | 已有 Expert kernel、权重加载、线程/NUMA 与提交接口 | 本地 ARM bank 格式、实际加载库、输入输出布局、异步所有权 |
| KT 的 SGLang 集成 | MoE 接入方式、placement 与 Prefill 协同思路 | fork 版本、本地 adapter、request/placement/allocation epoch 契约 |
| upstream SGLang | 模型组织、Scheduler、KV pool、服务与观测设施 | 本地 patch 与 upstream 差异；不在本轮替换整个运行时 |
| Qwen 模型实现 | GDN/QSA、N-gram embedding、GR、MoE 的语义 | checkpoint/config 是否匹配、实际运行了哪个分支和后端 |
| Ascend 算子与后端 | 已安装且验证过的 torch_npu/CANN/扩展算子 | dtype、shape、layout、stream/event 与错误路径是否支持 |
| HiCache | 未来可评估原生集成，而不是重写通用 prefix tree | 模型混合状态、分配器、跨请求共享、后端与版本契约 |

**“独立组件”在本轮的含义：** 可以让 KT Expert 通过一个受限回放入口脱离 HTTP 进行验证；不是要求复制一份完整模型运行器，也不是把所有 SGLang 内部组件拆成独立库。

**ABI 限制：** 官方 Python API 示例中的 CUDA stream 参数不是本地 NPU stream 的通用 ABI。执行者必须沿真实桥接代码核对，禁止把 NPU handle 强行代入 CUDA 参数“试试看”。[U2]

### 2.3 需要收紧的 HiCache 前提

SGLang HiCache 是围绕 prefix 缓存与多层存储组织的系统，不只是单次 page memcpy。[U6]

当前核对到的 Ascend 支持表虽然列出了 hierarchical cache，但同时注明 **mamba cache 不受支持**。Qwen3.8 roadmap 中“HiCache 已验证”的记录没有在该条目中完整限定 Ascend、GDN/PLE、量化和本地 patch 的组合。因此：**不能由两个一般性支持声明推导“Round9 的混合状态现在直接打开即可”。** [U7][U8]

此外，“可缓存更多历史 prefix”与“单个活跃请求能把所有 attention 所需状态移出 HBM 后持续计算”不是同一个合同。未来真做长上下文时也要分别核验，不能认为启用 HiCache 就自动解决活跃工作集容量。

本轮据此采取：**HiCache 不实施、不作为 KT 核心验收阻塞项。**

### 2.4 对 Qwen3.8 名称与架构的处理

官方 `Qwen3.8-Flash-Next` 模型卡公开了 Gated DeltaNet、Qwen Sparse Attention、Gated Residual、N-gram embedding 与 MoE；其中语言模型结构为 48 层，按 3 个 GDN 层与 1 个 QSA 层交替分组。[U9]

这只是公开目标的参照。**本地必须先核对精确模型 ID、revision、config 与 checkpoint。** 不能把所有名为 Qwen3.8 的模型都当成同一结构，也不能用官方模型卡替换本地实际配置。

PLE 沿用 [L-A]/[L-B] 的命名，指本项目接入的 N-gram embedding 路径；其具体 row 构造、历史长度、注入位置以模型定义和本地源码为准，不在本文件中推测 PLE 缩写展开或算法公式。

---

<a id="guardrails"></a>

## 3. 全局约束：所有任务共同遵守

### 3.1 不扩大的功能范围

本轮不做：通用 KV allocator 重写、BF16 Context Tier 新接线、FP8 KV 权威存储、HiCache/prefix 共享、16K/64K/128K 长上下文资格、Graph、MTP、TP2/HCCL、多请求并发、多模态、外部网络服务和通用 Scheduler 重构。

本轮也不做：全量重建 bank、全模型双份高精度驻留、为了对照强行 all-NPU、降低启动门槛、为了统计流量而反复冷启动或清理全局缓存。

已有能力保持原冻结设置。特别是 PLE Async 默认关闭，prefix-cache 默认关闭；新增开关必须显式、默认关闭且不影响未启用路径。[L-B][R]

### 3.2 共享物理卡保护：原值保留

| 门禁项 | 上限或下限 |
|---|---:|
| 训练任务 HBM 上限 | 15,360 MiB |
| Round9 allocator 上限 | 37,376 MiB |
| Round9 owned process 上限 | 38,400 MiB |
| 整卡 HBM 绝对上限 | 56,832 MiB |
| 最低保留空间 | 8,704 MiB |
| MXFP4 启动 Host 可用内存门槛 | 48 GiB / NUMA-node，不降低 |

上述数字沿用 [R]，不是本方案重新测出的安全值。**这些约束需要同时成立。** allocator 用量、进程用量、整卡用量不是可直接相加的互斥项，不能重复计账，也不能拿单项未超限代替所有门禁通过。

训练上限与 Round9 进程上限相加为 53,760 MiB，与整卡绝对上限之间仅剩 3,072 MiB；该差额不能被默认当作可增加 Round9 配额的余量，驱动/上下文等开销仍须由整卡监控覆盖。

### 3.3 身份边界与禁止操作

继续采用以下身份逻辑，不恢复“固定训练 PID 白名单”：

```text
明确容器名 /mzy-train
  → 从已批准共享凭据核对完整 64-hex 容器 ID
  → /proc/<pid>/cgroup 精确匹配
  → 该容器轮换子进程合计受同一训练预算约束
```

- 新容器 ID、容器外进程、不可确认身份或监控失效：fail-closed，停止或不启动 Round9 自有任务。
- 不因为容器名相同就自动批准重建后的容器；若报告未保存原 ID，先恢复证据链，不擅自建立新信任边界。
- 仅允许终止精确确认属于本轮的 Round9 进程/容器；禁止全机 `pkill`、停止训练、重启卡、重启驱动或修改共享训练配置。
- 不进入 `/mzy-train` 安装依赖，不修改其文件、挂载、镜像、环境变量或调度配置。
- 物理卡 15 与容器逻辑设备号必须核对，不将容器内 `0` 默认解释为物理卡 0。
- CPU 核、线程池与 NUMA 资源沿用已批准配置；没有 CPU/Host 资源余量证据时不做并行 microbenchmark 或多后端同时加载。
- 不执行全局 `drop_caches`。`POSIX_FADV_DONTNEED` 仅可沿用已有批准路径，限定 KT 自有文件并记录范围；不能新建宽泛清理器。

### 3.4 代码与构建边界

- 不修改 v61 冻结构建；在经确认的独立工作区完成本轮改动。
- 不自动 rebase、升级依赖、替换 SGLang fork、安装 moving tag 镜像或重新签发设备资格。
- 默认只新增组件回放、受限观测、分析、测试和文档；源文件改动必须进入显式 patch 清单。
- `layerwise.py`、`dynamic_adapter.py`、`ple_async.py` 在 T0—T4 原则上继续保持资格基线字节一致。T5 若确需改 PLE 调用位置，应只改已经定位的调用侧，不能顺带重写 ring 或复制算法。
- 不 push、不合并 main、不打 tag，不恢复旧 Round7 定时任务。
- `2026-09-20-04-PAUSED.md` 不修改、不暂存；不使用 `git add .` 或 `git add -A` 将未知文件卷入提交。

---

<a id="baseline"></a>

## 4. 任务 T0：复核基线、授权与执行环境

**目的：** 确认接手的是报告中的系统，而不是名字相似的目录、容器或依赖集合。

### 4.1 报告中的基线标识

```text
冻结构建目录：
/mnt/data1/kt-round9/framework-dev-20260921-p1-timeline-v6

本地检查点提交：
57de93dfb round9: close prefill cancellation and profile working set

build receipt SHA256：
0a00966bcd9f96dca6b39e87474ce39fd902d65de46ce03c65c8855fdab250d5

source tree SHA256：
8f68fb1157353c0e95c2e52ad93c8a2136d1d0a7dce8867324a3d17f0d6f2099
```

`source tree SHA256` 不是 Git commit SHA。必须复用原 receipt 的文件范围、排序、路径和归一化算法；禁止自行换一个目录散列算法，再声称“哈希不匹配/已匹配”。

### 4.2 执行步骤

- [ ] 读取 `CURRENT_EXECUTION.md`、`62_CANCEL_CLOSURE_AND_TIMELINE_RESULTS.md` 和状态 JSON，确定真实 checkout、证据根目录、构建入口及受保护文件。
- [ ] 将短提交 `57de93dfb` 解析为完整 commit，记录 tracked diff、untracked 文件清单和 submodule 状态。
- [ ] 按原生成规则核验 receipt、源码树与关键证据哈希；分别记录 `MATCH`、`MISMATCH`、`NOT_FOUND`，不能用“文件存在”代替内容核验。
- [ ] 记录本地实际的 KT、KT-SGLang fork、upstream 基准、模型 revision、bank manifest、torch/torch_npu/CANN、扩展库路径与哈希、镜像 digest。
- [ ] 恢复共享凭据中的训练完整容器 ID，核对物理设备映射、预算、Host/NUMA 启动条件、Round9 当前残留进程。
- [ ] 建立独立工作区及本轮证据目录，不覆盖冻结产物和既有报告。
- [ ] 读取原测试清单，确认依赖测试仍在冻结 Torch 环境运行；不为方便而向宿主或训练容器补装 Torch/NumPy。
- [ ] 形成第一版 `BASELINE_AND_SCOPE.md` 与机器可读锁定清单。

### 4.3 必须记录但不猜测的字段

`model_id`、`model_revision`、实际层数/层类型、CPU precision、NPU precision、placement profile、mask 哈希、CPU 线程与 NUMA 绑定、scheduler chunk、Expert microbatch、shadow 容量、PLE 模式、KV dtype、设备后端以及超时/日志预算。

缺失字段写成 `UNKNOWN` 并说明可如何查证。**身份、数值参考、资源限额、运行命令等执行前置字段为 UNKNOWN 时，相应物理测试不得启动。** 非阻塞的公开上游信息可以保留未知，不要求为了填满表格升级依赖。

### 4.4 验收与阻塞条件

**通过：** 基线可定位、关键哈希可追溯、共享门禁可确认、执行路径已锁定。

**停止物理执行：** 哈希异常未解释、训练身份不可恢复、出现未知外部 NPU 进程、资源契约不明、依赖缺失需要动宿主/训练环境。

遇到阻塞仍应交付只读审计结果，状态为 `BLOCKED_WITH_EVIDENCE`，而不是反复重建环境碰运气。

---

<a id="component-map"></a>

## 5. 任务 T1：建立 KT/Qwen/Ascend 的真实组件图

**目的：** 把“我们在做 KT 还是 SGLang”的问题落实为源码、接口和运行路径，而不只是一张概念图。

### 5.1 官方与本地分别锁定

建立四份记录：KT 官方参照、KT-SGLang 集成参照、upstream SGLang 参照、本地实际构建。每份包含来源地址、commit/revision、涉及文件、阅读日期和验证等级。

如只能访问网页，没有固定 SHA，写明 `WEB_REFERENCE_ONLY`。**不将阅读官方 main 变成替换本地冻结依赖的操作。**

官方 Expert Scheduling 与 Native Precision 文档分别作为 placement 和 Dual Prefill 的语义参考。文档与源码的阈值边界、后端限制存在差异时，记录差异，并以选定版本源码确定本地实际执行契约。[U4][U5]

### 5.2 本地必须定位的调用链

```text
请求入口 → Scheduler / Model Runner
  → 实际 Qwen 模型类 / 层定义
    → GDN 或 QSA（依层类型）
    → Router / Shared Expert / Routed Experts
       → KTEP / 本地 adapter
         → Kunpeng CPUInfer 与 Expert kernel
         → Ascend resident Expert / 临时 shadow Expert
       → 合并输出 / GR 等后续路径
    → PLE 的真实注入点与历史消费者
```

上图只表示核验对象，不规定 PLE/GR 的精确执行先后；先后关系由模型源码输出。

### 5.3 每个模块的契约记录

| 字段 | 要回答的问题 |
|---|---|
| `source_file` / `symbol` | 真正执行的文件、类、方法是什么？ |
| `loaded_binary` / hash | 是否调用扩展库？实际加载的是哪个二进制？ |
| `execution_phase` | Prefill、Decode，还是两者使用不同分支？ |
| `input_output_contract` | shape、dtype、layout、device、stride 约束是什么？ |
| `state_contract` | 读写哪些持久状态，归属哪个 request/allocation epoch？ |
| `dispatch_condition` | 根据长度、精度、设备、flag 如何选后端？ |
| `async_contract` | 提交、完成、可消费、可回收分别在哪里？ |
| `reference_contract` | 正确性 oracle 和既定误差规则来自哪里？ |
| `observed_provider` | 实际跑到 CPU kernel、NPU kernel、Torch fallback 还是 reference？ |
| `evidence` | 源码位置、测试、trace 事件或算子名，不能只写“支持” |

### 5.4 KT 核心核验重点

**CPU Expert：** BF16/Q8_0/MXFP4 各自的 bank、packing、解量化、GEMM 和聚合路径；C128 与 160 分区按本地源码定义，不能擅自把它们解释成 CPU 核数或模型层数。

**Placement：** 确认实际 profile 属于 `frequency_per_layer` 还是 `frequency_global_budget`，记录 resident 总量、各层数量、统计来源与 tie-break。Q64/Q32 的含义以 manifest 为准，不把 Q 当成量化 bit 数。[L-A]

**Dynamic：** 确认 request-boundary / Prefill-End 的统计范围、复制、publication 和 generation 校验；本轮审计语义，不改预算算法，也不把它变成吞吐 sweep。

**Prefill：** 分开 scheduler token chunk、Expert compute microbatch 和 shadow 生命周期；确认触发 Hybrid/Layerwise 的实际条件及 32-token 限制所在层级。

**数据移动：** 区分输入/输出 activation 搬运、临时 Expert 权重搬运、resident promotion 和布局变换，不能把所有 bytes 归为一种“KT offload”。

### 5.5 Qwen 路径核验重点

| 模块 | 本轮必须明确 |
|---|---|
| GDN | Prefill 的分块计算与 Decode 的递归更新分别调用什么；卷积/递归历史如何维护 |
| QSA | indexer、block 选择、KV 读写、attention 计算分别在哪里；短输入是否绕过真正稀疏路径 |
| PLE | row ID 构造、CPU gather、源表分片、H2D、consumer wait 的实际位置及 history 边界 |
| GR | 残差门控/分支处理是否来自原模型路径；不能在归因时无说明地全部漏进 Other |
| MoE Router / Shared Expert | Router、shared Expert 与 routed Expert 是否被分开计时和验证 |

**分支覆盖限制：** 短 prompt 下的 QSA 可能不能代表长上下文稀疏选择成本；源码上存在一个 sparse kernel 不等于这轮请求执行过它。对应路径必须标 `SOURCE_AUDITED_NOT_EXERCISED`，不为覆盖它而擅自扩大到 16K。

### 5.6 交付与验收

交付 `COMPONENT_AND_KERNEL_MAP.md`，附机器可读映射。每条主路径都必须能追到实际符号与后端；未定位项明确 `UNRESOLVED`。

这里不提前写假的文件路径或函数接口。**只有映射完成后，才生成本轮允许修改的精确文件清单。** 未解析到真实路径，不允许凭经验编造 `free_page()`、`forward_npu()` 或任意 kernel 名称。

---

<a id="expert-replay"></a>

## 6. 任务 T2：建立可脱离 HTTP 的真实 KT Expert 回放

**目的：** 给项目形成一个“真正属于 KT 核心”的长期验证入口，不让每次核验 Expert 都依赖昂贵的完整 HTTP 请求。

### 6.1 范围与形态

优先复用现有 CPUInfer/Expert 测试 harness。仅在缺少入口时添加一个薄回放层，不复制 Router、模型层、bank loader 或 Scheduler。

回放数据来自既有正确性 fixture，或 T4 中受限捕获的真实层输入。T2 可以先完成 harness 与已有 fixture 验证，再用 T4 的真实输入补验，不能因此提前无预算抓取全模型激活。

一个 replay bundle 至少包含：

```text
模型与层身份
bank/source/precision/layout 身份
输入 hidden states（受限大小）
top-k expert IDs 与 routing weights
resident mask / logical-to-physical map 身份
需要的 Expert 权重引用或有限切片
参考输出及参考实现版本
数值比较协议
```

禁止将整个 checkpoint 或全部层 activation 复制成测试附件；数据与输出仅写入本地批准目录，不上传外部服务。

### 6.2 最小用例集合

| 用例 | 目的 | 资源控制 |
|---|---|---|
| Decode 型小输入 | 核验少 token 时 CPU Expert 的真实提交、计算、合并 | 使用一个已识别层与受限 Expert 集合 |
| Prefill 型 microbatch | 核验现有工作区边界以内的批量路径 | 不超过冻结支持的 microbatch 上限 |
| CPU 分支非空、NPU 分支为空 | 验证分支归属与空分支处理 | 组件级构造，不把完整模型切到全 CPU |
| CPU/NPU 混合路由 | 验证 routing、map 和结果合并 | 只使用已批准 resident/shadow 容量 |
| NPU 分支非空、CPU 分支为空 | 验证对称边界 | 组件可容纳才执行，不申请 all-NPU 模型 |
| 分区/布局边界 fixture | 保留 ARM 分区与 ND/NZ 的已验证约束 | 复用现有边界样例，不重扫全部 160 分区 |

不同后端不要求本轮做全量笛卡尔积。优先覆盖 v61 实际使用的主后端；其他格式复用已有合格证据，仅在存在差异或缺口时做有限补测。

### 6.3 正确性判据

- 量化 kernel 的参考应使用**相同量化 bank**的可信解码/计算路径，避免把量化误差与 kernel 错误混在一起。
- 原始 BF16 与低 bit bank 的质量差异另列，不要求它们 bit-exact，也不把“与原模型不逐位一致”自动归为实现错误。
- 输入 dtype、累积精度、反量化 scale、activation、routing 权重和输出聚合顺序必须记录。
- 沿用已资格化的 `atol/rtol` 或误差协议，不能看到失败后临时放宽。没有协议时，先建立独立参考和误差分析，不给优化候选 PASS。
- 对不改算术的身份/复制检查使用 exact；对存在浮点路径差异的计算使用已冻结比较规则。
- 随机种子固定不等于所有底层 kernel 都确定性执行。必要时先量出 reference 自身重复波动，报告首个分歧点。

### 6.4 如何计时才有价值

单独记录加载/prepare、CPU 排队、CPU worker 执行、NPU 计算、结果等待和合并；不要把 `CPUInfer.submit()` 的 Python 调用耗时当成 Expert 计算耗时。

组件测试主要用来定位实现和相对成本。小 Expert 集合可能更容易命中 CPU cache，因此**不能把组件 tokens/s 外推成完整模型吞吐**。CPU 工作集、线程、NUMA、实际路由分布和 cache 状态均写入结果。

### 6.5 最小测试顺序

- [ ] 对新增回放/比较/身份校验逻辑先写失败测试。
- [ ] 验证真实 fixture 在现有基线通过，错误 bank ID、布局或层身份会被拒绝。
- [ ] 实现最小薄封装，保持原 kernel 和 loader 不变。
- [ ] 在冻结依赖环境运行测试，必要 NPU 测试通过共享门禁后串行执行。
- [ ] 用至少一个实际模型层的有限输入完成回放；只有合成输入时标为 `SYNTHETIC_ONLY`。

**验收：** 至少一个主后端的真实层输入可重放、输出满足冻结 reference、调用与资源边界可解释。即使最终不优化，也应留下可继续用于 Kunpeng kernel 开发的验证工具。

---

<a id="timeline"></a>

## 7. 任务 T3：扩展已有时间轴，而不是新建 profiler 框架

**目的：** 将 v61 的生命周期/搬运观察补足为 KT 与 Qwen 模块成本归因；只新增缺失的观测点和离线分析。

### 7.1 保留上一轮已经修好的观测约束

[R] 记录了两个真实问题，本轮必须直接继承其修复经验：

- `LayerwiseTransport.preallocate()` 完成后，才绑定依赖 `backend.adapter` / `backend.method` 的观察器。
- 真实 `_CopyOnlyAdapter` 是只读 slotted facade；不能重新使用 `adapter.copy_experts = wrapper` 的实例方法替换。
- 使用既有只读 forwarding proxy，保持返回值、异常和调用次数；观察器不能触发第二次实际复制或额外同步。
- disabled 路径不产生额外持久分配；enabled 路径的有界缓冲计入原有 Host/设备预算。

### 7.2 需要增加的最小事件类别

| 类别 | 至少记录 |
|---|---|
| 请求与 batch | request ID/epoch、batch ID、Prefill/Decode、真实 token span、开始/结束/取消 |
| CPU Expert | layer、worker/task ID、enqueue/start/end、token-expert assignment 数量、bank 身份 |
| NPU Expert | resident/shadow 分支、layer、执行后端、关联 device event 或 trace correlation ID |
| Expert 搬运 | source/destination、shadow identity、请求字节、实际复制字节口径、prepare 与 copy 分开 |
| GDN | Prefill/Decode 分支、层、状态 epoch、实际算子区间 |
| QSA | indexer、选择、KV 访问/更新、attention 区间；记录未执行分支 |
| PLE | row 身份、CPU gather 提交与完成、H2D 提交与完成、consumer wait、slot 释放 |
| GR/其他 | 已识别的 GR、Router、Shared Expert、embedding、LM head 等；剩余部分明确列 Unknown |
| 资源与环境 | owned HBM、allocator、整卡 HBM、Host 工作量、训练活跃度、日志丢弃计数 |

不要求在每个 token 上同步提取完整 tensor、计算大哈希或序列化大量数据。row/权重内容核验在受限正确性窗口完成；常规时间轴主要记录身份与大小。

### 7.3 三级计时证据，不能混用

**H：Host 事件。** 使用单调时钟记录提交、轮询、等待、状态转换。它可以证明控制顺序与 Host 阻塞，但不能直接证明 NPU kernel 的起止。

**D：Device 事件。** 在冻结 torch_npu/CANN/现有 profiler 能力下，记录实际 stream/event 或设备 trace。没有可用设备计时能力时，输出 `DEVICE_TIMING_UNAVAILABLE`，不得用 Host 函数返回时间伪装成设备完成时间。

**C：关联关系。** 将 Host 请求、CPU task、copy、device kernel 与 layer/phase 关联起来。Host/NPU/客户端的时钟域需要明确；未校准的不同时间轴不能直接叠加得出 overlap。

禁止为了方便计时，在每层或每个算子前后增加全局 synchronize。新增同步可能消灭本来想观察的 overlap，或者把旁路观测变成新的执行策略。优先使用事件记录、受限设备 trace 和窗口结束后的集中读取。

### 7.4 成本口径必须分开

报告至少包含三个视图，而不是只画一个总和等于 100% 的饼图：

1. **用户可见/请求 wall time：** 正常请求的 TTFT、token 间隔、请求结束时间，以及独立的取消链路时间。
2. **各模块工作量：** 调用次数、Host 执行/等待区间、CPU worker 区间、NPU kernel 区间、请求/实际搬运字节。
3. **关键路径与可隐藏等待：** 哪些等待真正阻塞后续 consumer；哪些计算/传输已经重叠；剩余未归因区间有多少。

例如 CPU wait 可能与 NPU Expert 执行同一段时间，父级 GDN span 可能包含子 kernel。**不能把父子 span、wait 与其等待的计算、并行 stream 时长直接相加后说它们占总请求的百分比。**

离线分析需要有 interval union、父子/关联标识和未覆盖区间处理。依赖信息不足时，只报告 inclusive duration，明确“不构成互斥占比/关键路径归因”。

### 7.5 搬运字节与带宽口径

给每个计数标明以下类别之一：

```text
COPY_REQUESTED_BYTES       API/逻辑层申请复制的字节
COPY_ACCOUNTED_BYTES       适配器按已知布局记账的复制字节
DEVICE_TRANSFER_BYTES      有设备侧依据的实际传输字节
HOST_READ_BYTES            可确认的 Host 读取字节
UNKNOWN                    当前工具无法观察
```

必须解释 ND/NZ padding、重复调用、view/alias、prepare 空操作和布局变换是否计入。无法确认实际 DMA 流量时，保留 Host 记账名称，不升级成 `DEVICE_TRANSFER_BYTES`。

有效带宽只有在字节与对应传输时间来自同一测量范围时才能计算。不能用 187.99 GB 除以取消等待 343.934 s 推算 Expert 链路带宽。

CPU Expert 的 DRAM 读取不经过 H2D 队列。测不到内存控制器带宽就写 `UNKNOWN`，不以 CPU 利用率、H2D bytes 或 outstanding credit 代替。[L-A][L-B]

### 7.6 有界性与扰动控制

- 沿用已有事件容量、日志限制与预算；T0 将实际数值写入 manifest。
- 如新增事件导致上限不够，缩小采样窗口或层范围，不自动扩大 ring、开启无限日志或丢事件后仍报告完整。
- Host 事件丢失和设备 trace 丢失分别记录。工具不提供设备丢失计数时写 `UNKNOWN`，不能写 0。
- 设备 profiler 只覆盖 Round9 自有进程与批准窗口，不进行全设备训练进程追踪。
- 同一 workload 至少保留 profiler OFF 的结果。重 profiler 请求用于解释路径，不作为正式速度结果。
- 若观测本身显著改变控制顺序或内存，先修观测方式，不在被扰动路径上做优化结论。

### 7.7 离线测试必须覆盖

- [ ] slotted adapter、预分配前未就绪、重复绑定与解绑。
- [ ] proxy 的返回值、异常、复制次数与原对象一致。
- [ ] begin/end 缺失、缓冲溢出、取消中断与带原因的截断。
- [ ] 嵌套 span 不重复计时；两个相交区间的 union 不重复累计。
- [ ] Host/device 时钟未关联时，分析器拒绝输出确定 overlap。
- [ ] 字节单位、时间单位、计数器累计值与请求增量不混用。
- [ ] 把未知字段保留为未知，不能以 0 填空来生成完整图表。

**验收：** 可以对既有 trace 做可信离线重放；新增字段有实际来源；覆盖不完整时能正确降级为 `PARTIAL_HOST_ONLY` 或 `PARTIAL_ATTRIBUTION`。

---

<a id="attribution"></a>

## 8. 任务 T4：一个主配置的完整模型归因

**目的：** 获得正常完成的 Prefill 与 Decode 成本，而不是继续以取消请求和累计 bytes 作为性能替代品。

### 8.1 配置只选一个

主配置从 v61 冻结 manifest 选择，固定模型、bank、CPU/NPU dtype、placement/mask、线程/NUMA、Prefill 模式、chunk、microbatch、shadow、KV dtype 与 PLE 默认设置。

**本轮不同时 sweep Q32/Q64、BF16/Q8_0/MXFP4、Static/Dynamic、Hybrid/Layerwise。** T1 对这些能力做语义表，T4 只对实际主配置做有界观测。

若 Dynamic 已在基线启用，记录每次运行的初始 placement 与更新身份；A/B 比较不能让前一次请求留下的 placement 随意污染下一次。只能使用已验证的重置/重建方式，不能为了公平新增未经验证的 reset 函数。

### 8.2 默认物理探测预算：需在本轮获准执行后使用

建议本轮主线预算如下，属于本方案建议值，不是上一轮已批准的额外资源：

| 项目 | 本轮默认上限 |
|---|---:|
| 新增完整模型 HTTP 请求 | 8 个，取消请求与 reuse 请求分别计数 |
| 其中重设备 profile 请求 | 3 个以内 |
| Round9 模型容器启动 | 2 次以内，优先一次启动完成 |
| 同时活跃模型请求 | 1 |
| 主执行配置 | 1 |
| 新增长上下文请求 | 0 |
| 自动失败重试 | 0；一次失败先分析，不滚动重跑 |

预热和失败请求也计入预算，不藏在“准备工作”中。时间/日志/内存预算沿用已批准 watchdog 与冻结配置，由 T0 写明；若这些限制无法容纳一次必要测试，则交付阻塞证据，不自动加大超时。

### 8.3 建议 workload

| ID | 工作负载 | 建议请求数 | 回答的问题 |
|---|---|---:|---|
| W-BASE | 精确复用 v61 同进程成功短请求 fixture | 2 次非重 profile + 1 次 profile | 正常路径能否重现，基础 Prefill/Decode 调用与观测扰动 |
| W-CHUNK | 64 个实际输入 token，最多 4 个输出 token | 1 次非重 profile + 1 次 profile | 两个 32-token chunk 正常完成时的权重/状态成本 |
| W-CANCEL | 原 181-token 取消 fixture，随后同进程短请求 | 合计 2 个，涉及观测/集成改动时执行 | 取消 fence、drain、cleanup 与 reuse 不退化 |
| W-DECODE | 仅当 W-BASE 的 Decode 太短：输入不超过 32 token，输出上限 8 | 最多 1 个，使用剩余预算 | 给连续 Decode 分支增加受限覆盖 |

W-BASE 的实际长度、采样设置和输入来自本地证据，不在这里编造。W-CHUNK/W-DECODE 是本轮建议的新探测，不应被写成“过去已经验证过”。

输入长度以 tokenizer 与实际发送 token IDs 为准，包含模板影响；不能用字符数代替 token 数，不能把截断后输入仍标成完整原 prompt。提前 EOS 时报告实际输出数，不强行改模型语义凑满 Decode 步数。

若主配置不是实际 32-token scheduler chunk，不能套用表中“两个 chunk”的解释；T1 必须记录真实 scheduler/microbatch 区别，再调整 W-CHUNK 的实验描述，仍不扩大默认长度预算。

### 8.4 执行流程

- [ ] 检查共享身份与所有预算；记录训练容器是否有活跃 NPU 子进程及 Host 活跃度。
- [ ] 运行 W-BASE 的非重 profile 请求，确认正确性和资源回收；不通过则不继续增加 workload。
- [ ] 在冻结采样窗口执行 W-BASE profile，关联 CPU worker 与 NPU 事件。
- [ ] 完成正常 W-CHUNK 对照，记录每个 chunk 的 prepare、copy、compute、state dependency。
- [ ] 仅在必要时使用 W-DECODE 补足 Decode 覆盖，不把它当作长稳态性能结果。
- [ ] 对涉及运行路径的新增观测完成 W-CANCEL 与 reuse，比较语义，不要求复现相同 343.934 s。
- [ ] 结束窗口、导出证据、关闭设备 profiler；避免后台持续捕获。
- [ ] 使用 T2 的组件回放与本次真实输入相互核对后端路径。

### 8.5 正常请求指标定义

**TTFT：** 从客户端发出请求到收到第一个有效生成 token 的时间。首个 HTTP 字节、SSE metadata 或角色字段不是首 token。客户端和服务端指标分别命名，不能混用时钟。

**Prefill wall time：** 服务端对实际输入 span 开始计算到其 Prefill 完成的时间；排队、tokenization 与采样是否包含要明确。

**TPOT：** 对收到 N 个生成 token 的正常请求，N > 1 时可报告 `(最后 token 时间 − 首 token 时间) / (N − 1)`；另记录各个 token 间隔。少量 token 仅属于 probe，不宣称稳态 TPOT 或可靠 p95。

**取消指标：** 独立记录 ACK、Scheduler accept、CANCEL_SEEN、fence、retirement、cleanup。取消请求不与正常请求混算 tokens/s。

### 8.6 归因报告必须回答

1. CPU routed Expert 是否真的执行？Prefill/Decode 分别执行多少，哪些路由在 NPU resident/shadow 分支？
2. CPU Expert 主要时间位于排队、packing/解量化、GEMM 还是结果合并？无法拆分的部分是哪一段？
3. Expert 权重搬运发生在什么层/跨度，和 scheduler chunk 的关系是什么？
4. GDN、QSA、PLE、GR 各自的实际 provider 是什么？有没有 reference/fallback 被误当成优化 kernel？
5. PLE 的 Host gather 完成与 H2D 完成分别在何时？consumer 到达时是否真的等待？
6. H2D 与 CPU compute、NPU compute 是否分别重叠？是只没看到与 Expert compute 重叠，还是也检查过其他合法计算？
7. 这次 workload 覆盖了哪些模型分支，哪些因为短输入未覆盖？
8. HBM 压力主要来自哪里？bank、resident、shadow、workspace、KV、状态及 profiler 分配是否分清？
9. 343.934 s 对应的旧 chunk 等待，在新观测中有没有可定位的同类长耗时；证据能解释到哪一层？
10. 对下一轮 kernel 或调度开发，当前最有根据的一个候选是什么？

### 8.7 共享干扰与结果命名

v58 的训练 NPU 进程峰值只有 1,358 MiB，证明真实共卡发生过，但不是 15,360 MiB 训练预算边缘压力测试。v61 无活跃训练 NPU 子进程也不等于独占 Host/DRAM。[R]

本轮每个请求标记训练 NPU 与 Host 活跃度，报告为**共享环境下的条件性观测**。不能为了制造安静基准暂停训练，也不能将训练活动明显不同的样本合并成可靠 A/B 提升。

如果只完成有界短请求，最终状态最多是 `FULL_MODEL_BOUNDED_ATTRIBUTION`；设备时间缺失或关键依赖无法关联则为 `PARTIAL_ATTRIBUTION`。这些都不自动使 `PERFORMANCE_QUALIFICATION` 通过。

---

<a id="decision"></a>

## 9. 从归因到开发：唯一候选的选择规则

### 9.1 不预设“PLE 一定最值得优化”

[R] 给出的是 PLE 的合法独立窗口线索，不是完整模型瓶颈排序。因此本轮先完成 KT 与 Qwen 成本图，再按下表判断。

| 证据指向 | 本轮结论/动作 | 不做什么 |
|---|---|---|
| CPU Expert 关键路径占主导，有具体 packing/GEMM/NUMA 证据 | 输出下一轮 Kunpeng kernel/线程局部性立项卡 | 不顺便重写 GDN/QSA 或做 HiCache |
| GDN/QSA 的 fallback 或具体 NPU kernel 明显阻塞 | 输出对应 Ascend 算子立项卡，注明 Prefill/Decode shape | 不因模型名称新就重写全部算子 |
| PLE consumer 有暴露等待，且现有合法窗口能覆盖其部分传输 | 可进入 T5 单一 launch/wait 窄实验 | 不改 row 算法、不加 ring 深度、不偷看未来 token |
| Expert copy 关键路径明显，但没有安全复用关系 | 提交逐层/slot 流量证据和后续实验设计 | 不直接实施 changed-slot 或 next-layer staging |
| 没有可信设备计时或共享干扰不可控 | `INSUFFICIENT_EVIDENCE`，列出最小缺证 | 不填“预估占比”冒充实测 |
| 所有候选预期收益小或代价/风险更高 | `NO_CHANGE_SELECTED` | 不为交付代码而修改稳定计算路径 |

### 9.2 算子重写的立项门槛

本轮若决定“下一轮应写/迁移某个 kernel”，必须同时给出：

- 精确算子边界、输入输出 shape/dtype/layout、Prefill/Decode 分支与适用范围。
- 实际后端和加载来源；现有算子为何不能满足正确性、布局或性能要求。
- 可运行的独立 reference、既定误差协议、边界测试，以及失败后回退路径。
- 至少一份真实模型关联的热点证据，不能只用微基准快慢。
- 预期减少的是计算、搬运、launch 还是 consumer wait，以及新增 workspace 是否有预算。

若某路径仅占互斥关键路径比例 p，即使完全消除它，理想加速上限也只能按 `1 / (1 - p)` 估算；这只是推导上限，不是测得收益。若 p 不能从可靠互斥关键路径得到，就不要套公式产生虚假精确度。

### 9.3 新增代码的边界

T0—T4 允许新增回放、受限 instrumentation、分析和测试。**不允许自动追加一个 kernel 实现 sprint。** 热点立项卡应成为下轮输入，而非被本轮执行者自行扩大为多模块重写。

这不意味着项目永远只做分析：本轮交付的独立 Expert 回放、后端映射和数值 oracle，正是下一轮真正进行 ARM/NPU 算子开发的实验基础。

---

<a id="ple-experiment"></a>

## 10. 任务 T5：条件性的 PLE launch/wait 窄实验

**默认状态：不启用。** 只有本轮执行授权包含该分支，且下列门槛全部满足时才实施。否则交付候选说明，正常结束本轮主线。

### 10.1 进入门槛

- PLE 实际出现在主配置请求路径上，而不是仅在源码中存在。
- 已确认当前 token/span 的 row ID 合法，gather 与 H2D 的依赖可分开观察。
- consumer 确有暴露等待，且存在不依赖本次 PLE 数据的独立计算窗口。
- 现有两 slot、现有 transfer stream/event 与预算即可尝试，不需要增加常驻资源。
- 设备侧计时可用，能区分“更早 enqueue”与“真实传输重叠”。
- 已有 exact row reference、取消/drain/reuse 测试以及可回退的默认路径。

### 10.2 唯一允许的行为变化

**只调整已有合法 PLE 工作的 launch/wait 位置。** 不改 n-gram/row ID 算法、不改源表和 BF16 row 内容、不增加 slot、不新增推测预取、不扩大已批准 token span、不改 Expert placement、bank、QSA/GDN 数学语义。

Prefill 可以使用已知 prompt 的合法后续输入；Decode 只可在当前输入 token 已经确定后发起所需 lookup。尚未采样的未来 token 不属于当前正确性路径。[L-B]

若发现现有 stream 或函数封装让合法重叠根本无法形成，应终止该实验，不借此新增复杂异步框架。

### 10.3 三个基准身份

| 标识 | 含义 | 用途 |
|---|---|---|
| A | 当前默认安全路径，PLE Async OFF | 判断候选最终是否值得替代默认 |
| B | 当前已实现但默认关闭的 Async，原 launch/wait | 判断移动位置本身的影响 |
| C | Async + 仅移动合法 launch/wait | 实验候选 |

C 对 B 才能隔离“调度位置变化”；C 对 A 才能回答“是否比默认值得用”。不能把从 Sync 切到 Async、扩大预取和改 placement 的效果混在一起。

### 10.4 实验顺序与资源上限

先在 T2 风格的受限组件/调用片段上验证 row、event、slot reuse，再考虑完整模型。组件不通过，不消耗完整模型预算。

若分支获准执行，建议**额外最多 8 个短 HTTP 请求、最多一次额外模型容器启动**：其中覆盖 A/B/C 的功能对照、B/C 的受限设备时间轴，以及 C 的取消与同进程复用。预热和失败仍计入上限，输入不超过主线已经批准的短输入范围。

这些少量样本用于证明功能、依赖和 overlap，**不是正式性能资格的样本量**。不为了得到显著提升结果而继续加跑。若无法完成可信非 profile 对比，速度结论保留 `INCONCLUSIVE`。

### 10.5 验收与决策

必须分别输出以下四项：

1. **语义：** 同一模型/source/layer/history 下 row identity 与 row 内容 exact；下游输出满足冻结 reference 协议。
2. **时间轴：** 设备传输与合法独立计算存在实测交叠，不以更早 Host enqueue 代替。
3. **端到端：** profiler OFF 的短请求没有明确退化；样本不足或训练干扰不可比时写 `INCONCLUSIVE`，不宣布加速。
4. **生命周期：** 取消 fence 之后没有新增 prefetch；已提交任务可 drain；两 slot、ticket、consumer 和临时资源回到合法基线；reuse 通过。

可使用下列结论：

```text
NOT_ENTERED                    没有满足进入门槛
FUNCTIONAL_ONLY                正确但尚无设备重叠证据
OVERLAP_OBSERVED                有真实重叠，端到端收益仍待判断
NO_BENEFIT                     正确但未见值得保留的收益
REGRESSION_REJECTED             发生错误或明确退化，拒绝候选
CANDIDATE_ONLY_DEFAULT_OFF      有继续评估价值，默认仍关闭
```

**本轮不自动把 PLE Async 改成默认开启。** 重叠存在也不代表总请求更快：额外排队、Host 资源争用和同步开销可能抵消收益。

---

## 11. 回归重点：五类最容易漏掉的问题

| 风险 | 所属任务 | 必须验证的行为 |
|---|---|---|
| 预分配后对象才出现，或真实 adapter 只读 | T3 | 延后绑定、proxy forwarding，禁用路径无变化；不能只用可变假对象测试 |
| 相同 layer/expert ID 但模型、bank、layout 或 epoch 不同 | T1/T2/T5 | identity 不匹配必须拒绝复用，不能拿旧 shadow/row 当新数据 |
| cancel 发生在 gather/copy/consumer 之间 | T3/T5/T6 | fence 之后不提交新工作；在途工作安全排空；资源与结果提交正确 |
| 训练子进程轮换或新容器同名重建 | T0/T4/T6 | 原完整 ID 内轮换可识别，新 ID 或未知进程 fail-closed，仅停止 Round9 |
| Host/device span 嵌套、丢事件、单位或时钟混淆 | T3/T4 | 分析器拒绝虚假带宽/overlap/100% 占比，保留 UNKNOWN 与缺证 |

测试数量不作为目标。上一轮的 65 项重点回归、14 项 P1 测试与依赖容器测试可以作为入口，但必须读取实际测试名与环境，不能只复述历史通过数量。[R]

---

<a id="closeout"></a>

## 12. 任务 T6：回归、证据冻结与环境恢复

### 12.1 回归分层

**无设备层：** 源码定位、身份 key、manifest/schema、离线 trace 分析、proxy 生命周期、错误注入、资源记账。

**冻结依赖层：** 原来依赖 NumPy/Torch 的测试在批准的 Torch 环境运行；宿主缺依赖记为环境/收集错误，不包装成断言通过。

**组件设备层：** 主后端 Expert 回放、布局边界、实际异步事件与限定资源用例。

**完整模型层：** T4 的正常请求与必要的 cancel/reuse；T5 仅在执行过候选时补候选回归。

`compileall` 与 `git diff --check` 可以保留，但它们不能代替数值正确性、生命周期与真实模型证据。

### 12.2 差异审查

- [ ] 列出每个新增/修改文件、用途、调用点、默认行为、测试和回退方式。
- [ ] 核验默认关闭时，原执行序列和资源占用没有非预期变化。
- [ ] 列出与资格基线不再字节一致的文件；不能隐瞒改动，也不能未经授权“重签”资格。
- [ ] 分开“新文档/观测工具”和“真实计算/调度行为变化”，后者必须有独立说明。
- [ ] 本地提交只暂存显式文件列表；未知用户文件保持原样。

### 12.3 资源回收与退出

请求清理以各类 post-construction persistent baseline 为判据；不要求持久 ring、pool 或模型权重在进程内变成 0。[L-B]

结束本轮后，停止精确确认属于本轮的 Round9 测试容器/进程，确认没有本轮设备任务和资源所有权残留；训练容器保持 running 且不被修改。

不要求整卡一定回到 [R] 中约 2,878 MiB，因为训练任务可能自然恢复活跃。最终应报告**本轮 owned 进程已退出、训练身份/状态未变、整卡当前读数及外部活跃度**，而不是为了复现旧数值干预训练。

### 12.4 完成条件

本轮主线可收口的最低条件是：

- 基线、官方/本地边界与真实模型身份有清楚记录。
- 至少一个真实 KT Expert 组件回放成立，或明确的 blocker 已定位到接口/参考/资源层。
- Prefill/Decode 已有受限完整模型归因，或者设备权限/预算/计时不足导致的缺证被明确记录。
- 输出唯一后续候选或 `NO_CHANGE_SELECTED` / `INSUFFICIENT_EVIDENCE`，不模糊地罗列十几个方向。
- 默认行为、共享门禁和取消语义没有退化；环境与用户文件已核对恢复。

缺少完整模型归因时可以结束本轮工作，但总状态应为 `PARTIAL` 或 `BLOCKED_WITH_EVIDENCE`，**不能仅因文档写完就宣布技术目标通过。**

---

<a id="deliverables"></a>

## 13. 交付目录与最小机器可读记录

### 13.1 建议新增目录

以下是本方案定义的新交付路径，不是声称它们已存在于本地仓库。由 T0 在独立工作区内创建；已有同名目录时不得覆盖，改用新的本轮 run ID 子目录并在接手入口登记。

```text
artifacts/round9-kt-mainline/
  BASELINE_AND_SCOPE.md
  COMPONENT_AND_KERNEL_MAP.md
  KT_QWEN_ATTRIBUTION_REPORT.md
  NARROW_EXPERIMENT_DECISION.md
  NEXT_ROUND_STATUS.json
  baseline-lock.json
  source-map.json
  workload-manifest.json
  profile-budget.json
  patch-manifest.json
  evidence/
    run-manifest.json
    working-set-timeline.jsonl
    working-set-timeline-summary.json
    device-profile-index.json
    component-replay-results.json
    correctness-results.json
    monitor.jsonl
    test-results.json
  checksums.sha256
```

可以沿用现有成熟证据格式，只补必要字段；不要为了满足目录树重写全部工具。未执行的实验不生成伪造结果文件，在索引中记录 `NOT_RUN`。

### 13.2 每个 run 的共同身份

所有输出必须能关联到同一 `run_id`，至少包含：

| 字段组 | 内容 |
|---|---|
| 源码与构建 | commit、tracked diff hash、source-tree receipt、镜像 digest、实际依赖版本 |
| 模型与权重 | model/config/tokenizer revision、bank manifest hash、CPU/NPU precision、布局 |
| 调度身份 | placement strategy/budget、mask hash、request/placement/allocation epoch |
| 工作负载 | 实际 token IDs 或可重现引用及 hash、输入长度、实际输出数、采样设置 |
| 资源 | 设备映射、批准训练容器 ID、CPU/NUMA、各预算、运行前后 owned 与全卡读数 |
| 观测 | profiler 模式、时钟域、采样窗口、事件容量、丢失/截断、覆盖分支 |
| 结果 | reference 协议、通过/失败/阻塞、首个分歧、错误栈、证据文件与哈希 |

机密/敏感 prompt 不为方便直接写入公开文档；使用本地受控 fixture 与哈希，并保证本地执行者可复现。

### 13.3 状态 JSON 初始模板

下面是**计划初态**，不是本轮运行结果。所有新任务初始为 `NOT_RUN`；历史结果明确标注来源为用户报告。

```json
{
  "schema_version": 1,
  "plan_name": "ROUND9_KT_MAINLINE_ATTRIBUTION",
  "plan_date": "2026-09-22",
  "plan_state": "PROPOSED",
  "execution_authorization": "NOT_RECORDED",
  "baseline_commit_reported": "57de93dfb",
  "baseline_artifacts_verified": "NOT_RUN",
  "inherited_reported_status": {
    "provenance": "USER_PROVIDED_V58_V61_REPORT",
    "P0_CANCEL_CLOSURE_FULL_MODEL": "FULL_MODEL_SMOKE",
    "P1_TIMELINE_PHYSICAL_PROFILE": "FULL_MODEL_PROFILE",
    "P1_OPTIMIZATION_DECISION": "NO_CHANGE_SELECTED"
  },
  "current_round": {
    "BASELINE_LOCK": "NOT_RUN",
    "COMPONENT_AND_KERNEL_MAP": "NOT_RUN",
    "KT_REAL_COMPONENT_REPLAY": "NOT_RUN",
    "QWEN_BACKEND_COVERAGE": "NOT_RUN",
    "DEVICE_TIMING": "NOT_RUN",
    "PREFILL_DECODE_ATTRIBUTION": "NOT_RUN",
    "CANCEL_REUSE_REGRESSION": "NOT_RUN",
    "SHARED_GUARD_REGRESSION": "NOT_RUN",
    "ENVIRONMENT_CLEANUP": "NOT_RUN"
  },
  "decision": {
    "selected_branch": "NONE",
    "ple_experiment": "NOT_ENTERED",
    "ple_async_default": "OFF",
    "new_kernel_implementation": "OUT_OF_SCOPE",
    "context_tier_implementation": "OUT_OF_SCOPE"
  },
  "qualification": {
    "PRODUCT_QUALITY": "PENDING",
    "PERFORMANCE_QUALIFICATION": "PENDING"
  },
  "evidence_index": [],
  "blockers": []
}
```

阶段成功应逐项填证据，不用一个笼统的 `PASS` 覆盖所有字段。沿用历史证据而未重测时，写 `INHERITED_NOT_RERUN` 并引用原 evidence hash；不把继承结果说成本轮重新通过。

---

<a id="acceptance"></a>

## 14. 总验收表：什么可以 claim，什么仍不可以

| 交付/状态 | 最低证据 | 可以描述 | 不可以描述 |
|---|---|---|---|
| `BASELINE_LOCKED` | 本地 receipt、依赖、身份与资源凭据 | 接手与测试基线可追溯 | 已覆盖官方最新版本全部行为 |
| `COMPONENT_MAP_AUDITED` | 实际源码/符号/加载后端与调用条件 | KT/SGLang/Qwen/设备职责已定位 | 所有分支都在真实模型上执行过 |
| `REAL_COMPONENT_PASS` | 实际模型层输入、真实 bank、冻结 reference | 主 Expert 组件可独立验证 | 完整模型质量通过，所有精度均合格 |
| `FULL_MODEL_BOUNDED_ATTRIBUTION` | 正常完成请求、关联计时与覆盖说明 | 窄配置的 Prefill/Decode 成本有证据 | 正式性能资格、稳态服务或长上下文已完成 |
| `PARTIAL_ATTRIBUTION` | 可用部分、缺失层级、原因 | 已知开销与未知范围清楚 | 完整关键路径百分比或确定瓶颈排名 |
| `OVERLAP_OBSERVED` | 校准关联的设备区间与独立计算交叠 | 该窗口发生真实 overlap | 整体速度一定改善 |
| `CANDIDATE_ONLY_DEFAULT_OFF` | 正确性、依赖与受限对比 | 有值得后续资格化的候选 | 自动打开默认、可部署、优于官方 |
| `NO_CHANGE_SELECTED` | 逐候选证据与拒绝理由 | 本轮不需要增加执行复杂度 | 所有可能的优化都已穷尽 |
| `BLOCKED_WITH_EVIDENCE` | 确切阻塞点、最小复现或缺证清单 | 有可接手的失败/边界信息 | 技术目标已全部完成 |

### 14.1 三种基线必须分清

**正确性基线：** 同 bank、同 shape、同语义的可信 reference，用于发现实现错误。

**工程运行基线：** v61 冻结配置及本轮 profiler OFF 路径，用于判断新增代码是否破坏正常行为。

**官方性能基线：** 只有模型、精度、硬件、并行配置、workload、采样和测量口径足够可比时才能建立。官方 DeepSeek/多 GPU/x86 数据不是本地 Qwen/单 Ascend/Kunpeng 的公平直接对照。

本轮优先建立前两种。第三种没有合适条件就保留缺失，不强行加载全 NPU 模型，也不拿不同平台数字相除宣称领先。

### 14.2 本轮之后仍保留的资格状态

```text
PRODUCT_QUALITY = PENDING
PERFORMANCE_QUALIFICATION = PENDING
```

CMMLU、长时间稳定服务、尾延迟、共享边缘压力、16K+ 真实上下文、Graph/MTP/TP2 等应另行定义工作负载与资源授权，不能从几次真实模型 smoke 外推。

---

## 15. 与前两份文档的对应关系

本方案改变的是**本轮优先级**，不是否定原有设计约束。

| 原文事项 | 本轮处理 | 原因 |
|---|---|---|
| [L-A] 四个来源不能混为一个官方 | T1 细化成源码/后端/运行证据图 | 直接回答项目定位问题 |
| [L-A] CPU Expert 与 placement 是实际平台落地 | T2 优先形成真实组件回放 | 回到 KT 核心开发基础 |
| [L-A] Frequency 两类预算不同 | T1 核对 manifest，不改策略 | 防止对齐名称时破坏受控实验 |
| [L-B] P0 cancel 闭环 | 继承报告证据，保持回归 | 已有 full-model smoke，不再把它当主研发方向 |
| [L-B] PLE 只移动 launch/wait，不重写 ring | T5 唯一条件性行为实验 | 原约束继续有效，但不预设收益 |
| [L-B] Shadow 粒度与 identity | T1/T3/T4 明确观测 | 不凭累计 bytes 直接改搬运策略 |
| [L-B] BF16 Context Tier allocator 接入 | 本轮后置 | 它是 Context 平面的有效缺口，但不是当前 KT 主线的前置 |
| [L-B] FP8 KV | 本轮不实施 | reference 不等于真实容量收益；先不扩大 Context 范围 |
| [L-B] Joint Budget 策略 | 只观察现有竞争与等待 | 不把资源治理扩成新的带宽调度课题 |
| [L-B] 长上下文/高级功能 | 继续后置 | 先得到稳定窄配置与真实成本图 |

**项目主线保持为：Kunpeng Expert Engine + CPU/NPU Expert 执行协同 + Qwen3.8 Ascend 语义与必要算子优化。** SGLang 是承载这些工作的依赖和集成层；只有阻塞这些目标时才修改其内部接口。

---

<a id="execution-brief"></a>

## 16. 可直接交给本地执行模型的任务指令

> 从用户报告的 `57de93dfb` 和 v61 冻结构建接手，执行“Round9 KT 主线核验与 Qwen3.8 异构推理归因”。先读取 CURRENT_EXECUTION、62 号结果文档、状态 JSON 和原始证据，复核 receipt、源码树、依赖及共享凭据。未经本轮执行授权，不启动新物理测试。
>
> 本轮目标不是继续实现 KV offload/HiCache，而是建立真实 KT Expert 组件回放、GDN/QSA/PLE/GR 的源码与后端调用图，并在一个冻结主配置下获得正常完成的 Prefill/Decode 有界归因。优先复用已有 CPUInfer 测试入口、只读 forwarding proxy 和 bounded timeline，不重新构造 Serving/Profiler 框架。
>
> 严格保护物理卡 15 与 `/mzy-train`：使用批准的完整容器 ID 和 cgroup 边界，轮换子进程共享原训练预算；身份未知、预算异常或监控失效时 fail-closed，只处理 Round9 自有环境。训练 HBM 15,360 MiB、Round9 allocator 37,376 MiB、Round9 owned process 38,400 MiB、整卡绝对 56,832 MiB、最低余量 8,704 MiB 的门禁不放宽；MXFP4 的 48 GiB/NUMA-node 启动门槛不降低。不得修改训练、全局 drop_caches、重启设备或操作不属于本轮的进程。
>
> 按 T0→T1→T2→T3→T4→决策→T6 执行。任务进入物理阶段前要有已核验的配置和预算。主线最多 8 个新增完整模型请求、3 个重 profile 请求、2 次模型容器启动；预热与失败也计数。失败先定位，不自动循环重跑，不扩大到长上下文或多配置 sweep。
>
> 每个结论分清用户报告、源码核验、组件运行和完整模型证据。Host enqueue 不等于 device 完成，累计 bytes 不等于真实 DMA 或 HBM 驻留；嵌套、重叠和等待区间不能直接相加冒充关键路径百分比。短输入未覆盖的 QSA 分支如实标记，测不到的内存带宽或设备时间写 UNKNOWN。
>
> PLE Async 默认 OFF。只有本轮授权包含 T5，且有实际 consumer 等待、合法独立窗口、设备侧计时和 exact row reference，才允许做一个仅移动现有 launch/wait 的实验；不改 row 算法、不加 ring、不改 Expert/模型算术。T5 额外物理预算按本文件规定，完成后仍不得自动启用默认路径。
>
> 若瓶颈是 CPU Expert 或 GDN/QSA，交付具体 kernel 立项卡而不是自行开始重写；若证据不足或收益不成立，输出 INSUFFICIENT_EVIDENCE 或 NO_CHANGE_SELECTED。无修改是允许结果，但必须有实际组件图、回放入口和证据分析。
>
> 最后交付组件/算子地图、归因报告、唯一候选决策、测试与 patch 清单、机器可读状态和证据哈希。保留 PRODUCT_QUALITY/PERFORMANCE_QUALIFICATION=PENDING。停止本轮 Round9 测试环境，训练容器保持运行；不 push、不合并 main、不打 tag、不恢复旧 Round7 定时任务，不修改或暂存 `2026-09-20-04-PAUSED.md`。

---

## 附录 A：执行前应复核的 v61 证据哈希

下表从 [R] 原样转录，是预期值，不是本文件编制阶段已经校验过的结果。

| 文件 | 用户报告的 SHA256 |
|---|---|
| `midprefill-cancel-result.json` | `ad68bedae8b191e174d2ce2611e114e39c387df01cb863ad560222141d963bcb` |
| `working-set-timeline.jsonl` | `6eac0e98c55955d70ad890bfd8b43e5027e97357448b83c82032c4015cac5840` |
| `working-set-timeline-summary.json` | `f331871c0a8c7b0e0a3d847db9408097a88865a6823513b300b4a4823a252d46` |
| `monitor.jsonl` | `dc883bd8f0354df0d26a9263f3e03b7a02136364424987d7ed0413ab9e81f665` |

哈希只能帮助确认读到同一份文件，不自动证明文件中的解释、计时口径或 PASS 判据正确；T0/T3 还需要阅读 schema、生成逻辑和关键内容。

---

## 附录 B：本轮常见术语

| 术语 | 本文件中的含义 |
|---|---|
| Component replay | 固定有限输入，直接重放某个真实组件，不通过完整 HTTP 链路 |
| Oracle / Reference | 用来判断输出是否正确的可信实现/数据，不是候选自己的输出 |
| Backend / Provider | 实际执行算子的实现或库，不只是模型接口名称 |
| Kernel | 在 CPU/NPU 上执行具体计算的底层实现 |
| Placement | 将哪些 Expert 留在 NPU、哪些交给 CPU 的安排 |
| Resident / Shadow | 常驻设备 Expert 与临时设备权重工作区，生命周期不同 |
| Canonical source | 能明确定位模型与权重语义的原始权重来源 |
| Identity key | 判断数据/缓存是否属于同一模型、层、格式、来源和状态的完整标识 |
| Request / Placement / Allocation epoch | 请求、权重放置和资源分配各自的生命周期版本，不能互相代替 |
| Consumer wait | 真正使用某份数据的计算等待其就绪的区间 |
| Critical path | 决定本次结果最早何时完成的依赖链，不等于所有耗时之和 |
| Inclusive duration | 包含子工作或重叠工作的范围时间，不能直接当成互斥占比 |
| Overlap | 两段工作在可比较的时间轴上真实同时执行，不只是异步 API 已返回 |
| Bounded | 请求、工作区、事件数、时间窗口等有预先确定的上限 |
| Qualification | 在明确范围与指标下完成资格验证，高于一次 smoke 或一次 profile |
| Fail-closed | 无法证明身份或资源安全时不继续执行，只收口本轮自有工作 |

---

<a id="references"></a>

## 附录 C：官方公开参考与适用边界

以下资料均在 2026-09-22 通过公开网页核对。地址使用代码形式保留，便于本地执行者锁定具体 revision；这不是自动执行安装、拉取最新镜像或更新依赖的指令。

### [U1] KTransformers 主仓库

地址：`https://github.com/kvcache-ai/ktransformers`

核对范围：Overview、Inference capabilities、原一体化代码的 archive 说明。

用途：确认 KT 的异构推理定位与当前组件组织。不能据此声称每一个后端或模型均已在本地支持。

### [U2] KT-Kernel README

地址：`https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md`

核对范围：CPU 后端、ARM/KML 源码构建入口、NUMA、SGLang fork 集成与独立 Python API。

用途：确认可做组件级使用与具体依赖边界。Python API 示例的 CUDA stream 不等于本地 NPU ABI；某后端的精度支持也不能迁移到另一后端。

### [U3] 官方 DeepSeek-V4-Flash Ascend 教程

地址：`https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md`

核对范围：NPU Attention/dense/resident Expert 与 Host routed Expert 分工、SGLang NPU 环境、镜像/依赖边界。

用途：作为同平台异构架构参考。模型不同、精度不同，不能直接继承其性能或全部支持结论。

### [U4] KT Expert Scheduling 教程

地址：`https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md`

核对范围：placement 策略、resident 数量/比例与 dynamic 更新入口。

用途：语义比较。per-layer/global budget 的实际算法必须再核选定源码，不按 flag 名字认定相同。

### [U5] KT Native Precision 教程

地址：`https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/Native-Precision-Tutorial.md`

核对范围：Hybrid 与 Layerwise Prefill、额外设备工作区需求。

用途：解释分阶段执行思想。不把官方阈值、格式或设备性能当成本地测量结论。

### [U6] SGLang HiCache System Design and Optimization

地址：`https://docs.sglang.io/docs/advanced_features/hicache_design`

核对范围：prefix reuse、设备/Host/存储分层及元数据组织。

用途：界定未来可复用的系统职责；不是保证任意模型的单请求活跃状态都可直接逐页卸载。

### [U7] SGLang Ascend 支持表

地址：`https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/reference/support_features.mdx`

核对范围：Hierarchical cache、Mamba cache 和相关参数。

关键限制：hierarchical cache 条目注明当前不支持 mamba cache。该限制提醒本地执行者不能将通用支持标记等同于 Qwen 混合状态组合已经合格；具体运行仍须按冻结版本核验。

### [U8] Qwen3.8-Flash-Next SGLang Roadmap

地址：`https://github.com/sgl-project/sglang/issues/38731`

核对范围：kernel optimization、FP8 KV、HiCache、PLE offload 等路线条目。

用途：发现上游方向与已报告线索。Issue 中的验证备注不等于本地 Ascend + KT + Qwen 的完整兼容性证明，也不能替代已合并源码和实际测试。

### [U9] Qwen 官方模型卡

地址：`https://huggingface.co/Qwen/Qwen3.8-Flash-Next`

核对范围：Gated DeltaNet、Qwen Sparse Attention、Gated Residual、N-gram embedding、MoE 与语言模型层组织。

用途：核对目标模型结构；本地 model/config/revision 仍须单独锁定。官方上下文能力和模型评测不能直接转化为 Round9 的部署资格。

---

**最终收口句：** 本轮要交付的是“KT 核心可独立验证、Qwen 后端可定位、真实开销可归因、下一项改动有证据”，而不是再增加一个名字像完整功能的组件。
