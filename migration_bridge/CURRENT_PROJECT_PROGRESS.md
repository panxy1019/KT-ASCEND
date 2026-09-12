# KTransformers × Kunpeng × Ascend 当前整体进展

> 文档状态：当前工程总览
> 更新时间：2026-09-11
> 项目：KTransformers × Kunpeng × Ascend Migration
> 当前主开发分支：`feature/kt-round4a5-p2-closure`
> 本文事实基线：P2 A3 runtime/test candidate `d36dd8f3e01c9a86d87309e972b11e75b537b68e`，
> SGLang 子仓库 `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2`

## 1. 执行摘要

项目已经完成从“确认最新版 KTransformers 在 Kunpeng 上能否运行 CPU
Expert”，到“DeepSeek-V2-Lite 在 Kunpeng CPU 与 Ascend NPU 上完成 TP1
真实模型单层多 Expert Hybrid”的主要技术链路。

当前最重要的已验证结果是：

```text
CPU_EXPERT_PLANE                         = A3_VERIFIED_READY
ASCEND_RUNTIME_PLANE                     = A3_VERIFIED_READY
HYBRID_MOE_SYNTHETIC_SINGLE_LAYER        = A3_VERIFIED_READY
DEEPSEEK_V2_LITE_TP1_SINGLE_CPU_EXPERT   = A3_VERIFIED_READY
PAIRWISE_NUMERICAL_ACCEPTANCE            = HELDOUT_VERIFIED
P1_MULTI_EXPERT_SINGLE_LAYER             = A3_VERIFIED_READY
P2_MULTI_LAYER                           = A3_VERIFIED_READY
P3_MULTI_LAYER                           = NOT_RUN
DEEPSEEK_V2_LITE_TP1_MULTI_PLACEMENT     = IN_PROGRESS_P3_REQUIRED
```

因此，项目并不是“整体未跑通”：CPU 平面、Ascend runtime、Synthetic
Hybrid、真实模型单 CPU Expert、真实模型单层四 CPU Expert、数值合同和 P1
质量验证都已经完成。Round 4A.5 已定位并修复四层 P2 的 same-path
nondeterminism，完整关闭 P2；当前未闭环的是八层 P3 的 scale-out、stability
和最终质量资格验证。

## 2. 当前冻结目标与运行边界

当前资格验证对象固定为：

| 项目 | 冻结值 |
|---|---|
| 模型 | DeepSeek-V2-Lite |
| 模型 revision | `604d5664` |
| Serving | `kvcache-ai/sglang` 的 `sglang-kt` 路径 |
| CPU Expert backend | LLAMAFILE / CPUInfer |
| CPU | Kunpeng 920 / aarch64 |
| NPU | Ascend NPU0 一张卡 |
| 并行 | TP=1，batch=1 |
| 模型 dtype | BF16 |
| CPU GGUF 实验权重 | BF16-valued weights materialized as F32 |
| Graph | OFF |
| Deferred Experts | OFF |
| Dynamic Placement | OFF |
| MTP / Speculative | OFF |
| CPU threadpool | 1 |

当前结论不外推到 TP2/TP4/TP8、HCCL、Graph、量化、动态迁移、MTP、推测
解码或生产并发性能。

## 3. 当前系统架构

实际 Hybrid 路径保持如下职责分离：

```text
SGLang model / router
        │
        ├── global expert IDs + routing weights
        │
        ▼
KTEPWrapperMethod
        │
        ├── NPU-owned routes ──> Ascend grouped MoE / resident NPU weights
        │
        └── CPU-owned routes ──> KTMoEWrapper
                                  └── LlamafileMoEWrapper
                                      └── CPUInfer
                                          └── LLAMAFILE ARM MoE
        │
        ▼
CPU partial + NPU partial in FP32
        │
        ▼
single final BF16 cast
        │
        ▼
outer model layer adds shared expert exactly once
```

已经冻结并验证的系统语义包括：

- router 继续使用 global logical expert ID；
- CPU/NPU ownership 完整、互斥；
- logical-to-physical NPU mapping 显式维护；
- CPU-owned route 在 NPU grouped MoE 中被禁用；
- NPU-owned route 不会进入 CPU MoE；
- routing weight 和 `routed_scaling_factor` 各应用一次；
- shared expert 由 outer model layer 唯一拥有；
- CPU/NPU partial 使用 FP32 合并，然后只做一次最终 BF16 cast；
- stream/native handle 所有权属于 torch_npu/CANN，CPUInfer 只借用；
- host callback 不发起设备工作、不同步设备。

## 4. 分阶段进展

### 4.1 Round 1：最新版代码与 A3 基线审计

Round 1 冻结了最新版 KTransformers 和 SGLang 调用链，确认了 A3 的
Kunpeng/Ascend/CANN 环境，也发现最新版仓库中的 KML 配置与源码不完整：
CMake 仍引用已不在当前树中的 KML 目录，A3 也没有可证明兼容的 KML
runtime。

本轮没有通过猜测恢复历史 KML，也没有直接加入未经验证的 Ascend patch。

结论：

```text
LATEST_REPOSITORY_AUDIT = COMPLETE
KML_CURRENT_TREE_BUILD  = BLOCKED
ASCEND_GAP              = DOCUMENTED
```

### 4.2 Round 1.5：CPU 基线与 KML 源码考古

在 disposable CPU 容器中完成了当前代码的 aarch64 CPU-only 构建，证明：

- 不依赖 KML 也可以构建 `kt_kernel`；
- CPUInfer 可真实 load/forward；
- LLAMAFILE ARM 通用 MoE 路径可运行；
- 1,000 次 submit/sync 稳定；
- 历史 KML 目录确实在旧提交中存在，后来被删除；
- 将旧 KML 树整体恢复到当前 API 风险过高。

正式选择 Route A：LLAMAFILE first，KML 降级为未来独立优化项。

### 4.3 Round 2A：ARM LLAMAFILE CPU Expert Plane

完成了 CPU Expert 平面的工程化资格验证：

- aarch64 variant 正确报告为 `arm`，不再错误伪装成 AVX2；
- deterministic GGUF fixture 和 production GGUF key 路径完成；
- 4/8 experts、Top-K=2、identity/permutation mapping 通过；
- qlen 1/2/8/32/64 通过；
- multi-layer isolation 通过；
- CPUInfer 1,000-forward deterministic，RSS delta 为 0；
- threadpool 1/2 与 NUMA 可观测性完成；
- 当轮测试为 21 passed。

出口状态：

```text
CPU_EXPERT_PLANE = A3_VERIFIED_READY
```

### 4.4 Round 2B：Ascend Runtime Plane

基于公开 CANN Runtime API 完成 Ascend runtime 平面：

- Ascend-enabled aarch64 build；
- torch_npu stream 到原生 `aclrtStream` 借用桥；
- `aclrtLaunchHostFunc` callback；
- D2H/H2D pinned transfer；
- CPUInfer submit/sync callback；
- D2H → CPU MoE → H2D runtime pipeline；
- callback/stream/CPUInfer 生命周期和 RSS stress；
- CPU-only build 仍不产生 ACL 依赖。

当轮结果是 23 passed、1 个预期 CUDA-only skip。

出口状态：

```text
ASCEND_RUNTIME_PLANE = A3_VERIFIED_READY
```

### 4.5 Round 2C：Synthetic Single-Layer Hybrid MoE

在真实 CPUInfer + LLAMAFILE 与真实 Ascend NPU 上完成 synthetic 单层 Hybrid：

- CPU/NPU ownership 和 global/local mapping；
- CPU-only、NPU-only、mixed、reversed mixed routes；
- sequential 与 overlapped 路径；
- qlen 1/8/32；
- sequential/overlap contribution bitwise equality；
- 真实时间区间证明 CPU/NPU 存在 overlap；
- shared expert outer-owner contract；
- 1,000 mixed overlap cycles、20 次 wrapper recreate、100 次 stream lifecycle。

当轮最终 44 项回归通过。

出口状态：

```text
HYBRID_MOE_SINGLE_LAYER = A3_VERIFIED_READY
```

### 4.6 Round 3：DeepSeek-V2-Lite TP1 Ascend 集成

Round 3 将 synthetic Hybrid 接入真实 DeepSeek-V2-Lite 和 SGLang serving：

- All-NPU TP1 BF16 baseline 完整启动、prefill、decode、generation；
- 修正 import-time CUDA probe 和 router CUDA JIT；
- 接入 Ascend KTEP stream/event/native handle；
- 发现并修复 decode D2H staging race；
- Layer 17 / Expert 8 作为唯一 CPU Expert；
- 真实权重 CPU repeat byte-identical；
- 15-case full-model A/B 为 15/15 exact token；
- 最大 `|delta logprob|=0.0810541`；
- 两轮 576-token stability campaign 均通过；
- Round2A/2B/2C 和 SGLang 回归在当轮全部通过。

出口状态：

```text
DEEPSEEK_V2_LITE_TP1 = A3_VERIFIED_READY
```

### 4.7 Round 4A：P1/P2/P3 Placement 设计和 P1 初始验证

基于独立 selection corpus 完成 expert route profiling，并冻结三种 placement：

| Placement | CPU-enabled layers | CPU experts | 状态 |
|---|---|---:|---|
| P1 | `{17}` | 4 | 后续已在 4A.4 恢复资格 |
| P2 | `{1,9,17,26}` | 16 | Round 4A.5 A3 VERIFIED READY |
| P3 | `{1,5,8,12,17,19,22,26}` | 32 | NOT RUN |

P1 的 Layer 17 CPU experts 固定为 `{6,8,25,36}`。Controlled C0-C4、mapping、
routing、shared/scaling、sequential/overlap 和 1,000-forward 均通过，但初始
full-model exact token 只有 30/45，因此没有继续 P2/P3。

### 4.8 Round 4A.1：数值语义缺陷修复

该轮找到并保留了两个 production fix：

1. 对齐 gate、up、SwiGLU multiply、down output 和 route weight 的
   Ascend-visible BF16 rounding boundary；
2. CPU/NPU partial 不再分别提前 BF16 round，而是在 Hybrid boundary 使用
   FP32 相加后做一次最终 BF16 cast。

P1 exact request 从 30/45 提升到 42/45，证明原严格门禁成功捕获了真实实现
缺陷。剩余 3 个 failed request 实际属于一个 near-tie trajectory。

### 4.9 Round 4A.2：CPU GEMM Backend 调查

对 LLAMAFILE、OpenBLAS、BLIS、ATLAS、ACL NEGEMM 和当前 KML 可行性进行
隔离调查。OpenBLAS 修复了 `v_struct_03` 的已知 near-tie，却在 `v_en_01`
引入新的 near-tie，最终仍为 42/45。

结论不是“OpenBLAS 无法计算”，而是简单更换 CPU GEMM backend 不能让全
corpus greedy trajectory 单调逼近 All-NPU。实验 adapter 被移除，生产路径
保留 LLAMAFILE。

### 4.10 Round 4A.3：Global-Epsilon Contract 尝试

该轮建立了 qualification Q 和独立 held-out H。Stable exact、near-tie
membership 和主要 numerical envelope 在 H 上均通过，但旧合同还冻结了：

```text
max tie-set size <= 6
```

H 中出现 top-16 truncated tie-set size 至少为 16，因此合同按预设规则正确
REJECT。没有在看到 H 后修改阈值，也没有继续 P1/P2/P3。

### 4.11 Round 4A.4：Pairwise-Margin Contract 与 P1 恢复

Round 4A.4 使用新的机制、新 Q2 和全新 H2，基于 pairwise logit-order
stability 重新建立合同。

Q2：

```text
18 prompts
1,152 teacher-forced positions
36,104 candidate pairs
pairwise distortion max = 1.75
B_pair = 1.25 × Q2 max = 2.1875
stable exact = 639 / 639
ambiguous membership = 513 / 513
```

H2 held-out：

```text
16 new prompts
1,024 positions
stable exact = 654 / 654
ambiguous membership = 370 / 370
pairwise overflow = 0
max distortion = 0.6875
all finite = PASS
```

H2+F 自由生成共出现 5 个 first divergence，全部发生在 pairwise-ambiguous
区域，5/5 Hybrid token 均属于冻结 ambiguity set，没有 stable-region flip。

质量验证：

| Benchmark | All-NPU | Hybrid P1 | 结论 |
|---|---:|---:|---|
| C-Eval 128 | 57 | 57 | 无回退 |
| GSM8K MC 128 | 45 | 47 | paired bootstrap 未显示显著回退 |

正式合同状态：

```text
PAIRWISE_NUMERICAL_ACCEPTANCE = HELDOUT_VERIFIED
```

随后 P1 重新资格验证通过：

```text
C0-C4                         = PASS
CPU-not-hit                  = 32/32 exact
same-process repeat          = PASS
clean restart / prefixes     = PASS
sequential == overlap        = PASS
1000-forward unique hash     = 1
expert rel-L2 max            = 0.0035398305 <= 0.01
quality                      = PASS
P1_REQUALIFIED               = A3_VERIFIED_READY
```

## 5. 当前 P1、P2、P3 状态

### P1：已完成

P1 是当前最高级别的完整已验证能力：真实 DeepSeek-V2-Lite、真实 routing、
单层 4 个 CPU experts、60 个物理 NPU experts、真实 serving、held-out 数值合同、
free generation 和质量 A/B 均已闭环。

```text
P1_REQUALIFIED = A3_VERIFIED_READY
MULTI_EXPERT_SINGLE_LAYER = A3_VERIFIED_READY
```

### P2：Round 4A.5 已完整关闭

旧 blocker 已归因为 raw `acl.rt.memcpy` 指针传输绕过 torch/torch_npu 的
stream 和 tensor-lifetime 语义。生产路径改为 framework-managed blocking
Tensor `copy_` 后，在清理无必要诊断 hook 的候选提交上重新资格验证：

- 9 prompts × 10 repeats × 64 tokens，同进程 exact；
- 8/16/32 prefix 全部 exact；
- clean restart 9/9 exact；
- Q2 `137/137` stable exact、`119/119` ambiguity membership、0 overflow；
- H2 `155/155` stable exact、`101/101` ambiguity membership、0 overflow；
- 4/4 layers、16/16 experts、74,807 CPU route hits；
- CPU-not-hit 32/32 exact；
- 同环境 frozen quality：C-Eval 27/128 vs 27/128，GSM8K 31/128 vs 31/128；
- retained regression：Round2A 21/21、Round2B 23+1 skip、Round2C 44/44、
  SGLang 4/4、Round3/P1 compact PASS。

因此：

```text
MULTI_LAYER_P2 = A3_VERIFIED_READY
P3_ALLOWED     = YES
```

### P3：严格未运行

P3 仍未运行。已验证冻结 placement 为 layers
`{1,5,8,12,17,19,22,26}`、32 CPU experts，canonical SHA
`77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d`。
P3 pairwise、coverage、2,304-token stability 和最终质量仍待执行；不得扩大
`B_pair` 或重新拟合合同。

## 6. 当前生产代码状态

目前需要保留的核心 production change 包括：

- ARM variant 与 LLAMAFILE/CPUInfer CPU plane；
- CANN stream、callback、transfer 和 lifecycle runtime plane；
- Hybrid ownership/mapping/merge；
- SGLang KT EP Ascend stream path；
- CUDA import/JIT guard；
- physical NPU expert count 和 CPU-owned route disable；
- mapping dtype normalization；
- Ascend-visible BF16 expert boundaries；
- FP32 Hybrid partial accumulation。

Round 4A.4 本身没有修改 production arithmetic。子仓库新增的
premerge accelerator contribution capture 是默认关闭的 debug/test
instrumentation，只有显式设置 numerical-dump 环境时启用。

## 7. Git 与交付状态

当前 P2 closure 分支已经推送；最终证据提交与不可移动 tag 在 Round 4A.5
freeze 步骤完成后继续推送：

| 仓库 | 分支 | 已推送事实基线 |
|---|---|---|
| KTransformers parent | `feature/kt-round4a5-p2-closure` | A3 runtime/test candidate `d36dd8f3e01c9a86d87309e972b11e75b537b68e` |
| SGLang child | `feature/kt-ep-round4a4-pairwise-margin-qualification` | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |

Round 4A.4 机器可读合同：

```text
migration_bridge/round4a4/PAIRWISE_NUMERICAL_ACCEPTANCE_CONTRACT.json
canonical SHA256:
223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717
```

## 8. A3 环境纪律

所有 build、pytest、模型运行和数据采集均在 disposable container 中执行，
只使用 NPU0。宿主机系统库、KML、业务容器和 NPU1+ 未修改。

Round 4A.4 容器 `kt-r4a4-pairwise` 当前已经停止，NPU0 已释放。模型、GGUF
和大型运行产物没有通过修改宿主环境来“固化”。

## 9. 当前还没有完成的能力

以下内容仍然不能对外宣称完成：

- P3 八层/32 CPU Expert qualification；
- P3 2,304-token stability 和 RSS/HBM campaign；
- DeepSeek-V2-Lite TP1 multi-placement 最终 READY；
- TP2/TP4/TP8、HCCL；
- Graph、Deferred Experts、Dynamic Placement；
- MTP、Speculative Decoding；
- W8A8/MXFP4/其他量化；
- NUMA scaling 和生产并发吞吐；
- 泛化到所有 Kunpeng/aarch64 CPU 的 ISA portability；
- KML production integration。

## 10. 当前交接结论

项目下一阶段不应该重新讨论是否采用 KML，也不应该重新放宽数值合同。P2
已经在完全相同的 frozen pairwise contract 下恢复 exact determinism 并通过
quality。当前唯一主线是从不可移动的 P2 final tag 创建 P3 分支，对八层/32
CPU expert placement 执行 system、pairwise、coverage、free generation、质量与
stability 门禁。上游 Ascend alignment 必须留到独立 Round 5。

详细问题、证据边界和下一步排查顺序见：

`migration_bridge/CURRENT_BLOCKERS_AND_NEXT_STEPS.md`
