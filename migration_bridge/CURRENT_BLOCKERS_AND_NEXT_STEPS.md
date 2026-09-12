# KTransformers × Kunpeng × Ascend 当前问题与下一步排查计划

> 文档状态：当前 blocker、风险和执行建议
> 更新时间：2026-09-11
> 当前首要任务：`P3 ENTRY AND SCALE-OUT QUALIFICATION`
> 适用配置：DeepSeek-V2-Lite revision 604d5664、TP1、BF16、LLAMAFILE /
> CPUInfer、Ascend NPU0、Graph/Deferred/Dynamic/MTP/Speculative OFF

## 1. 当前问题总览

本文件第 2–11 节保留了 P2 blocker 的原始复现与排查逻辑，作为 root-cause
审计记录。Round 4A.5 已经关闭这些问题；当前没有已知 P2 blocker，下一风险面
是 P3 八层 scale-out。

| 优先级 | 问题 | 当前状态 | 是否阻止 P3 |
|---|---|---|---:|
| P0 | P3 八层/32 expert scale-out 尚未执行 | `NOT_RUN` | 最终 multi-placement 尚不能 READY |
| Closed | P2 同路径 greedy 输出非确定 | `FIXED_AND_A3_REQUALIFIED` | 否 |
| Closed | retained Round 2B runtime pipeline 异常大输出 | `FP32_BUFFER_TEST_FIXED; 23+1 PASS` | 否 |
| Closed | retained Round 2A 的 4 个旧 dtype 断言 | `TEST_CONTRACT_FIXED; 21/21 PASS` | 否 |

此外还有 ISA portability、NUMA enforcement、性能和未来 TP/Graph/量化等已知
风险，但它们不是当前 P2 failure 的已证实原因。

## 2. P0：P2 Same-Path Nondeterminism

### 2.1 冻结配置

P2 使用原 Round 4A placement，没有重新挑选低频 expert：

```text
CPU-enabled layers = {1, 9, 17, 26}
CPU experts/layer   = 4
total CPU experts   = 16
physical NPU experts per enabled layer = 60
placement SHA256 =
f6d4e9c6a2e5e8060e846dbc7c628d069c9aa6150aaa1e2690ea28a51ba286a3
```

P2 F32 GGUF：

```text
size   = 8,858,371,392 bytes
SHA256 = 2c4cd307a53b761c4191cf108d1e00ec2bd2c99b48a595a28ac8fdd0e0edd1fa
```

### 2.2 精确复现

同一个 serving process、相同 model/tokenizer、相同 prompt、相同 seed、相同
greedy setting，重复 10 次 64-token generation：

| Prompt | Repeat | Unique output hashes | First differing token |
|---|---:|---:|---:|
| `v_en_01` | 10 | 6 | 1 |
| `v_zh_01` | 10 | 1 | none |
| `v_struct_01` | 10 | 9 | 3 |
| `v_struct_02` | 10 | 9 | 2 |

同时出现 8/16/32-token request 不是同 prompt 64-token request 前缀的情况。

硬门禁要求：

```text
Hybrid(prompt, config, seed, run1)
==
Hybrid(prompt, config, seed, run2)
```

实际结果不满足，因此这不是允许由 pairwise numerical tolerance 吸收的
cross-backend 差异，而是同一 execution mode 自身违反 exact determinism。

### 2.3 Sequential control

为了判断问题是否只来自 CPU/NPU overlap，额外使用：

```text
SGLANG_KT_HYBRID_NO_CPU_STREAM=1
```

运行 sequential control。结果：

| Prompt | Repeat | Unique output hashes | First differing token |
|---|---:|---:|---:|
| `v_en_01` | 10 | 4 | 1 |
| `v_struct_01` | 10 | 8 | 3 |

因此已证实：

```text
OVERLAP_ONLY_HYPOTHESIS = REJECTED
SAME_PATH_NONDETERMINISM = CONFIRMED
```

这不等于已经证明 overlap 完全无关；它只证明禁用 CPU auxiliary stream 后问题
仍然存在，所以不能把根因局限为 dual-stream overlap race。

## 3. P2 中已经通过的部分

定位问题时必须保留以下反证，避免重新调查已关闭的问题。

### 3.1 Pairwise numerical contract 没有溢出

| Subset | Positions | Stable exact | Ambiguous membership | Max distortion | Overflow |
|---|---:|---:|---:|---:|---:|
| Q2 subset | 256 | 137/137 | 119/119 | 2.125 | 0 |
| H2 subset | 256 | 155/155 | 101/101 | 1.875 | 0 |

冻结合同的 `B_pair=2.1875` 没有被扩大，所有 logits/metrics finite。

### 3.2 Placement coverage 通过

```text
CPU-enabled layers exercised = 4 / 4
selected CPU experts hit      = 16 / 16
expert coverage               = 100%
total CPU route hits          = 74,784
routing weights finite        = YES
```

因此不能通过“换低频 expert”或“降低 CPU hit”规避问题，也没有证据表明 failure
是因为某个 layer/expert 从未被实际覆盖。

### 3.3 P1 同样的核心路径确定

P1 的单层 4 CPU Expert 配置已经通过：

- same-process 10-repeat；
- clean restart；
- 8/16/32/64 prefix；
- sequential == overlap；
- 1,000-forward single hash；
- CPU-not-hit 32/32 exact；
- C0-C4 explicit reference；
- held-out pairwise contract 和 downstream quality。

这说明基本的 LLAMAFILE expert kernel、单层 mapping、P1 merge 和单层生命周期
本身不是一个普遍非确定源。问题由 P1 扩展到 P2 后出现，优先关注多层状态、
buffer 复用、跨层 callback/lifetime 和多 wrapper 交互。

## 4. 当前能下的结论与不能下的结论

### 4.1 已证实

```text
P2 repeated greedy output is not exact
P2 prefix consistency fails
failure occurs in overlap mode
failure also occurs in sequential-control mode
P2 pairwise subset envelope does not overflow
P2 route coverage is complete
all captured pairwise values are finite
```

### 4.2 尚未证实

当前不能把根因直接写成以下任意一种：

```text
CPU GEMM reduction nondeterminism
CANN callback race
SGLang scheduler race
stale host buffer
uninitialized output buffer
cross-layer mapping corruption
CPUInfer task lifetime bug
NPU grouped-MoE nondeterminism
thread scheduling / NUMA effect
```

这些都是候选 hypothesis，不是已经完成的 root-cause attribution。下一轮必须
通过从 layer contribution 到 logits 的正向证据决定哪一个成立。

## 5. 优先级最高的根因假设

以下排序依据是“P1 单层通过、P2 多层失败、sequential control 仍失败”。

### H1：跨层 host/device buffer 生命周期或复用

多个 CPU-enabled layer 可能让 host input/output、route IDs、route weights 或
callback args 的复用时序不同于 P1。若上一层/上一 token 的 buffer 在 consumer
完成前被下一层覆盖，就会产生 run-to-run 变化。

验证信号：相同 teacher-forced history 下，某层 CPU/NPU partial 的输入 hash
一致但输出或 merge 前 buffer hash 不一致；加入 buffer poison/guard 后失败位置
变化或触发 invariant。

### H2：多 CPUInfer wrapper/task 的 callback 与完成顺序

P2 同时存在 4 个 CPU-enabled layer。即使关闭辅助 CPU stream，CPUInfer task、
host callback 或 wrapper 内部队列仍可能共享状态或完成标记。

验证信号：每层 task sequence/callback sequence 不稳定，或 layer-local sync
后 determinism 恢复。

### H3：某一新增 P2 layer 的局部输出不确定

Layer 1、9 或 26 中的某一层可能单独触发 kernel shape、mapping、weight offset
或 output initialization 问题；Layer 17 已在 P1 中通过。

验证信号：诊断性 layer-subset bisection 能把 failure 缩小到某个 layer；该
subset 只用于归因，不改变最终 P2 acceptance placement。

### H4：partial merge 前存在未初始化或残留值

CPU-owned/NPU-owned routes的 placeholder、output slice 或 routed sum 如果没有在
每个 forward 明确归零，CPU hit pattern 变化可能读取历史值。Round 2B retained
pipeline 中观察到异常大输出，使这一方向需要优先交叉检查，但两者尚未证明是
同一个根因。

验证信号：分配后强制 poison，然后在合法写入点清零；任何 poison 到达 merge
都立即 fail-fast；不同运行中首次不同字节位于未完全覆盖的 output slice。

### H5：更底层的 CPU 或 NPU kernel 同路径不确定

虽然 P1 1,000-forward 和单 expert repeats 均稳定，多层真实 shape/调用序列可能
触发不同 reduction/thread scheduling。该假设优先级低于生命周期/复用，但不能
在逐层输出捕获前排除。

验证信号：输入、weights、route 和 buffer 初始化完全一致，单层 isolated
kernel 输出仍出现不同 hash。

## 6. 建议的下一轮最小排查顺序

下一轮不应该直接重写 Hybrid，也不应该先跑 P3。建议严格执行以下顺序。

### Step 0：冻结复现条件

只保留最早、最短的两个复现：

```text
v_en_01, first divergence token 1
v_struct_01, first divergence token 3
```

固定 model、tokenizer、P2 placement、GGUF、seed、threadpool、NPU0 和 serving
参数。每个实验至少重复到能区分 deterministic/split outcome。

退出证据：相同原始 P2 failure 可在新 disposable container 中复现；否则先处理
environment drift。

### Step 1：用相同 token history 做 teacher-forced 双运行比较

Greedy 分叉后 history 不同会污染后续归因。应将两次 Hybrid run 都强制使用
同一 All-NPU 或 frozen baseline history，只比较分叉前后 matched-history 状态。

至少捕获：

```text
layer input
router IDs / weights
CPU input/output
NPU premerge output
merged routed output
shared output
layer output
final logits
```

退出证据：定位第一个 run-to-run 不同的 layer/stage，而不是只看到 token 不同。

### Step 2：诊断性 layer bisection

保持最终 P2 placement 不变作为 acceptance source of truth，但允许仅为根因归因
运行以下诊断组合：

```text
{17}
{1}
{9}
{26}
{17,1}
{17,9}
{17,26}
{1,9,17,26}
```

不能用通过的子集替代 P2，也不能据此重新选择低频 expert。目标只是判断 failure
需要某个 layer 还是需要多 wrapper/多层组合。

退出证据：最小 failing layer set 和最小 passing control。

### Step 3：逐阶段 byte hash 与 deterministic repeat

对最小 failing set，在每个 layer/stage 输出中记录：

```text
shape
dtype
device
data_ptr / buffer generation
SHA256
finite
write-complete sequence
callback/task sequence
```

比较至少 10 次。找到第一个不同 hash 后停止向后追 token，转而调查该 stage
的 producer/consumer。

### Step 4：buffer ownership 与 poison 检查

为 debug-only 路径加入：

- 每个 forward/layer 唯一 generation ID；
- callback args generation assertion；
- host/device output poison；
- merge 前 finite/poison absence assertion；
- borrowed stream handle 和 wrapper identity 记录；
- output slice write coverage。

所有插桩默认关闭，不改变 production arithmetic。

### Step 5：同步点定位，不把同步当最终修复

在已定位的 producer/consumer boundary 逐点加入 debug-only hard sync，判断哪一个
同步能让 run-to-run hash 恢复唯一。同步实验只用于证明 ordering/lifetime
hypothesis；最终修复必须落在正确 ownership/event/callback contract 上，不能用
全局同步掩盖问题。

### Step 6：最小 production fix 与回归

只有 root cause 被前向证据证明后，实施最小修复。修复后依次运行：

1. 首个 captured failing token regression；
2. 最小 failing layer set 10-repeat；
3. 原 P2 四 prompt 10-repeat 和 prefix；
4. P2 Q2/H2 frozen pairwise subset；
5. 4/4 layers、16/16 experts coverage；
6. P2 quality smoke；
7. Round 2A/2B/2C、Round 3、SGLang KT EP、P1 回归。

只有这些全部通过，才能恢复 P3。

## 7. P2 修复后的硬验收条件

P2 必须同时满足：

```text
system routing/mapping/ownership/shared/scaling = EXACT
same-path repeated greedy token IDs             = EXACT
8/16/32 prefixes of 64                          = EXACT
CPU-not-hit control                             = EXACT
expert rel-L2                                   <= 1e-2
pairwise distortion                             <= 2.1875
stable top1                                     = 100% exact
ambiguous top1                                  ∈ frozen ambiguity set
pairwise overflow                               = 0
all values                                      = finite
4/4 CPU-enabled layers                          = hit
>=75% selected CPU experts                      = hit
quality smoke                                   = PASS
```

不允许为 P2 单独修改 `B_pair`、candidate K、ambiguity rule 或质量协议。

## 8. P1：retained Round 2B Runtime Pipeline Failure

Round 4A.4 收尾时重新运行 retained Round 2B 模块：

```text
22 passed
1 skipped
1 failed
```

失败项：

```text
test_d2h_cpuinfer_moe_h2d_pipeline
```

观察到：

```text
max_abs_error = 1.7014118346046923e+38
relative_l2   = Infinity
```

该测试在 Round 2B 原始开发轮曾通过，因此不能把本次结果当作“历史上从未工作”。
当前应标记为 unresolved regression。它可能来自测试与新 FP32 partial contract 的
不兼容、buffer 初始化/lifetime、当前容器运行条件或真实 runtime regression。

建议与 P2 H4 同步调查：先确认 output host/NPU buffer 的 dtype、size、初始化、
copy completion 和 lifetime。若异常值来自残留/未写 buffer，它可能提供比
full-model P2 更小的复现；若只是 retained test contract 过时，则应独立更新测试并
证明 numerical pipeline 在当前 boundary 下正确。

在重新证明前，不应把当前 Round 2B retained regression 写成 PASS。

## 9. P2：retained Round 2A 旧 dtype 断言

Round 2A retained suite 当前结果：

```text
17 passed
4 failed
```

4 个失败都来自测试仍假设 CPU wrapper output 是 BF16，而 Round 4A.1 已经明确
要求 CPU/NPU routed partial 保持 FP32，到 merge 后才做单次最终 BF16 cast。

其中 GGUF E2E 的实际数值仍为：

```text
max_abs_error = 0.0001857281
mean_abs      = 0.0000500941
relative_l2   = 0.003893714
```

建议修复方式是更新 retained test 对 partial boundary 的 dtype 和 reference
contract，并继续保留 numerical envelope。禁止为了让旧测试变绿而撤销 FP32
partial accumulation production fix。

## 10. 非当前 blocker 的已知风险

以下风险需要保留，但不应混入当前 P2 root-cause conclusion：

### ARM ISA portability

当前 aarch64 build 使用 `armv8.2-a+fp16+dotprod+sve+bf16`，A3 实机通过，但
不能宣称兼容所有 Kunpeng 920 或任意 generic aarch64 CPU。

### NUMA

NUMA node、threadpool mapping 和 process affinity 可观测；非特权容器中的强制
memory binding 没有形成生产级证明。

### Performance

当前工作以 correctness 为主。Round 4A 初始 P1 的 45-request wall sum 约为
All-NPU 53.82 秒、Hybrid 109.97 秒。没有生产 throughput 或 latency claim。

### Future modes

TP2/HCCL、Graph、Deferred、Dynamic Placement、MTP、Speculative、W8A8、
MXFP4、生产并发和 NUMA scaling 均尚未资格验证。

## 11. 明确禁止的“修复”方式

当前问题不能通过以下方式关闭：

- 更换低频 CPU experts；
- 减少 P2 CPU hit；
- 删除 `v_en_01`、`v_struct_01` 或其他困难 prompt；
- 把 same-path mismatch 解释为 cross-backend near-tie；
- 扩大 `B_pair`；
- 修改 candidate Top-K 或 ambiguity membership rule；
- 让 CPU-owned expert 回到 NPU 执行；
- 用全局同步作为没有根因证明的永久修复；
- 在 P2 未通过时启动 P3；
- 修改 A3 宿主机或业务容器来制造不可复现环境。

## 12. 建议下一轮交付物

建议下一轮围绕 P2 determinism 单独建立目录和交付：

```text
migration_bridge/round4a5/
├── 00_BASE_AND_REPRODUCTION.md
├── 01_FIRST_DIVERGENCE_CAPTURE.md
├── 02_LAYER_BISECTION.md
├── 03_BUFFER_AND_CALLBACK_LIFETIME.md
├── 04_ROOT_CAUSE.md
├── 05_MINIMAL_FIX.md
├── 06_P2_DETERMINISM_REQUALIFICATION.md
├── 07_P2_NUMERICAL_AND_QUALITY.md
├── 08_REGRESSION.md
├── tools/
├── evidence/
└── ROUND4A5_SUMMARY.md
```

Round 4A.5 的成功标准应该是修复并重新资格验证原 P2，而不是提出新的 numerical
acceptance mechanism。

## 13. 主要证据入口

```text
migration_bridge/round4a4/ROUND4A4_SUMMARY.md
migration_bridge/round4a4/15_P2_RESUME.md
migration_bridge/round4a4/19_REGRESSION.md
migration_bridge/round4a4/PAIRWISE_NUMERICAL_ACCEPTANCE_CONTRACT.json
migration_bridge/round4a4/evidence/p2-repeat-analysis.json
migration_bridge/round4a4/evidence/p2-sequential-control-analysis.json
migration_bridge/round4a4/evidence/p2-route-coverage.json
migration_bridge/round4a4/evidence/p2-q2-validation.json
migration_bridge/round4a4/evidence/p2-h2-validation.json
migration_bridge/round4a4/evidence/regression-round2a.log
migration_bridge/round4a4/evidence/regression-round2b.log
migration_bridge/round4a4/evidence/regression-round2c.log
```

## 14. 当前最终决策

```text
PAIRWISE_NUMERICAL_ACCEPTANCE = HELDOUT_VERIFIED
P1_REQUALIFIED                = A3_VERIFIED_READY
MULTI_LAYER_P2                = A3_VERIFIED_READY
MULTI_LAYER_P3                = NOT_RUN
P3_ALLOWED                    = YES
FINAL_MULTI_PLACEMENT         = IN_PROGRESS_P3_REQUIRED
```

下一步唯一合理的主线是：从不可移动的
`round4a5-p2-a3-verified` 标签创建 P3 分支，冻结 layers
`{1,5,8,12,17,19,22,26}` / 32 CPU experts，并在不改变 placement、不修改
`B_pair=2.1875`、不降低 coverage 的前提下依次执行 P3 determinism、pairwise、
coverage、free-generation、2,304-token stability 和 downstream quality。

任何 stable flip、pairwise overflow、same-path nondeterminism、coverage collapse、
质量回归或 memory/lifetime failure 都必须停止 P3，不得通过更换 expert、减少
CPU hit 或扩大阈值规避。P2 的完整关闭证据见
`round4a5/ROUND4A5_FINAL_SUMMARY.md`。
