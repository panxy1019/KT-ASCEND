# Round18 最终报告：Native-preserving KT Offload 与 Native Expert Pool

日期：2026-09-30 UTC
最终状态：`ROUND18_COMPLETE_NO_PROMOTION`
默认配置：保持 Round17 S0，不迁移到 Round18 vLLM Native Hybrid 或 expert pool

## 1. 结论

Round18 回答了本轮唯一架构问题：能否保留 vLLM-Ascend 原生
`AscendMoERunner`、router、shared expert、finalize 和图执行组织，只把专家容量
问题交给 KT。

结果分为两部分：

1. **最小 Native Hybrid 在 eager 下成立。** 原生 router 在 512 logical
   experts 上产生 Top-10，Physical-Q32 owner 只分配 32 个设备专家，原生
   Ascend kernel 对 resident occurrence 产生 partial，C128/MXFP4 计算其余
   occurrence，二者只合并一次。完整 48 层模型的 canary、Decode、取消和
   reuse 均通过。
2. **冻结栈下没有可晋级的 Native graph 或 Native expert pool。**
   PIECEWISE 受通用动态 scalar/shape ABI 绑定问题阻断；Breakable 被冻结
   Ascend 平台禁用；FULL replay 不重新进入当前 CPU eager seam。Phase II
   又发现唯一容量安全的 Q32 pool 缺少可运行的 current-token device planner，
   当前 NZ29 parent 也没有不重建完整 parent 的固定 slot writer。

因此本轮没有产生可做 fresh 请求 ABBA 的候选。未运行请求性能不是“测得没有
收益”，而是前置正确性、容量和 graph-compatibility gate 已经拒绝候选。默认
继续使用 S0。

## 2. Phase I：保留 AscendMoERunner 的 Native Hybrid

### 2.1 Native resident partial

`native-partial-v004` 使用真实冻结镜像、真实 layer0 权重、原生 Ascend
selector、`AscendUnquantizedFusedMoEMethod` 和 AllGather communication。

| 项目 | 结果 |
|---|---|
| logical / physical experts | 512 / 32 |
| 物理权重 | BF16、NZ29、Q32 |
| 用例 | all-resident、all-nonresident、mixed 1/9、5/5、9/1、multi-token、Top-K tie |
| 最大 relative L2 | 0.0026470785960555077 |
| repeat | exact |
| 生命周期 | 20 轮，allocator 回到 post-construction baseline |

这证明当前 Ascend native expert map 的 `-1` 在这条实际路径中可以跳过
nonresident occurrence。它没有重算 Top-K、没有对 resident 子集重新归一化，
也没有把 shared expert/finalize 混入 partial。

### 2.2 CPU complement 与完整模型

`native-hybrid-v001` 证明 CPU complement 只加入
`FusedExpertsResult.routed_out`；shared expert、routed scaling、finalize、
residual 仍由原 `AscendMoERunner` 负责一次。

`native-hybrid-full-smoke-v003`（远端快照别名
`native-hybrid-full-v003`）在 `cudagraph_mode=NONE` 下完成：

- 五类 deterministic canary；
- 32-step Decode；
- 中途取消，且没有 unfinished request；
- 同进程 27-token reuse；
- 48 个 native runner 与 48 层 CPU bank 的真实命中。

这是一项 eager correctness 资格，不是请求性能资格。

### 2.3 为什么停止 PIECEWISE

PIECEWISE 的连续诊断没有停在第一个错误：

- v011 在一个 GDN 邻近区域生成非法 dynamic reshape；
- v013 绕过完整 GDN boundary 后，在独立的 HyperConnection
  `hc_combine` 再现同类问题，实际 32-row tensor 被绑定为 33 rows，并要求
  reshape 为 `[33, 4, 2560]`。

这证明根因属于冻结 PIECEWISE 栈的通用动态 scalar/range-to-shape ABI
绑定，而不是某一个 GDN、KV 或 routed-expert 算子。继续逐算子 opaque 或
shape patch 会变成没有边界的 runtime 重写。因此 Phase I 正式冻结，禁止继续
PIECEWISE 修补。单-page 场景的 `unsqueeze(0)` 也不能据此宣称 KV 或完整图
问题已解决。

Breakable 在冻结 Ascend platform 中被强制关闭。FULL/FULL_DECODE_ONLY 的
replay 只调用已捕获 NPU graph，不重新进入 Python quant-method `apply()`；当前
CPU branch 因而无法为每个 token/layer 创建新 work ID 和新结果。强行 capture
会复用 capture-time CPU output 或让模型退回 eager。

Phase I 结论：`ROUND18_NATIVE_HYBRID_WORKS_NO_GRAPH_RECOVERY`。

## 3. Phase II：fresh route 与 expert pool 经济性

### 3.1 数据

`native-route-trace-v016` 从当前正确 Native Hybrid 模型捕获 W1/W2/W4：

- 8,784 个 route events；
- 48 层；
- 512 logical experts；
- Top-10；
- W1/W2/W4 实际输出分别为 16/128/32；
- 输出有限 logprob，finish reason 为 length。

该运行包含 D2H route capture 和文件写入，只能用于工作集分析，不能用作性能。
数据复核还发现 W1/W2 使用同一 prompt：Prefill route 完全相同，W1 Decode
是 W2 Decode 的严格前缀。因此 W2 是延长 continuation；W4 才是不同输入的
交叉检查。

### 3.2 容量

相同类型的 vLLM Q32 完整运行时历史回执为约 47.34 GiB reserved。冻结
whole-card cap 为 55.5 GiB，余量约 8.16 GiB。

| Pool | 48层 routed payload | 同类 runtime 估算 reserved | 裁决 |
|---|---:|---:|---|
| Q32 | 14.0625 GiB | 47.34 GiB | 唯一已经构造过且未越 cap 的容量 |
| Q64 | 28.1250 GiB | 61.40 GiB | 超 cap |
| Q96 | 42.1875 GiB | 75.47 GiB | 超 cap |
| Q128 | 56.2500 GiB | 89.53 GiB | expert payload 自身已超过 cap |

Q50 的纯算术估算也只余约 0.25 GiB，无法容纳 graph pool、slot staging、
format workspace 与运行波动，不能作为安全候选。

### 3.3 工作集与 miss 流量

所有 Decode invocation 都只有 10 个 unique experts，Q32 在“单次调用容量”
上完全足够。Prefill 不同：

| Workload / phase | unique experts min / median / max | Q32 fit |
|---|---:|---:|
| W1/W2 Prefill | 10 / 83 / 160 | 13.54% |
| W4 Prefill | 72 / 111 / 182 | 0% |
| W1/W2/W4 Decode | 10 / 10 / 10 | 100% |

必须区分三种数字：

- invocation fit 只回答“放不放得下”，不提供 planner 或 load；
- W1 static-frequency 是唯一无需 current-token route readback 的预装策略，
  但 W2/W4 Decode 仍需约 1.659/1.320 GiB BF16 payload/输出 token；
- sequential LRU/Belady 先知道当前 route，Belady 还知道未来 route。W2 Q32
  Decode 的 0.635/0.387 GiB/输出 token 是在线/离线 oracle，不是冻结 FULL
  可运行策略。

因此 Decode route locality 本身没有否定 expert pool；真正否定当前实现的是
容量只允许 Q32，而 Q32 的完整结果需要动态 fill，冻结 FULL 又没有
current-token device planner。每 token 将 route D2H 给 Python、再 H2D 权重并
恢复模型会重建 Round16 的细粒度 eager break，违反本轮目标。

## 4. 固定 slot 与 backing format

### 4.1 v017 控制器失败

v017 在准入后把绝对测试文件路径错误地传给 `unittest` 模块加载器。容器在
NPU probe 之前退出，未构造 owner、未写权重。原始 FAIL 保留，随后用新 run
identity v018 修复控制器。

### 4.2 v018 真机结果

v018 只构造一个 layer0 Physical-Q32 owner：

| tensor | shape | dtype | NPU format |
|---|---|---|---:|
| w13 | `[32,2560,1280]` | BF16 | 29 |
| w2 | `[32,640,2560]` | BF16 | 29 |

allocator before/after/peak 均为 314,576,384 bytes。两个 parent 均为 NZ29。
探针在 mutation 前返回 `FAIL_FIXED_SLOT_POSTLOAD_FORMAT_INELIGIBLE`：

- `mutation_started=false`；
- 没有 publish generation；
- 没有 parent tensor assignment；
- 没有完整 parent rebuild；
- parent 指针未被替换。

冻结 `index_copy_` 的 NZ 路径会整理完整 destination format，并最终把完整结果
写回 parent；Round9 的真实 trace 也观察到每 projection update 的完整 32-slot
parent 事件。`npu_format_cast(...,29)` 是整 tensor 格式转换，不是 blob-to-slot
writer。官方 TorchNPU 合同还明确，转为 FRACTAL_NZ 后不支持修改 tensor 的
`view`、`slice` 等操作。因此不能靠 raw pointer、`as_strided`、slice copy 或
完整 parent 重建绕过。

### 4.3 其他格式

冻结源码存在 per-expert `w13_weight_list/w2_weight_list` consumer，原生 BF16
grouped matmul 可以在源码层接受 list。现有 list 却是从已经构造并 post-load
的 parent `unbind().clone()` 得到；Dynamic EPLB 还拒绝 EP1、只提供 D2D，
发布 `log2phy` 而不是 Q32 的 512-entry `expert_map`，且 publish 顺序不满足
weight-ready 后再 map-visible 的合同。

因此 independent list 是未来有依据的组件 seam，不是当前 S0 的 drop-in
writer。由于 fresh route/容量/graph planner gate 已经停止完整实现，本轮不为
它启动完整模型或请求实验。为了避免下次重新搭脚手架，本轮保留了一个带双重
设备开关的单层 TP1/EP1 离线探针：直接分配 32 对独立 BF16 expert tensor，
按 `w13 -> w2 -> ready -> 512-entry map` 发布，并限制为组件级手工
NPUGraph A/B/A。它通过离线合同测试，但状态严格保持
`OFFLINE_READY_DEVICE_NOT_RUN`；这不是未完成的 Round18 设备动作，也不是
consumer 或 Graph 已通过的证据。

W8A8 没有进入。它会改变精度，且在 planner/graph gate 已失败时新增量化质量
资格没有收益。预处理 NZ blob 也没有找到受支持的 Host blob→固定 parent slot
接口。

## 5. Graph-safe residency

冻结源码有一个邻近的 graph-external mutation 时序：Dynamic EPLB 的
`forward_before/forward_end` 位于模型图外，可以 wait、原位 copy expert storage
和更新 device mapping；FULL replay 前会同步 current stream。

这个 seam 只证明“后续 replay 前可存在受控 mutation”，没有解决：

1. EP1 Dynamic EPLB 被明确拒绝；
2. 现有 loader 只有 rank 间 D2D，没有 Host backing；
3. `log2phy` 不能表示 Q32 pool 的 `resident→slot / miss→-1`；
4. 当前 token route 在 FULL replay 前不会交给 Host planner；
5. 没有 device-side miss detection、transfer 与 weight→map 原子 publish；
6. 当前 NZ29 parent 没有固定 slot writer。

因此 Phase II 的准确终态为
`ROUND18_EXPERT_POOL_GRAPH_COMPAT_BLOCKED`。这不是“所有 expert pool 理论上
永久不可行”；重新打开必须同时提供：

- 无 per-token Host route readback 的 device-side current-token planner；
- 不重建 parent 的 format-aware slot writer；
- 所有 projection ready 后再发布同地址 mapping 的事件合同。

这三项构成新的 Ascend expert residency runtime，不是本轮既有接口的小修补。

## 6. 为什么没有请求性能数字

Round18 没有任何候选同时通过以下门：

- graph replay correctness；
- fresh work ID/result；
- capacity；
- 完整 expert contribution；
- fixed-slot write；
- graph-safe current-token planning。

因此没有启动 `NONE→GRAPH→GRAPH→NONE`，也没有启动
`S0→BEST18→BEST18→S0`。这避免把错误或不完整模型的速度当成收益。

Round17 S0 的约 262.8 ms/token、3.56 output tok/s 只在 benchmark 中标为历史
背景。Round18 没有 fresh candidate pair，不能声称比 S0 更快，也不能声称
实测更慢。

## 7. 本轮八个问题的回答

1. **原生 kernel 能否在 512 logical 下只算 Q32 partial？** 能，真实组件通过。
2. **不替换 AscendMoERunner，只加 CPU break，完整模型是否正确？** eager 正确。
3. **NONE→PIECEWISE→Breakable→FULL 回收多少 TPOT？** 没有合法 graph
   candidate，故没有 TPOT 回收数据；PIECEWISE ABI blocked，Breakable 不支持，
   FULL 与 CPU seam 不兼容。
4. **CPU break 是否是保不住 FULL 的决定性边界？** 对当前 seam 是。FULL replay
   不重新进入产生 CPU work 的 Python 路径。
5. **可用 HBM 下 route hit 是否足够？** Q32 Decode 的 invocation fit 为 100%，
   理论 locality 有价值；但 static plan miss 太多，LRU/Belady 不能在当前 FULL
   中运行，故没有可部署 policy。
6. **BF16/W8A8/预处理格式谁的 miss cost 最低？** 当前没有资格给出排序。
   BF16/NZ29 parent fixed-slot 被实测阻断；W8A8 未因 graph gate 失败而进入；
   预处理 blob 缺正式 writer。
7. **能否不做 per-token Host route readback而保持 Graph？** 当前冻结栈不能。
   邻近 mutation seam 存在，但 device-side current-token planner 缺失。
8. **最终是否在同资源 TP1 上超过 S0？** 没有可测候选，未形成晋级；S0 保持默认。

## 8. 资源与回收

最后一个设备块 v018：

- fresh 2×10 s admission PASS；
- 物理12/card6/chip0；
- 容器退出并删除；
- 退出后物理12无进程；
- 全机登记文件 FD/mmap 引用为空；
- 仅对 179 个已登记只读 bank/model 文件执行 `POSIX_FADV_DONTNEED`；
- fincore 确认释放 342,753,280 bytes；
- 没有全局 `drop_caches`；
- 没有修改外部训练/服务；
- timer 保持暂停。

## 9. 最终配置与项目结论

Round18 不改变默认配置。保留 Round17 S0：SGLang+KT、C128/MXFP4、Uniform
Q32 BF16/NZ29、promoted QSA/GDN、PLE ASYNC、H129、TP1、并发1。

本轮最重要的正结果是：Native partial 和最小 Hybrid 数学成立，而且无需替换
`AscendMoERunner`。最重要的负结果是：冻结 Ascend graph/runtime 没有把
current-token planning、固定 slot 格式写入和原子 mapping publish 组合起来。
在这些机制出现前，继续修 PIECEWISE、重建 NZ parent 或把 route 拉回 Host
都不会实现本轮目标。

最终状态：`ROUND18_COMPLETE_NO_PROMOTION`。
