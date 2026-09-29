# KT-ASCEND Round17 最终性能结题报告

日期：2026-09-29
终态：`ROUND17_COMPLETE_NO_NEW_PROMOTION`
性能晋级：无
默认配置：继续使用冻结 S0

## 1. 最终回答

当前单卡 TP1、单请求 W2 的 fresh 速度是 **259.93 / 265.63 ms/token**，完整串行吞吐是 **3.5946 / 3.5206 output tok/s**。两块中位的中位为 **262.78 ms/token、3.5576 output tok/s**。

最后一轮没有找到一个既保持模型数学和状态、又能在现有架构内达到晋级门的实现：

- 完整 QSA parent 确实是主要成本，两个诊断块为 **70.08–73.42 ms/token**；但冻结栈中成熟 fused-attention 原语只覆盖约 **8.96–9.59 ms/token** 的 score/softmax/PV core。即使这部分成本理想归零，完整 parent 也只改善 **12.4%–13.4%**，低于 20% 组件门。
- 更大的 decoder segment 必须跨越 HyperConnection、投影/残差、router、resident/CPU merge 以及 QSA/GDN 可变状态。当前约 **89–98 ms/token** 的相邻依赖区间包含大量必要 NPU 计算，不是可整体删除的 Host glue。现有 runtime 没有能安全承载这些 state generation、动态 route 和 CPU work ID 的 bounded graph backend。

所以当前约 3.6 tok/s 的主要限制仍是 **48 层串行 NPU 主干计算，加上逐层 Host dispatch、状态处理和执行边界累计**。它与 NPU 有直接关系，也与当前 Host/NPU 分段方式有关；它不是 C128/MXFP4 CPU Expert 算力不足。若还要获得显著提升，需要新的模型专用 fused-QSA backend 或 state-aware decoder graph/runtime，而不是继续优化一个小子算子。

## 2. 冻结配置

本轮没有改变产品数学或部署策略：

- Qwen3.8 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`；
- 48 层，其中 36 GDN、12 QSA；每层 512 routed experts、Top-10；
- Kunpeng C128：NUMA 2/3/4/7，4×32 workers，MXFP4 只读 bank；
- Ascend physical12/card6/chip0，Uniform Q32 resident，BF16/NZ29；
- promoted GDN、QSA Vectorized Prefill / DIRECT_BF16_ROW Decode、PLE ASYNC；
- chunk32、H129、TP1、并发1；
- Dynamic、deferred、R2/R3、transpose-view、shadow320 和 Round15 graph candidate 均关闭。

本轮仍冻结 CPU Expert、TaskQueue、更多 resident、Layerwise、vLLM、W8A8、MTP、并发和长上下文方向。

## 3. Fresh profiler-OFF 请求基线

两个独立模型进程使用各自的 2×10 秒准入、缓存退休和容器生命周期。每块 W1/W2/W4 各执行 1 warmup＋2 measurement，并有 1 条冻结 canary。

| Block | Workload | TTFT | TPOT | E2E | serial output tok/s |
|---|---|---:|---:|---:|---:|
| v519 S0-A | W1 128/16 | 2.552 s | 270.235 ms | 6.606 s | 2.4222 |
| v519 S0-A | W2 128/128 | 2.598 s | 259.926 ms | 35.609 s | 3.5946 |
| v519 S0-A | W4 64/32 | 1.300 s | 259.290 ms | 9.339 s | 3.4266 |
| v520 S0-B | W1 128/16 | 2.686 s | 273.251 ms | 6.785 s | 2.3583 |
| v520 S0-B | W2 128/128 | 2.622 s | 265.632 ms | 36.357 s | 3.5206 |
| v520 S0-B | W4 64/32 | 1.306 s | 263.024 ms | 9.460 s | 3.3828 |

所有 9 个性能请求/块都生成完整固定长度；finite logprob、QSA provider、GDN binding、CPU/NPU path 和 PLE 事件由原 driver 核对。canary 生成 4 token 后命中冻结 stop token，未与固定长度性能样本混算。

本轮 fresh 值比上一轮某些历史块略慢，但仍落在项目已观察到的 253–266 ms/token 范围。正式结论只使用 v519/v520；没有用跨轮旧数字作候选配对。

## 4. QSA 完整 parent 子账

v512/v514 各自完成 6 个诊断请求。分析只选 W2 measurement 的稳态 Decode，移除首尾 token，得到每块 125 个 batch。每 token 的实际调用次数固定为 12 QSA、36 GDN、48 Expert/decoder layer。

| 12 层合计 | v512 | v514 |
|---|---:|---:|
| scheduler diagnostic wall | 317.561 ms | 297.754 ms |
| QSA parent Host | 73.417 ms | 70.077 ms |
| QSA parent device | 73.666 ms | 70.327 ms |
| indexer Host | 35.772 ms | 34.125 ms |
| sparse core Host | 9.128 ms | 8.962 ms |
| sparse core device | 9.587 ms | 9.420 ms |
| output projection Host | 0.586 ms | 0.545 ms |
| parent 未被 immediate child 覆盖 | 19.519 ms | 19.053 ms |

诊断 wall 高于 profiler-OFF 请求，因此只用于模块归因。Host 与 device 的 parent 数值接近，证明 QSA 是实际 device critical work，而不是单纯的 Python 时间。

### 4.1 A1 为什么不实现

`torch_npu.npu_fused_infer_attention_score` 等成熟 Ascend 原语可以覆盖 score/softmax/PV，但不能替代模型特定的压缩 indexer、signed ranking、causal selection、GQA mapping、selected K/V 准备和 state update。

对两个块分别计算硬上限：

- Host：9.128/73.417 = **12.43%**；8.962/70.077 = **12.79%**；
- device：9.587/73.666 = **13.01%**；9.420/70.327 = **13.39%**。

这是把 sparse core 假设成零成本的极限，仍达不到 20% parent 门。真实适配只可能更低。因此没有写一个注定无法晋级的 A1 wrapper。

### 4.2 A2 为什么不实现

超过 20% 必须覆盖 indexer＋K/V materialization＋state＋sparse attention 的整体。冻结栈没有成熟原语提供这组模型语义。过去的 selector-only 因果实验已经表明：组件快 5–6 倍且 selected set exact，完整 W2 TPOT 反而慢 0.41%，吞吐下降 0.63%。

A2 的有效形态是新的 Qwen3.8 专用 Ascend fused-QSA backend，需要独立数值、state、fallback、取消和 allocator 工程。它不再是最后一轮允许的 bounded local fusion。A 线因此以 `A_NO_BOUNDED_PARENT_CANDIDATE` 收口；这不是断言未来完整后端无收益。

## 5. 更大 exact NPU segment

两块相邻依赖位置为：

| Host 同钟域区间 | v512 | v514 |
|---|---:|---:|
| attention/GDN 结束 → 同层 Expert 开始 | 57.718 ms | 51.984 ms |
| Expert 结束 → 下一层 attention/GDN | 40.071 ms | 36.705 ms |
| 最后 Expert → layer end | 0.408 ms | 0.385 ms |

这些区间不是可直接删除的 89–98 ms。它们包含必要的矩阵计算、norm、residual、HyperConnection、router、resident/shared expert、merge、状态访问和 queued NPU completion。

### 5.1 B1 单层 segment

Round15 的 resident-only graph 是直接因果证据：局部 Host/device 分别快约 28%/34%，但 W2 TPOT 只改善 1.20%，完整串行吞吐只增加 0.73%，并增加 984 MiB HBM。它证明小 graph 能跑，也证明被捕获 child 不是请求限制。

本轮没有测出一个 launch-only 子区间，其可消除成本能够达到完整层 15%。若把全部依赖区间当成“glue”，就会把真实 NPU 计算错误地计作 graph 收益，并可能把同步移出测量边界。因此 B1 没有进入实现。

### 5.2 B2 跨子模块 segment

有意义的 B2 必须建立新的 state-aware segment runtime：

- 为 48 层不同权重建立有限 graph cache；
- 正确更新 KV/QSA/GDN/PLE state address generation；
- 每 token 刷新 hidden、route IDs/weights 和有效长度；
- CPU Expert 保持图外，非空 CPU 分支每 token 产生新的 work ID；
- CPU result ready 后再 replay 依赖段；
- cancel/stale epoch 不 late commit；
- capture 不推进真实递归 state；
- partial state write 后不能无回滚地 eager fallback；
- graph pool、固定 buffer 和 metadata cache 有界且可关闭。

这套架构不存在于冻结 runtime。直接在本轮补齐会同时改变 state ownership、buffer lifecycle、graph cache 和执行顺序，无法作为一个受控候选。B 线以 `B_NO_SAFE_BOUNDED_EXACT_SEGMENT` 收口。

本轮没有逐 kernel profiler，所以 kernel launch count 保持 `UNKNOWN`。该 UNKNOWN 没有写成零，也不是停止的单一原因；去留依据是完整 parent 的 device 覆盖、既有 graph 干预和 exact state/CPU 边界所需的新架构。

## 6. CPU 与 NPU 的最终判断

CPU Expert 的直接证据仍然成立：C128/MXFP4 服务约 5.5 ms/token，但 TaskQueue 真正暴露约 0.13–0.14 ms/token，CPU 结果在 consumer 前提前约 34–36 ms ready。继续优化 microkernel、timeout 或删除 sync 不会让 260 ms/token 请求显著变快。

NPU 与分段执行是主限制，但需要准确表述：

1. QSA、GDN、resident Expert 和 dense backbone 有大量必要 NPU 计算；
2. 48 层串行结构放大每层的 metadata、Host launch、state、projection/norm/residual 和 stream dependency；
3. Host/device parent 接近，不能把所有 Host wall 解释成 Python；
4. 小范围 kernel 或 graph 的局部加速会被其他 parent 淹没；
5. 下一次有机会达到两位数收益的切口必须覆盖完整 QSA 或建立更完整的 exact decoder execution backend。

所以“慢与 CPU/NPU 是否有关”的答案是：**与当前单卡 NPU 主干及其逐层执行组织强相关；与 Kunpeng Expert 算力无主要因果关系。**

## 7. 为什么没有请求 ABBA

冻结合同规定：A 必须完整 QSA parent 两块至少快 20%，B 必须完整 layer 两块至少快 15%且 launch 显著下降，才进入请求 ABBA。本轮两个方向都没有生成能满足该门的 bounded implementation。

因此：

- 没有执行虚构的 candidate ABBA；
- 没有把“没实现”写成“请求无收益”；
- 没有用闲置设备次数探索第三条路线；
- 没有组合 A+B；
- S0 继续是最终配置。

这符合 `ROUND17_COMPLETE_NO_NEW_PROMOTION`：完整 parent/segment 都有量化去留结论，且没有误写性能目标达成。

## 8. 资源与生命周期

| 指标 | v519 | v520 |
|---|---:|---:|
| whole-card HBM peak | 28,933 MiB | 28,933 MiB |
| observed board baseline | 3,282 MiB | 3,282 MiB |
| estimated owned HBM peak | 25,651 MiB | 25,651 MiB |
| Host cgroup peak | 341.85 GiB | 341.24 GiB |

Host 峰值主要包含只读 bank/checkpoint file cache，不能等同 anon working set。每个正式块结束后先归档和哈希，再检查引用并对登记文件做 POSIX_FADV_DONTNEED。最终：

- Round17 自有容器剩余 0；
- 物理 12 设备进程剩余 0；
- 未恢复 timer；
- 未使用全局 drop_caches；
- 未停止或修改外部训练/服务。

## 9. 失败与修复透明度

本轮保留 v510、v513、v515、v517、v518 原始 FAIL。它们分别对应 shared provider 重复包装、block identity、预算 source gate、容器路径和 PYTHONPATH。每次后续验证都有代码或身份变化，没有 unchanged retry。

v519 在 10/10 请求完成并写出 PASS 结果后，runner 的打印语句误读 `summary` 而非 `summaries`。请求结果仍完整有效，容器已退出；该 postcheck 被修正后 v520 从启动到哈希全程 PASS。详见 `FAILURE_LEDGER.md` 和 `EXECUTION_LEDGER.json`。

## 10. 项目结题建议

当前配置可作为已验证的窄配置基线，但不应宣传为高吞吐部署。最后一轮没有新的性能晋级，`PERFORMANCE_OBJECTIVE=NOT_ACHIEVED`。

如果未来重新投入，只建议二选一并按新项目立项：

1. **完整 fused-QSA backend**：同时覆盖压缩 indexer、selected K/V、sparse GQA 和 state；
2. **state-aware exact decoder runtime**：让更大的 NPU 区段可捕获，同时把 CPU work 明确留在图外。

任何新项目都应先证明完整 parent/layer 的两位数收益，再加载整模型。继续优化 selector、resident-only child、C128 microkernel、TaskQueue 或小同步点的预期回报已经被证据否定。

## 11. 证据入口

- `FRESH_S0_RESULTS.json`、`BENCHMARK.csv`：本轮 fresh profiler-OFF 基线；
- `QSA_PARENT_ACCOUNT.json`：两块完整 QSA parent 子账；
- `A_QSA_DISPOSITION.*`、`B_SEGMENT_DISPOSITION.*`：候选去留；
- `REQUEST_DECISION.md`：请求级决定；
- `EXECUTION_LEDGER.json`、`FAILURE_LEDGER.md`：运行与修复；
- `SOURCE_BUILD_IDENTITY.json`：源码与构建身份；
- `RESOURCE_RETIREMENT_RECEIPT.json`：最终资源；
- `raw/`：原始 result、analysis、monitor、admission、远端 SHA；
- `EVIDENCE_SHA256.json`：本地交付哈希。

上游机制参考：Ascend `npu_fused_infer_attention_score` 文档与 op-plugin 测试用于确认成熟原语的覆盖边界；它们不被当作本地已验证性能。Round15 graph 原始结果用于因果筛选，不被升级为 Round17 候选。
