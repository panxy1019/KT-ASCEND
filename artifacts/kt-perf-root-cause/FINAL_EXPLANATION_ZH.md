# KT 主线 W2 性能根因收口报告

日期：2026-09-29
终态：`KT_ROOT_CAUSE_CONFIRMED`
性能晋级：无，S0 保持默认

## 结论

当前 KT 的 3.x output tok/s 不是由 Kunpeng C128 的 MXFP4 Expert 算得慢造成的。直接连接 native enqueue、worker start/complete、TaskQueue sync、48 层 Expert parent 和 scheduler batch 后，C128 的真实服务时间约 **5.5 ms/token**，但它与 resident NPU 工作并行，在 Host 到达 consumer wait 前已经提前完成 **约 34–36 ms/token 的累计余量**。真正暴露到请求关键路径的 TaskQueue wait 只有 **0.13–0.15 ms/token**。

慢因是 48 层串行执行中多个 NPU/Host 区域的累计：Routed Expert parent 约 86.50 ms/token、QSA 完整 attention 约 66.08 ms/token、GDN 约 37.64 ms/token、其余 HyperConnection、router/shared expert、projection、norm/residual 和调度区域约 98.47 ms/token。不存在一个每 token 额外等待 100 ms 的 callback；也没有证据支持删除现有同步。

profiler-off 的真实部署速度来自两个 fresh S0 进程：W2 TPOT 分别为 **253.190 ms** 和 **262.896 ms**，串行吞吐分别为 **3.689** 和 **3.554 output tok/s**。诊断观察增加了 2.3%–11.6% 左右的墙钟，v509 全层观察更高，因此模块百分比只用于因果排序，部署速度仍以上述 profiler-off 数据为准。

## 固定对象

本轮没有改变模型数学或部署策略：

- Qwen3.8 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`；
- 48 层，36 GDN、12 QSA、512 routed experts/layer、Top-10；
- Kunpeng C128，NUMA 2/3/4/7，4×32 workers，MXFP4 bank；
- Uniform Q32 resident，BF16/NZ29；
- promoted GDN、QSA Vectorized Prefill / DIRECT_BF16_ROW Decode、PLE ASYNC；
- H129、chunk32、TP1、concurrency1；
- Dynamic、deferred、R2/R3、transpose-view、shadow320 均关闭。

生产 native SHA 为 `614ae60f…a6d`。诊断 native SHA 为 `bccee617…a62`，唯一语义差异是 env-gated、buffered 的 TaskQueue timestamp。它没有修改 worker、queue predicate、pending、同步、precision 或 route。

## 证据链

### 1. 两个 fresh 直接块

v506 和 v507 各执行固定六请求序列，W2 各得到 125 个稳态 Decode token、6,000 个完整层调用。两块结果一致：

| 指标 | v506 | v507 |
|---|---:|---:|
| Expert parent | 84.44 ms/token mean | 82.24 ms/token mean |
| CPU native task | 5.57 ms/token median | 5.53 ms/token median |
| TaskQueue exposed wait | 0.144 ms/token | 0.130 ms/token |
| CPU ready slack | 35.64 ms/token | 33.89 ms/token |
| sync 时 pending 非零 | 0/6000 | 0/6000 |

这一步把原来的 `CPU_SUBMIT/CPU_RESULT` 外层时间拆成了真实 worker service 与 consumer exposure。旧 Round11 单层 fixture 仍有参考价值：其中 CPU task 约 86.5 μs/token/layer；当前实际顺序约 116 μs/token/layer，约慢 1.34×。即使如此，完整请求仍不等它，因此不构成瓶颈。

### 2. v509 全层账本

v509 将 Host 父调用覆盖扩展到全部 36 GDN 和 12 QSA 层，保留 device event 的早/中/晚抽样。125 个稳态 token 的顶层父调用分区如下：

| 顶层区域 | mean ms/token | median ms/token | 诊断 wall 占比 |
|---|---:|---:|---:|
| Routed Expert，48 层 | 86.502 | 85.280 | 29.76% |
| QSA attention，12 层 | 66.076 | 63.419 | 22.74% |
| GDN，36 层 | 37.639 | 36.801 | 12.95% |
| PLE consumer | 1.949 | 1.926 | 0.67% |
| dense/router/norm/dispatch/scheduler 聚合 | 98.467 | 95.725 | 33.88% |
| scheduler wall | 290.633 | 293.131 | 100% |

QSA indexer 被正确标成 QSA attention 的 child，没有重复相加。CPU task 与 CPU wait 也嵌套在 Expert parent 内。`TOKEN_ACCOUNTING.csv` 保存每个 token 的原始分区。

“dense/router/norm/dispatch/scheduler 聚合”随后用相同 monotonic 时钟上的相邻父调用进一步分解，未再启动模型：

| 依赖区间 | mean ms/token | 诊断 wall 占比 |
|---|---:|---:|
| attention/GDN 结束 → 同层 Expert 开始 | 51.140 | 17.60% |
| Expert 结束 → 下一层 attention/GDN 开始 | 32.059 | 11.03% |
| scheduler 开始 → 首层 attention/GDN | 9.258 | 3.19% |
| 最后一层 Expert → scheduler 结束 | 5.171 | 1.78% |
| PLE 两侧间隙 | 0.840 | 0.29% |

这五类区间逐 token 恰好重建 98.467 ms，详见 `DEPENDENCY_GAP_ACCOUNTING.json/csv`。前两类分别绑定到 `_before_attention/_after_attention`、HyperConnection mix/combine、MLP 外层 router/dispatch、shared/routed combine、projection/norm，以及下一层准备。它们是准确的依赖位置，不是纯 Python CPU 时间；异步提交的 NPU 工作可能在后续边界才完成。因此本报告称其为 Host/NPU glue 与分段执行累计，不把它伪装成 83 ms 的可删除 Host 开销。

### 3. Device 与 Host 的关系

早/中/晚抽样中：

- GDN device mean 约 1.100 ms/call，Host mean 约 1.046 ms/call；
- QSA device mean 约 5.758 ms/call，Host mean 约 5.506 ms/call；
- Routed Expert sampled device mean 约 2.080 ms/call，Host mean 约 2.053 ms/call；
- resident NPU Expert device mean 约 0.439 ms/call。

Host/device 结果接近，说明这些父调用包含真实 NPU 工作和依赖完成，不是 Python 计时把异步 launch 当成完成。不同 stream 的 device interval 没有与 Host wall相加；这里只用它验证 region 性质。

### 4. 完整请求因果实验

本轮复用了同一 S0 数学和范围内已完成的 profiler-off 干预，而没有新增人为 sleep：

1. **QSA selector。** 两块组件中 candidate/baseline 约 0.17–0.20，selected set exact；完整模型真实命中全部 12 QSA 层，14 条输出 exact。W2 TPOT 从 266.511 ms 变为 267.615 ms，反而慢 0.41%，吞吐下降 0.63%。所以 selector 不是完整请求根因。
2. **resident-MoE graph。** 局部 Host 快 28.2%、device 快 33.7%；W2 TPOT 只改善 1.20%，吞吐只提高 0.73%，额外占用 984 MiB HBM，TTFT 退化 5.29%。所以 resident subcall 可以变快，但边界太小。
3. **当前 TaskQueue 边界。** 18,000 次 direct call 中 CPU 均在 wait 前 ready；当前顺序下将 CPU service 理想化为零，关键路径最多只删除约 0.13 ms/token 的可见等待。这直接否定了 CPU microkernel 作为当前主优化方向。

三个实验共同解释了过去“组件快、请求不快”的现象：优化的是大 parent 内的一个 child，或者已经被 overlap 隐藏的工作；完整 token 仍要经过其他几十个串行 region。

## 对 CPU、NPU、Host 和 overlap 的回答

**CPU 实际算多久？** 当前 W2 约 5.5–5.6 ms/token，约 0.116 ms/layer。它并非零，但远小于 253–263 ms 的部署 TPOT。

**CPU 什么时候 ready？** 在每层的 native TaskQueue consumer wait 之前。所有 18,000 次观测都没有 CPU tail；v509 累计提前量约 36.4 ms/token mean。

**Host 在等谁？** Expert sync 包装内到 native sync 前约 9.84 ms/token，主要对应前面的 device/stream 工作和 bridge；真正 native TaskQueue wait 仅约 0.13 ms/token。Host 还承担每层 dispatch、mask/remap、module glue 和大量小 NPU launch。

**NPU 算多久、空在哪里？** 精确统一 device critical path 不能通过跨 stream duration 相加得到。本轮安全地得到的是：GDN/QSA sampled device duration 与完整 Host parent 接近，Expert parent 中 NPU/dispatch/glue 明显大于 CPU wait；没有证据显示 NPU 长时间等 C128。相邻边界账把额外 83.20 ms/token 定位到每层 attention→Expert 和 Expert→下一层 attention 的 glue/依赖区间，但没有权限充分的统一 device timeline 来把它再精确拆成“Host 正在提交”和“NPU 正在执行/等待”的百分比，故没有捏造“纯 NPU 百分比”。

**CPU/NPU overlap 是否有效？** 有效。CPU service 已被 resident NPU 与本层其他工作覆盖。A2 精确同层 overlap 没有新的可用 CPU tail，因此不实现第二套 overlap。

## 还能省多少

本轮没有验证出可晋级修复，因此以下只写边界，不写性能承诺：

- CPU kernel 的当前可见上限约 0.13 ms/token，低于 0.1%；继续优化 C128 不划算。
- 完整 QSA parent 约 66 ms/token，是最大命名单模块；若未来能在保持数学/state 的前提下缩短整个 parent 20%，理论上约 13 ms/token，但 selector-only 已被证伪。
- GDN 约 38 ms/token，当前已是 Round14a promoted 路径，没有本轮新候选。
- Expert parent 约 87 ms/token，但 resident-only graph 的请求收益只有约 1%；下一步必须覆盖 router/dispatch/resident/merge 更完整的区域。
- attention→Expert 与 Expert→下一层 attention 两类依赖区间合计约 83.20 ms/token，说明更大的切口应跨越 HyperConnection、router/shared expert、projection/norm 和稳定 NPU 子区段。它不是 83 ms 的理论可全删空间；未来 segment 必须先区分必要 NPU 计算与可减少的 launch/glue，再完成独立正确性、state/cancel 与请求 ABBA。

## 去留和限制

S0 保持默认，没有候选晋级。停止投入以下方向：Kunpeng microkernel、TaskQueue timeout、删除同步、只优化 QSA selector、只捕获 resident-MoE 子调用。

本轮限制：

- v509 全层观察相对 profiler-off 有明显扰动，所以诊断模块占比不作为部署性能声明；
- 本轮没有保存新的 hidden/route detached fixture。当前结论使用真实 in-situ 48 层调用顺序，判断 consumer exposure 比离线回放更直接；未来若重新做 CPU kernel，应先保存当前 fixture；
- Round15 两项完整请求干预是每臂一个 fresh 进程、每 workload 两次 measurement，足以判断方向和数量级，不是四进程 ABBA；
- 没有硬件 PMU 权限，内存带宽、LLC miss 和 remote NUMA 仍为 UNKNOWN。它们不影响“CPU 已在 consumer 前完成”的时序结论。
- 98.47 ms residual 已全部定位到相邻依赖区间，但这些区间内的 Host active、device execution 与 stream wait 尚无统一绝对时钟细分；这不影响它们位于 48 层重复 glue 而非 C128 尾部的判断。

## 证据入口

- `TOKEN_ACCOUNTING.json/csv`：125 个稳态 token 的全层分区；
- `DEPENDENCY_GAP_ACCOUNTING.json/csv`：98.47 ms residual 的相邻依赖位置分解；
- `TASK_WAIT_CAUSAL_LINKS.json`：submit→task→sync→merge 直接连接；
- `CAUSAL_EXPERIMENTS.json`：QSA、resident graph 和 TaskQueue 边界干预；
- `OBSERVER_PERTURBATION.json`：诊断扰动边界；
- `REAL_WORKING_SET_REPLAY.json`：当前 in-situ 顺序与历史真实 fixture 的关系；
- `ROOT_CAUSE_CARDS.md`：按原因整理的决策卡；
- `RESOLVED_CODE_MAP.md`：实际文件、函数和 source/build identity。
