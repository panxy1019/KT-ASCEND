# KT W2 根因卡

所有时延均针对冻结 S0 的 W2：128 input / 128 actual output、TP1、单请求。部署速度以 profiler-off v403/v404 为准；下表的模块时间来自诊断块，用于因果归属和排序。v509 全层观察的 scheduler wall 中位数为 293.13 ms/token，高于 profiler-off 的 253.19/262.90 ms，因此不能把诊断百分比直接当作部署 SLA。

## 原因卡 1：Kunpeng CPU Expert 不是当前请求瓶颈

**结论：已确认排除。**

两块独立 fresh 诊断加一个全层块给出一致结果：

| 块 | CPU native service | TaskQueue 暴露等待 | CPU 在 Host sync 前已完成的余量 |
|---|---:|---:|---:|
| v506 | 5.570 ms/token | 0.144 ms/token | 35.638 ms/token |
| v507 | 5.527 ms/token | 0.130 ms/token | 33.888 ms/token |
| v509 | 5.574 ms/token | 0.132 ms/token | 36.045 ms/token |

三个块合计 18,000 个层调用；`pending_nonzero=0`，`cpu_tail_after_host_sync_start=0`。CPU 的约 5.5 ms 服务时间嵌套在 resident NPU 和本层其他工作内，不能再加到 token 墙钟。将 CPU kernel 理想化为零耗时，在当前 consumer 边界最多只能删除约 0.13 ms/token 的可见 TaskQueue wait。

这也解释了为什么旧 Round11 组件里 native compute 占单组约 66.7%，却不能推出完整模型 CPU-bound。旧 fixture 的父调用只有约 0.367 ms/group；它回答“CPU group 内部花在哪里”，没有回答“完整请求是否在等它”。

## 原因卡 2：Routed Expert 是重要区域，但其主要问题不在 C128 算术

**结论：显著、混合归因。**

v509 的全 48 层 routed-expert parent：

- mean 86.50 ms/token；median 85.28 ms/token；
- 占诊断 scheduler wall 29.76%；
- Host dispatch 约 14.76 ms/token；
- Host sync 约 13.53 ms/token，其中到 native sync 前约 9.84 ms/token，TaskQueue 真正等待约 0.13 ms/token；
- merge 约 1.21 ms/token；
- 其余约 57.00 ms/token 是 resident NPU expert、mask/remap、router/combine 和相关 Host/NPU glue。

早/中/晚三层的 resident NPU device event mean 约 0.439 ms/call，Routed Expert device parent mean 约 2.080 ms/call。Device 与 Host parent 大致同量级，表明父调用包含真实 NPU 工作和依赖等待，而不是纯 Python 提交成本。

Round15 的 resident-only graph 把该子调用 Host 中位数从 0.483 ms 降到 0.347 ms、device 从 0.364 ms 降到 0.241 ms，但 profiler-off W2 TPOT 只改善 1.20%，串行吞吐只提高 0.73%，同时额外使用 984 MiB HBM，W2 TTFT 退化 5.29%。这确认 resident 子调用可优化，也确认它不是完整请求的主导单点。

## 原因卡 3：QSA 完整 attention 是最大命名单模块，selector 不是完整根因

**结论：完整 QSA 值得继续研究；只优化 selector 不值得晋级。**

12 层 QSA attention 全父调用：

- mean 66.08 ms/token；median 63.42 ms/token；
- 占诊断 scheduler wall 22.74%；
- sampled device mean 5.758 ms/call，与 Host mean 5.506 ms/call 接近；
- QSA indexer mean 2.990 ms/call，但它嵌套在 attention parent 内，不能重复相加。

Round15 对相同数学的 full-set selector 做了直接干预：24 个组件行中 candidate/baseline 为 0.170–0.198，selected set exact，完整模型真实命中 12 个 QSA 层，14 个输出序列 exact。然而 profiler-off W2 TPOT 从 266.511 ms 变成 267.615 ms，串行吞吐下降 0.63%。

所以“QSA sample 最大”不能简化为“再快一点 indexer 就能救整请求”。剩余可操作对象是整个 QSA region：QKV、KV state/index、selected K/V gather、sparse attention、projection 及其连续 launch 区域。任何下一候选都应测完整 QSA parent，而不是只测 selector microbenchmark。

## 原因卡 4：36 层 GDN 是稳定的累计 NPU 成本

**结论：支持累计 NPU backbone 根因。**

36 层 GDN：

- mean 37.64 ms/token；median 36.80 ms/token；
- 占诊断 scheduler wall 12.95%；
- sampled device mean 1.100 ms/call，Host mean 1.046 ms/call。

Host/device 的接近说明这部分主要由设备工作决定，并非日志或 callback 空转。Round14a 已晋级当前 GDN 实现，本轮没有证据支持恢复被拒绝的 swish parent，也没有发现新的小范围修复。

## 原因卡 5：其余 98.47 ms/token 位于可重建的层间依赖区间，不是单个未知 sleep

**结论：代码边界明确，尚未细拆为每个 kernel。**

从 scheduler wall 中扣除全层 GDN、QSA、Routed Expert 和 PLE consumer 后，剩余 mean 98.47 ms/token、median 95.72 ms/token，占诊断 wall 33.88%。使用相同 monotonic 时钟上 97 个不重叠父调用的开始/结束位置，可将其完整分为：

- attention/GDN 结束到同层 Expert 开始：51.14 ms/token；
- Expert 结束到下一层 attention/GDN 开始：32.06 ms/token；
- scheduler 到首层 attention：9.26 ms/token；
- 最后一层 Expert 到 scheduler 结束：5.17 ms/token；
- PLE 两侧：0.84 ms/token。

这些位置对应冻结模型代码中的：

- HyperConnection mix/combine；
- attention 前后 norm/residual；
- MLP 外层 router、dispatch/combine、shared expert 与 scaling；
- 未包含在 attention/GDN parent 内的 projection；
- sampler、scheduler 间隙；
- v509 的额外 Host 观察开销。

这是 48 层重复的小段累积，不是证据缺失后虚构的 `100 ms callback`。v506/v507/v509 都没有发现约 100 ms/token 的 native wait；真正的 TaskQueue wait 小于 0.15 ms/token。`DEPENDENCY_GAP_ACCOUNTING` 对每个 token 都验证 known parents + named gaps 等于 scheduler wall，没有用平均数相减拼账。

依赖位置已明确，但这些 gap 不是纯 Host active time：异步 launch 的 NPU 工作可能在后续依赖中完成。要减少它，需要以完整 decoder region 为边界做更大 NPU segment/fusion，并进一步区分必要设备计算与可消除 launch/glue。Round15 的 resident-only segment 已说明只捕获一个小子调用收益不足。

## 原因卡 6：为什么当前约 3.x tok/s

profiler-off 两个 fresh S0 块：

| 块 | W2 TPOT | 串行 output tok/s |
|---|---:|---:|
| v403 | 253.190 ms | 3.689 |
| v404 | 262.896 ms | 3.554 |

当前每个 token 串行穿过 48 个 decoder layer。CPU Expert 已被 resident NPU 工作覆盖；主要墙钟来自 repeated NPU backbone、Expert 内的 NPU/dispatch/glue、QSA、GDN 和层间主干。单个局部 kernel 即使快 20–30%，其完整请求占比也太小，所以此前多个组件 PASS 没有转化为请求收益。

这与硬件的关系是：

- **CPU 性能不是当前限制。** C128 计算在 consumer 到达前完成。
- **NPU 有效计算是主要组成。** GDN/QSA sampled device time 与 Host parent 接近，Expert 中 resident NPU 和流依赖也明显大于 CPU exposed wait。
- **Host 调度有累计成本。** 48 层 dispatch、mask/remap、module glue 和大量小 launch 累积成几十毫秒，但不是一个可以直接删除的同步。
- **当前 CPU/NPU overlap 已经有效。** 再做同层 overlap 没有足够 CPU tail 可隐藏。

## 去留

- 保留 S0 默认配置。
- 不投入 Kunpeng microkernel、TaskQueue timeout 或“删除 sync”方向。
- 不晋级 Round15 selector 和 resident-only graph。
- 下一项若立项，应以完整 QSA region 或跨多个稳定 NPU 子区段的 exact segment 为边界；必须保持当前数学、state、取消和 same-layer merge。
