# Round17 B 线：更大 exact NPU decoder segment 去留

日期：2026-09-29
结论：`B_NO_SAFE_BOUNDED_EXACT_SEGMENT`
默认配置：保留 S0，不启动 segment 请求 ABBA。

## 两块依赖区间

每个 W2 Decode token 覆盖 48 个 decoder layer。相同 Host 单调时钟上的相邻 parent 边界给出：

| 区间 | v512 | v514 |
|---|---:|---:|
| attention/GDN 结束 → 同层 Expert 开始 | 57.718 ms/token | 51.984 ms/token |
| Expert 结束 → 下一层 attention/GDN 开始 | 40.071 ms/token | 36.705 ms/token |
| 最后一层 Expert → layer 结束 | 0.408 ms/token | 0.385 ms/token |

这些数值是依赖位置，不是“可删除的 Python 时间”。区间内包括 HyperConnection mix/combine、norm、projection、residual、router、resident expert、CPU/NPU merge、shared expert、QSA/GDN 状态工作，以及异步排队后在边界完成的 NPU 计算。

## B1：单层较大 segment

Round15 已证明小区域 graph 能正确运行：resident-only MoE graph 的组件 Host 快 28.17%、device 快 33.68%，输出 exact；但完整请求 W2 TPOT 只改善 1.20%，串行吞吐只提高 0.73%，并增加 984 MiB HBM。它说明 graph 技术本身有效，也说明只捕获 resident child 的边界太小。

本轮 profile 没有找出一个可独立删除、并达到完整层 15% 门的 launch-only 区域。QSA parent 的 Host/device 时间接近，GDN/Expert 也有真实设备工作；把上述 89–98 ms 整体当成 Host launch 收益会把必要计算误写成优化空间。

因此没有为了满足形式而把更多相邻 Python 调用拼进 graph。该实现既没有可验证的 15% 完整层上限，也可能只是把同步移到 segment 外。

## B2：跨子模块 exact segment

真正更大的 exact 方案需要同时解决：

1. 48 层各自不同的权重与有限 graph key；
2. 36 个 GDN 层和 12 个 QSA 层的可变 state 地址、request epoch 和 generation；
3. 每个 Decode token 更新 hidden、route IDs/weights、有效长度和状态，而不重用 capture 数据；
4. CPU Expert 保持图外，每个非空 CPU 分支产生新的 work ID；
5. CPU result ready 后才能进入依赖 segment，取消或 stale epoch 不能 late commit；
6. capture/warmup 不能推进真实递归状态，失败后不能在已部分写状态上直接 eager 重跑；
7. graph pool、固定 buffer 和 metadata cache 有界并可 close。

当前 runtime 只有小区域 graph prototype，没有这套 state-aware segment runtime。把它补齐是新的 decoder execution backend 项目，不是两个局部 segment 设计之一。最后一轮直接写大图会同时改变 graph cache、state ownership、buffer lifecycle 和执行边界，无法保持单变量与可审计性。

## launch count 与 UNKNOWN

本轮 observer 保存 parent/child 调用次数（每 token 48 layer、12 QSA、36 GDN），没有打开会显著扰动 Decode 的逐 kernel profiler，因此 kernel launch count 保持 `UNKNOWN`，没有被填成零。B 线不以“缺 launch count”为停止理由；去留依据是：

- 已测 parent 包含真实 device work；
- 既有 graph 干预给出“小 child 快、请求几乎不动”的因果结果；
- 尚无可定义、能预期达到完整层 15% 的 exact segment；
- 扩大到有意义边界必须先建立新的 state-aware graph runtime。

## 结论边界

- 本轮拒绝 B1/B2 两个有界设计，没有候选进入层级 probe、model smoke 或请求 ABBA。
- 这不证明 runtime-wide exact decoder graph 不可能；它明确了下一项目必须先解决的架构合同。
- CPU Expert 继续在图外，旧 CPU overlap 结论不变；未跨层 deferred，未删除同步。

主要机器证据：`B_SEGMENT_DISPOSITION.json`、两块 `round17-analysis.json`、上一轮 `artifacts/kt-perf-root-cause/CAUSAL_EXPERIMENTS.json`。
