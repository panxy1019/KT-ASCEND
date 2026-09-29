# Round15 性能、资源与预算账本

## 1. 请求性能总表

所有数字来自 profiler OFF 的请求测量。每条线只有一个 fresh 基线进程和一个 fresh 候选进程，每个 workload 两次 measurement；因此表格用于方向性判断，不用于置信区间或 ABBA 稳定性声明。

### A：QSA full-set short-context

| Workload | Arm | E2E s | TTFT s | TPOT s | 串行 tok/s | req/s |
|---|---|---:|---:|---:|---:|---:|
| W1 | v341 baseline | 7.0964 | 2.8650 | 0.28208 | 2.2547 | 0.14092 |
| W1 | v342 candidate | 6.8340 | 2.7607 | 0.27155 | 2.3412 | 0.14633 |
| W2 | v341 baseline | 36.6190 | 2.7720 | 0.26651 | 3.4954 | 0.02731 |
| W2 | v342 candidate | 36.8494 | 2.8621 | 0.26762 | 3.4736 | 0.02714 |

A 在 W1 方向改善约 3.7%，在 W2 方向退化约 0.4–3.3%。主观察点没有收益，因此不晋级。

### B：resident-MoE Decode graph

| Workload | Arm | E2E s | TTFT s | TPOT s | 串行 tok/s | req/s |
|---|---|---:|---:|---:|---:|---:|
| W1 | v354 eager | 6.9774 | 2.9001 | 0.27181 | 2.2931 | 0.14332 |
| W1 | v360 graph | 6.7507 | 2.7849 | 0.26438 | 2.3701 | 0.14813 |
| W2 | v354 eager | 36.8694 | 2.7342 | 0.26878 | 3.4717 | 0.02712 |
| W2 | v360 graph | 36.6035 | 2.8789 | 0.26555 | 3.4969 | 0.02732 |

B 在 W2 的 E2E、TPOT、串行吞吐方向改善 `0.72%`、`1.20%`、`0.73%`，TTFT 退化 `5.29%`。结果低于原 10% Decode 门并触发 TTFT 保护问题，因此不晋级。

统一原始表见 [BENCHMARK.csv](../../../artifacts/round15/BENCHMARK.csv)。

## 2. HBM 与 Host

| 对照 | Baseline whole/owned HBM MiB | Candidate whole/owned HBM MiB | 解释 |
|---|---:|---:|---|
| A v341/v342 | 35048 / 25754 | 35047 / 25754 | 无候选 HBM 增长 |
| B v354/v360 | 35047 / 25754 | 36031 / 26738 | 图对象增加 984 MiB |

B 的 Host cgroup peak 没有被该 monitor 记录，权威结果明确标为 `UNKNOWN_NOT_RECORDED_BY_THIS_MONITOR`。全机 `MemAvailable` 最低值约 1792.21/1792.18 GiB，只用于主机健康观察，不能替代 owned cgroup 峰值。

组件 allocator 中，v357 capture 前 allocated/reserved 为约 314.58/553.65 MB，capture 后约 314.60/448.79 MB，reset 后 allocated 为 314.58 MB；post-reset 增长 4096 bytes。完整模型额外 984 MiB 高于单层组件，是 48 层图和运行时池的合计结果。

## 3. 实际预算

| 预算项 | 硬上限 | 实际使用 | 剩余 |
|---|---:|---:|---:|
| 模型进程 | 20 | 13 | 7 |
| Generation attempts | 248 | 83 | 165 |
| Abort controls | 8 | 5 | 3 |
| 组件进程 | 20 | 4 | 16 |
| NPU device-hours | 64 | 0.837265 | 63.162735 |

“剩余”只表示未消耗额度，不授权在 Round15 已冻结后继续探索。失败启动、控制失败和资源失败都按合同计数；没有通过删除失败记录让账本变好看。

## 4. Run 账本

| Run | 阶段 | 结果 | 模型 / gen / abort / 组件 | NPU h |
|---:|---|---|---:|---:|
| 331 | P0 | 预算 stage gate，模型未启动 | 1 / 0 / 0 / 0 | 0 |
| 333 | P0 | 请求完成但缺五桶 producer | 1 / 4 / 0 / 0 | 0.07014 |
| 335 | P0 repair | 测量 salvage，原控制 FAIL 保留 | 1 / 4 / 0 / 0 | 0.07431 |
| 338 | P0 repair | 完整 PASS | 1 / 4 / 0 / 0 | 0.07152 |
| 339 | A component | PASS block A | 0 / 0 / 0 / 1 | 0.00668 |
| 340 | A component | PASS block B | 0 / 0 / 0 / 1 | 0.00667 |
| 341 | A request | baseline PASS | 1 / 14 / 1 / 0 | 0.10385 |
| 342 | A request | candidate PASS/no gain | 1 / 14 / 1 / 0 | 0.10228 |
| 347 | B component | int64 API feasibility PASS | 0 / 0 / 0 / 1 | 0.01147 |
| 348 | B control | manifest mismatch，模型未启动 | 1 / 0 / 0 / 0 | 0 |
| 352 | B resource | NUMA4 cold-load guard FAIL | 1 / 0 / 0 / 0 | 0.04063 |
| 354 | B request | eager baseline PASS | 1 / 14 / 1 / 0 | 0.09525 |
| 355 | B control | legacy source gate，模型未启动 | 1 / 0 / 0 / 0 | 0 |
| 356 | B integration | 请求完成但图零命中 | 1 / 14 / 1 / 0 | 0.09666 |
| 357 | B component repair | int32 production component PASS | 0 / 0 / 0 / 1 | 0.01070 |
| 358 | B integration | first capture self-check FAIL | 1 / 1 / 0 / 0 | 0.05184 |
| 360 | B request | graph candidate PASS/small gain | 1 / 14 / 1 / 0 | 0.09527 |

事件逐项求和与 `used` 完全一致。权威账本见 [ROUND15_EXECUTION_BUDGET.json](../../../artifacts/round15/ROUND15_EXECUTION_BUDGET.json)。

## 5. 最终经济判断

A 的组件节省没有带来 W2 请求收益。B 为一个约 0.4 ms 的 resident 子调用增加 48 层图对象，换得约 0.12 ms 的组件节省，但只转化为约 0.7% 的 W2 E2E/吞吐改善，同时增加 984 MiB HBM并损害 TTFT。当前配置下，两项都不值得打开默认开关。

这不否定更大 NPU 图区域的潜力；它说明下一步只有在能捕获更大、确实位于 critical path 且不跨 CPU/状态依赖的区域时，才值得重新立项。
