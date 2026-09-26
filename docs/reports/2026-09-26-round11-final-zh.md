# KT-ASCEND Round11 最终报告

日期：2026-09-26  
终态：`ROUND11_HYBRID_PERFORMANCE_AND_BACKEND_QUALIFIED`

## 1. 结论与资格边界

Round11 已完成 T0–T6。最终选择保持 Round10 HFINAL，不启用任何 Round11 新性能候选：

| 项目 | 最终值 |
|---|---|
| 模型 | Qwen3.8-Flash-Next，revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| EXTEND 选路 | request total tokens `<129` 走 Hybrid，`>=129` 走 Layerwise |
| Decode | Hybrid |
| CPU Expert | Kunpeng C128，NUMA 2/3/4/7，各 32 worker |
| CPU bank | 只读 MXFP4 |
| NPU resident | Static Q32，BF16/NZ29 |
| scheduler chunk | 32 |
| PLE | ASYNC |
| Dynamic | OFF |
| QSA | 原 HFINAL provider |
| Layerwise | 原 shadow32；R2/R3/transpose-view/shadow320 均 OFF |
| 并行范围 | TP1、串行单请求 |
| 性能运行 | profiler OFF |

本轮“完成”指固定 Ascend＋Kunpeng 窄配置在已测范围内通过边界、核心归因、Dynamic 经济性、B 路线热点和最终集成验证。它不包含并发、TP2、Graph、MTP、HiCache、KV offload、FP8 KV、64K/128K、全量 CMMLU 或部署资格。

## 2. Hybrid 边界与较大 workload

原配置的真实边界保持不变：128 token 走 Hybrid，129/130 token 走 Layerwise。实验性 forced-Hybrid 用于测量能力边界，没有覆盖默认配置。

T1 的最终资格为：

- Hybrid prompt 在已测点上通过至 768 token；
- Decode 在已测点上通过 32/64/128/256 step；
- deterministic canary、重复性、cancel 和 same-process reuse 通过；
- 1024 token 运行时，guard 因 NUMA4 的 free＋保守 reclaim headroom 低于冻结保留量而只停止本项目容器；没有 OOM，没有自动重试，清理完整；
- 769–1023 未测试，因此不是已资格区间；
- 在已资格范围内没有观察到 Hybrid/Layerwise 性能 crossover；
- 默认阈值仍为 129，不能把 forced-Hybrid 结果写成默认策略已扩大到 768。

同输入 129/130 token 的 forced-Hybrid 与 Layerwise E2E 比值分别约 0.01503 和 0.01359。这证明当前 Layerwise 路径在该短边界上的固定成本很大；它不构成 Layerwise prepare 本身的加速证据。

## 3. Kunpeng Expert 深度归因与候选去留

T2 在冻结的真实 layer0、3-token C128/MXFP4 replay fixture 上运行三种独立组件进程：原生基线、通用计数器进程和只增加队列时间戳的 attribution 进程。三者每个进程均完成 3,000 次迭代、输出身份复核和完整清理。

| 指标 | 结果 |
|---|---:|
| 原生基线 parent 中位数 | 375.5825 us/group |
| attribution parent 中位数 | 366.8575 us/group |
| 原生并行 compute＋merge | 244.7000 us/group |
| compute＋merge 占 parent | 66.70% |
| enqueue/task construction | 43.2620 us/group，11.79% |
| pinned gather/metadata | 19.4310 us/group，5.30% |
| queue admission wait | 4.4000 us/group，1.20% |
| 128 worker runtime max/min | 1.010896 |
| instrumentation/基线 | 0.97677 |

队列等待和 worker 不平衡都不足以支持调度候选；共享量化也没有达到选择门。主成本位于现有 NEON＋SDOT MXFP4 compute/merge，继续下降需要新的、独立资格的 MXFP4 kernel，而不是一个受支持的单变量开关。T3 因此以 `KUNPENG_NO_PROMOTION_NO_BOUNDED_CANDIDATE` 收口，原 C128/MXFP4 backend 保留。

`perf_event_open` 返回 EPERM，且没有已验证的 uncore event mapping，所以 LLC-specific miss、内存带宽和 remote NUMA access 保持 `UNKNOWN`。本报告没有用 CPU utilization 替代这些指标。

## 4. Dynamic break-even

T4 完成 Static-A / Dynamic-A / Dynamic-B / Static-B 四块测量。Dynamic 的真实 48 层事务、mapping/mask publication、state fence、drain 和 cleanup 均通过；同模式 token 输出一致，finite logprob 和状态连续性通过。

| 量 | 结果 |
|---|---:|
| `M`：exposed migration | 13.824945036 s |
| `S`：Static steady Decode | 0.3964049145 s/token |
| `D`：Dynamic steady Decode | 0.4052964265 s/token |
| `N_break_even` | 无有限值 |

因为 `D >= S`，迁移以后每个 Decode token 也没有变快，迁移成本不可能通过更长 Decode 回收。终态是 `NO_FINITE_BREAK_EVEN`；Dynamic 事务正确性保留，默认继续 OFF。Round11 没有重新打开 NZ update 优化。

## 5. GDN/QSA/PLE 热点与 QSA 候选

v139/v140 两个独立 bounded-observer 模型进程完成真实 HFINAL 请求归因。父子 span 没有重复相加；QSA attention 已包含 indexer，PLE exposed wait 单独记录。

| 模块 | 中位 attributable / critical-path share |
|---|---:|
| QSA | 53.871% |
| GDN | 19.065% |
| PLE consumer | 0.278% |
| PLE exposed wait | 0.0177% |

QSA 是单一允许候选。query-aware vectorized sparse-GQA 在 v144 真实设备组件中通过：

- 最大 relative L2：`5.645496551606974e-05`；
- context 32/64/128 的 candidate/baseline 中位数比：`0.05691 / 0.05675 / 0.05649`；
- Decode 比值：`1.00506`；
- repeat hash、source identity 和 allocator baseline 均通过。

请求级晋级没有完成：

- v145 在创建容器前被旧 launcher source gate 拒绝，0 model start、0 generation；
- v146 修复 launcher 后加载了模型，但 runtime `context_reference_binding.py` 又独立要求原 QSA hash，server 在 ready 前退出，1 model start、0 generation；
- 两次原始失败均保留，guard ACK、容器删除和物理 12 回收通过；
- v146 是集成 source-gate 合同失败，不是候选数值失败；
- T5 的 3 个 model-start 额度已由 v139、v140、v146 用完，没有继续重试。

因此 QSA 结论是 `COMPONENT_ONLY_NO_PROMOTION`。组件速度不能代替请求收益，最终继续使用原 HFINAL QSA provider。

## 6. T6 最终四进程集成

T6 使用 v147/v148/v149/v150，顺序为 BASELINE-A / FINAL-A / FINAL-B / BASELINE-B。由于本轮没有候选晋级，四个标签实际执行完全相同的 HFINAL compute identity；这个设计用于检查跨进程复现和最终质量，不能将标签差异解释成代码加速。

每个进程固定 6 次 generation：一个 warmup、相同 input128 的 Decode32/64/128、再加质量或取消/复用用例。结果如下：

| Run | 模式 | Decode32 | Decode64 | Decode128 | 附加用例 | 结果 |
|---|---|---:|---:|---:|---|---|
| v147 | BASELINE-A | 27.528126 s | 40.323357 s | 72.148553 s | cancel＋reuse | PASS |
| v148 | FINAL-A | 27.045362 s | 38.628271 s | 69.102611 s | marker＋JSON | PASS |
| v149 | FINAL-B | 28.123643 s | 39.365460 s | 67.925124 s | cancel＋reuse | PASS |
| v150 | BASELINE-B | 27.979791 s | 41.227488 s | 70.165958 s | quote＋transform | PASS |

严格对账确认：

- Decode32/64/128 的 output token IDs 在四个进程中逐项完全一致；
- 所有 performance request 的 logprob 有限，finish reason、完成 token 数和 Hybrid path 合同通过；
- arithmetic、marker、JSON、quote、transform 五类 deterministic canary 通过；
- v147/v149 的 mid-prefill cancel、无 late commit、lease 回收和 same-process reuse 通过；
- 四个 supervisor 全部 PASS，guard 收到退出 ACK，owned container 删除，physical12 为空；
- 自动重试为 0。

聚合只表示相同代码的进程间复现：

| Decode | BASELINE 中位数 | FINAL 中位数 | FINAL / BASELINE |
|---:|---:|---:|---:|
| 32 | 27.753959 s | 27.584503 s | 0.993894 |
| 64 | 40.775423 s | 38.996865 s | 0.956382 |
| 128 | 71.157255 s | 68.513867 s | 0.962851 |

这些比值没有对应行为变化，不能报告为 Round11 候选收益。它们是独立模型进程的重复性/进程波动证据。

## 7. 资源与项目缓存退休

T6 四块的最大观测值为：

| 资源 | 峰值或下界 |
|---|---:|
| whole-card HBM | 28,783 MiB |
| owned-process HBM | 25,766 MiB |
| Host cgroup peak | 336.194 GiB |
| system available 最低值 | 1,700.208 GiB |

每次真实启动前都重新完成 2×60 秒、12 个 10 秒窗口的 admission，并核对物理12/card6/chip0、NUMA2/3/4/7、训练容器完整 ID/cgroup 和未知 NPU 进程。运行中某些 NUMA 的 `MemFree` 会因模型/checkpoint/bank 的只读文件页降到很低；这不是把 page cache 当匿名常驻内存。guard 使用冻结的运行时 headroom 合同，所有成功块都正常完成。

按用户要求，T5/T6 每个完成块在以下条件全部满足后执行项目缓存退休：

1. 本项目容器和进程已退出；
2. 原始证据已归档并完成 SHA 校验；
3. 只处理登记的 `/mnt/data1/kt-round9/banks` 与冻结 checkpoint 下 `.gguf/.safetensors`；
4. `/proc/*/fd` 与 mmap 审计确认没有活动引用；
5. 调用 `POSIX_FADV_DONTNEED`；
6. `fincore` 验证 179 个登记文件的 resident pages 全部归零；
7. 重新读取 NUMA MemFree。

最终四块的退休结果：

| Block | 退休前 resident pages | 退休后 | result SHA-256 |
|---|---:|---:|---|
| v147 | 69,296,699 | 0 | `789e77e112e9230c5869126143bd97d3e9057feedd02f48a730fffc4b12c8a41` |
| v148 | 69,971,797 | 0 | `f35e35854a04c528da250267e8a6f95c66614ed7c5780c8ec52f456e748c4a58` |
| v149 | 69,762,950 | 0 | `edbefd25335183feab6724eefb18d87c2d3dd96452726e9dc03677867275d4fe` |
| v150 | 69,694,518 | 0 | `39a4f6e0ce43b60a0253940fa575d1f82a4977bb39908daf320590f937c4d89c` |

退休前页数属于四个不同时间块，不能相加成同时峰值。全过程没有调用全局 `drop_caches`，没有修改训练或其他任务，也没有在 full hash 前提前回收页面。

## 8. 预算与最终回收

| 预算 | 使用 / 上限 |
|---|---:|
| model starts | 19 / 21 |
| generation attempts | 80 / 108 |
| abort controls | 4 / 5 |
| automatic retries | 0 |
| T2 component processes | 3 / 6 |
| T5 component processes | 3 / 9 |
| T6 model starts | 4 / 4 |
| T6 generations | 24 / 24 |

最终只读核验：物理 12 返回 `No process in device`；没有 Round11 容器，没有 Round11 controller/guard/hook 进程。定时任务保持暂停。本轮没有 push、merge、tag、reset 或 clean，dirty worktree 保留。

## 9. 关键证据

| 证据 | SHA-256 |
|---|---|
| `ROUND11_T1_RESULT.json` | 见 `ROUND11_FINAL_STATUS.json` |
| `ROUND11_T2_ATTRIBUTION.json` | 见 `ROUND11_FINAL_STATUS.json` |
| `ROUND11_T4_BREAK_EVEN.json` | 见 `ROUND11_FINAL_STATUS.json` |
| `ROUND11_T5_HOTSPOT_RANKING.json` | `e2f0a22f9593d449c7d09ccefa4cc35580da8a0b2b3a8dbc58c94334ad7db6f3` |
| `ROUND11_T5_QSA_COMPONENT_V144_ANALYSIS.json` | `83ea9975210eed86ac487d8f0010afa2b663768b2fa2bef4930931a3e9dbd787` |
| v145 原始失败归档 | `ed01875c3954a798b23615e7f5f26c6047a6a8a865e1836ece40e9557af0255a` |
| v146 原始失败归档 | `b8bba47e4adb04b7108185a561081a28315369e082d9349bb62c8f3cd959f534` |
| v147 归档 | `0834b7139b8480fbf083f16bdfb9491964371aa93e6efe2945e5cc33dc24131c` |
| v148 归档 | `26d65cf039b6082c5975be30905bf71470aea8480a18577e5df618f1457604c5` |
| v149 归档 | `ec80137516febc416db5fbebaf1daf5faa56fa9216aa53b2626e1f4da18cf130` |
| v150 归档 | `257e7bbc4348fb0fbd80c5df573d6c2dc223ffeace883ae71aac2254f94f2754` |

权威终态在 `artifacts/round11/ROUND11_FINAL_STATUS.json`。T6 严格比较在 `ROUND11_T6_FINAL_ANALYSIS.json`；T5 去留在 `ROUND11_T5_FINAL_DISPOSITION.json`；缓存回收在 `ROUND11_RESOURCE_RETIREMENT_RECEIPT.json`；实际次数在 `ROUND11_EXECUTION_BUDGET.json` 和 `EXECUTION_LEDGER.md`。
