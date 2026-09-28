# KT-ASCEND Round14a 最终报告

日期：2026-09-28  
终态：`ROUND14A_COMPLETE_GDN_PROMOTED_B_C_NO_PROMOTION`

## 1. 结论

Round14a 已按冻结的 Round14 S0 身份完成 A/B/C 三条主线。唯一进入默认配置的新变化是 GDN 第二实现：NPU convolution/recurrence 加本地 sigmoid gate、RMSNorm 和 output projection。静态驻留继续使用 Uniform Q32；默认最大并发继续为 1。

本轮取得了真实单请求收益。GDN 的两块请求 ABBA 中，W1 TTFT 分别降低 53.98% 和 58.20%，W1 E2E 分别降低 36.43% 和 39.86%。W2 串行吞吐分别提高 16.05% 和 14.45%，W4 串行吞吐分别提高 28.17% 和 14.01%，质量、状态、取消、复用和资源保护均通过。

增量前层常驻 FPLUS1 在增加 4.39453125 GiB 名义 BF16 resident 的情况下，W2 串行吞吐只提高 2.55% 和 3.99%，没有达到两块均提高 5% 的门槛，因此不晋级。

并发 2/4 已经是真实 SGLang engine batch，不是 HTTP 排队。W2 的 c4 聚合吞吐在四块中为 5.068–5.284 tok/s，相对 c1 提高 1.744–2.460 倍；但短 W1 在升序块无法稳定满足冻结的 c2/c4 吞吐和逐请求时延门，且存在明显的档位顺序效应。统一默认因此不晋级，仍为 c1。W2 并发结果保留为后续按 workload 分类策略的正向证据，不写成当前默认资格。

Round14 遗留的 8 条英文公开代理题失败和 9 条待人工审阅继续保留。本轮通过的是数值、状态隔离、已冻结 canary、取消复用和窄输入输出范围，不宣称广泛产品质量。

## 2. 冻结起点与最终配置

起点为 Round14 S0：Qwen3.8 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`、Kunpeng C128（NUMA 2/3/4/7）、只读 MXFP4 CPU bank、Static Uniform Q32 BF16/NZ29、chunk32、PLE ASYNC、QSA Vectorized Prefill 加 DIRECT_BF16_ROW Decode、H129、TP1、串行单请求。Dynamic、deferred、R2/R3、transpose-view 和 shadow320 均关闭。

最终只改变 GDN provider。最终默认仍为 Uniform Q32、H129、并发 1；正式资格范围仍是输入不超过 128 tokens、输出不超过 128 tokens。详见 `FINAL_CONFIG.json`。

## 3. A 线：163.48 MiB 根因和 GDN 请求收益

### 3.1 明确对象和根因

Round14 观察到的 171,424,768 bytes（163.48 MiB）不是生产候选每层持久 workspace，也不是 36 层各复制一份。它来自旧组件探针的 eager reference 在 `torch.inference_mode()` 外运行：

1. `gdn.py:121` clone 初始 FP32 recurrent state；
2. `gdn.py:139/142` 建立 tokenwise recurrence 的 autograd graph；
3. `qwen4_exp.py:344` 将 `next_state.recurrent_state` 用 `copy_` 写入持久 request state pool；
4. 该持久 pool 因此保留 `CopyBackwards` 及其上游图。

主要 retained storage 是两个 `[48, 128, 128]` FP32 recurrent-state 图，分别贡献约 100,679,680 和 59,778,560 active bytes；其余约 1.45 MiB 来自 query/key/decay 和中间图。生产 ModelRunner 在 `torch.inference_mode()` 内执行，所以这是旧 probe 的 autograd false positive。

候选首次调用相对已构建组件只增加约 1.82 MiB transient；释放后仅剩 3,584 bytes。组件 close 后只剩约 8 KiB 已登记的全局 shape metadata cache，按每个 decorated function 最多四个 identity 有界，并非逐层/逐请求增长。

### 3.2 生命周期、数值和组件门

修复后的生产等价探针在 inference mode 下完成 20 个同 shape/分块/Decode 生命周期和 5 个 cancel/reuse 周期，没有调用 `empty_cache`，allocated growth 为 0，close 回收通过。

三个数值路径的 output relative L2 为 0.00679、0.00678 和 0.00638，均低于冻结的 0.03 门；conv state relative L2 为 0，recurrent state relative L2 约 0.00476–0.00528。

两块完整 parent 的 candidate/baseline host 比值为 0.07854 和 0.08059，device 比值为 0.07681 和 0.07794，均远超“至少改善 15%”的组件门。

### 3.3 请求 ABBA

| 块 | W1 TTFT 比值 | W1 E2E 比值 | W2 TPOT 比值 | W2 串行吞吐比值 | W4 串行吞吐比值 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| A，v294→v295 | 0.4602 | 0.6357 | 0.9173 | 1.1605 | 1.2817 | PASS |
| B，v297→v296 | 0.4180 | 0.6014 | 0.9566 | 1.1445 | 1.1401 | PASS |

两个 baseline/candidate smoke、五类 canary、多步 Decode、cancel/reuse、有限 logprob、36 个实际 GDN branch rows 和资源回收均通过。候选晋级为 Round14a 单请求父配置。

## 4. B 线：五桶归因与增量常驻

### 4.1 五桶内存

两个 fresh S0 profile 块给出重复一致的 storage 分类。下表是核心量级，alias 只计一次；Host 只读 bank 的逻辑文件字节不冒充匿名 RSS。

| 桶 | NPU storage |
|---|---:|
| Expert | 14.0625 GiB |
| Dense backbone | 约 9.1859 GiB |
| KV + GDN state | 约 0.3123 GiB |
| Workspace | 约 0.2931 GiB |
| PLE | 约 0.0612 GiB |

S0 allocator 约 24,599 MiB、reserved 约 25,152 MiB；两个块 Host cgroup 峰值约 338.4 GiB。MXFP4 bank 是 48 个只读文件，逻辑大小约 59.77 GiB，其 page cache 在归档和完整 SHA 后定向退休。

### 4.2 五桶时间

W2 Decode 的非可加、按实际 layer count 归一投影在两个块中稳定排序为 Expert、GDN、QSA、PLE：

| 模块 | 块 A | 块 B | 解释 |
|---|---:|---:|---|
| Expert | 11,194 ms | 11,275 ms | CPU MXFP4 DRAM compute 与 result dependency |
| GDN | 8,766 ms | 8,717 ms | 已在 NPU |
| QSA parent | 8,597 ms | 8,431 ms | 已在 NPU；parent 与 child 不重复相加 |
| PLE | 224 ms | 221 ms | exposed wait 很小 |

这些值处在重叠时钟域，只用于排名，不能相加成 E2E。Hybrid 请求没有 Layerwise 专家权重搬运；CPU Expert 的计算、activation/result 依赖与 Layerwise weight transport 已分开描述。

### 4.3 安全 k 和请求去留

每补满一层增加 480 个专家，相对 Q32 的名义增量为 4,718,592,000 bytes，即 4.39453125 GiB。为晋级 GDN、并发 4 的 state/workspace 和 guard/load headroom 预留后，allocator/owned cap 决定 `k_safe=1`；`k=2` 已越过投影 allocator/owned 上限。因此筛选 FPLUS1（layer0 Q512，其余 Q32）和同容量 U_PLUS Q42。

FPLUS1 与 U_PLUS 的组件数值、真实物理 map、四 NUMA 分区和回收通过；FPLUS1 进入请求 ABBA。结果：

| 块 | W2 TPOT 比值 | W2 串行吞吐比值 | 新增名义 resident | 结论 |
|---|---:|---:|---:|---|
| A | 0.9966 | 1.0255 | 4.3945 GiB | 未达 5% |
| B | 0.9605 | 1.0399 | 4.3945 GiB | 未达 5% |

W1/W4 保护通过，但主要门失败。FPLUS1 不晋级，最终保留 Uniform Q32。

## 5. C 线：真实并发 2/4

### 5.1 隔离和修复

C 从可信 S0 独立推进。实现为每个活跃请求建立独立的 KV/QSA runtime、GDN/conv state slot、PLE consumer/ticket、CPUInfer request identity、workspace/epoch 和 cancel closure；只读模型权重与有界 PLE ring 共享。H129 用每个 request 的 total token 判断，绝不使用 batch token 总和。

三次原始失败均保留：v322 暴露旧 Qwen Mamba pool `max_running=1` gate；v323 暴露旧 streaming serial gate；v324 证明真实 batch2 已运行，但 scheduler 在 after hook 前清除了 `req_pool_idx`。三者分别修复并使用新 run 验证，没有覆盖旧 FAIL。

v325 c2 和 v326 c4 smoke 均通过 5 类 canary、多步 Decode、取消一个请求时 peer 正确、同进程 reuse、Hybrid/QSA/PLE 三平面路径和真实最大 engine batch 2/4。

### 5.2 四块正式结果

顺序为 A `1→2→4`、B `4→2→1`、C `4→2→1`、D `1→2→4`。每个档位 W1 和 W2 各 12 个 measurement；每个模型块固定 82 generation，profiler OFF。

下表的吞吐是完整连续测量窗口中的 aggregate output tokens/s；括号为 c2/c1、c4/c2、c4/c1。TPOT 是逐请求中位值。

| 块 | W | c1 tok/s | c2 tok/s | c4 tok/s | 吞吐比值 | c1/c2/c4 TPOT (s) | c2 / c4 块门 |
|---|---|---:|---:|---:|---|---|---|
| A v327 | W1 | 1.695 | 1.988 | 2.123 | 1.173 / 1.068 / 1.253 | 0.290 / 0.558 / 1.138 | FAIL / FAIL |
| A v327 | W2 | 2.938 | 4.072 | 5.123 | 1.386 / 1.258 / 1.744 | 0.302 / 0.429 / 0.683 | — |
| B v328 | W1 | 1.356 | 1.925 | 2.099 | 1.420 / 1.090 / 1.548 | 0.439 / 0.580 / 1.144 | PASS / FAIL |
| B v328 | W2 | 2.102 | 4.118 | 5.170 | 1.959 / 1.255 / 2.460 | 0.439 / 0.427 / 0.676 | — |
| C v329 | W1 | 1.382 | 1.969 | 2.188 | 1.425 / 1.111 / 1.583 | 0.429 / 0.568 / 1.101 | PASS / PASS |
| C v329 | W2 | 2.162 | 4.215 | 5.284 | 1.949 / 1.254 / 2.444 | 0.423 / 0.418 / 0.662 | — |
| D v330 | W1 | 1.537 | 1.944 | 2.059 | 1.265 / 1.059 / 1.339 | 0.318 / 0.573 / 1.152 | FAIL / FAIL |
| D v330 | W2 | 2.853 | 3.954 | 5.068 | 1.386 / 1.282 / 1.776 | 0.311 / 0.447 / 0.688 | — |

c2 的冻结目标为每块 c2/c1 至少 1.20；c4 为每块 c4/c2 至少 1.10 且 c4/c1 至少 1.30，并受逐请求时延保护。A 的 W1 c2 与 c4 均失败；B 的 W1 c4/c2 为 1.090；D 的 W1 c4/c2 为 1.059，且升序 W1 时延保护失败。由于要求四块都通过，c2/c4 均不晋级为统一默认。

数据也揭示顺序效应：c1 在升序块先运行时 W2 为 2.85–2.94 tok/s，在反序块最后运行时只有 2.10–2.16 tok/s；c2/c4 稳定得多。这正是使用四 fresh 模型镜像而不是一个进程内单次比值的原因。

W2 的并发价值是真实的：c4 在所有块都达到真实 Decode batch4，聚合吞吐为 5.068–5.284 tok/s，代价是中位 TTFT 约 12.78–13.43 秒、TPOT 约 0.662–0.688 秒、E2E 约 96.9–101.0 秒。它适合后续独立研究基于 workload 的并发 profile，但不能在本轮覆盖 W1 后宣称统一默认收益。

## 6. 资源、预算和回收

最终预算使用：23/30 模型组、535/752 generation attempts、4/24 abort controls、10/32 设备组件进程、3.5959/64 NPU device-hours。失败启动、修复 smoke 和正式镜像均计入；blind retry 为 0。

四个 C 正式块的峰值如下：board HBM 29,175–29,176 MiB，owned HBM 26,158 MiB，Host cgroup 280.76–297.77 GiB，均低于冻结 cap。每块结束都取得 guard exit ACK、删除自有容器并确认物理 12 为空。

归档和全文件 SHA 在缓存退休之前完成。缓存退休只对登记的只读 `.gguf`/`.safetensors` 执行 POSIX_FADV_DONTNEED，并在全机 fd/mmap 审计无引用后进行；没有调用全局 drop_caches，没有修改训练或外部任务。最终物理 12 无进程、无 Round14a 容器，timer 仍暂停。

## 7. 最终边界

- 默认配置：promoted GDN + Uniform Q32 + concurrency1。
- 正式输入不超过 128 tokens、输出不超过 128 tokens。
- c2/c4 实现、状态隔离和真实引擎 batch 已验证，但统一默认性能门未通过。
- W2 c4 是条件正向结果，不是默认资格。
- FPLUS1 正确但收益不足，不以更大的 k 越过资源合同。
- 不恢复 Round14 的公开代理题质量审阅，不宣称广泛产品质量、并发 4 部署资格、长上下文、TP/EP/PP、HiCache、FP8 KV 或生产部署。

## 8. 主要证据

- `A_GDN_MEMORY_ROOT_CAUSE.json`
- `A_GDN_COMPONENT_RESULT.json`
- `A_GDN_REQUEST_RESULT.json`
- `B_PROFILE_RESULT.json`
- `B_RESIDENT_COMPONENT_RESULT.json`
- `B_RESIDENT_REQUEST_RESULT.json`
- `C_CONCURRENCY_COMPARISON_CONTRACT.json`
- `C_CONCURRENCY_RESULT.json`
- `C_CONCURRENCY_RESOURCE_RECEIPT.json`
- `BENCHMARK.csv`
- `ROUND14A_EXECUTION_BUDGET.json`
- `RESOURCE_RETIREMENT_RECEIPT.json`
- `FINAL_CONFIG.json`
- `FINAL_STATUS.json`

原始 v322–v330 tar、各 run final/supervisor、admission、preflight、monitor、事件文件及归档 SHA 保存在 `c-concurrency-results/`。A/B 原始证据与 snapshot/build/source gate 保存在各自目录。所有最终 top-level 证据的 SHA 由 `ROUND14A_SHA256SUMS.json` 冻结。
