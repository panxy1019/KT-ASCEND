# Round14 最终报告：A 同请求 TP8 对照、B 使用边界、C 关键路径候选

日期：2026-09-27
分支：`codex/round14-tp8-boundary-critical-path`
冻结 Git HEAD：`a18d05cf166851a1440fede49e21ea2ac1788fd6`
终态：`ROUND14_PARTIAL_WITH_GAPS`

## 1. 结论

Round14 已完整回答 A 的同请求部署方案对照；B 完成了一块 64 请求公开代理分布，但第二独立确认块未完成且 H512 出现质量回归；C 用两块 fresh S0 profile 找到 GDN 是最大剩余热点，第二种 NPU GDN 实现通过数值门，但未通过请求生命周期 allocator 复用门，因而没有进入完整模型请求测试。

最终默认配置不变，仍为 Round13 S0：Uniform Q32、Kunpeng C128（NUMA 2/3/4/7）、只读 MXFP4 CPU bank、BF16/NZ29 resident、chunk32、PLE ASYNC、promoted QSA Prefill＋DIRECT_BF16_ROW Decode、H129、Dynamic/deferred/R2/R3/transpose-view/shadow320 OFF、TP1 串行。

`performance_objective` 为 `NOT_ACHIEVED`。这不是 C 的性能负结果：C 在性能测量前被资源回收门拒绝，没有合法请求级速度样本。

## 2. A：Native TP8 old-QSA 与 KT S0 的同请求系统对照

四个独立进程严格按 `K-A → N-A → N-B → K-B` 执行。K 使用物理 12 的单卡 KT 异构方案；N 使用物理 0–7 的 Native TP8/BF16/old-QSA、Host PLE，无 KT、无整层 offload。两臂模型 revision、模板、采样和 prompt token IDs 相同；精度、卡数和 QSA 不同，因此只允许作系统部署方案比较。

### 2.1 W2（128 输入 / 128 实际输出）

| 配对块 | KT 串行 output tok/s | Native TP8 串行 output tok/s | Native/KT | KT TTFT | Native TTFT | KT TPOT | Native TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|
| A：v235 ↔ v251 | 3.042707 | **3.228323** | +6.10% | 5.0411 s | 6.3217 s | 0.29060 s | 0.26250 s |
| B：v238 ↔ v252 | 2.941903 | **3.341694** | +13.59% | 4.9849 s | 6.1245 s | 0.30085 s | 0.25725 s |

Native TP8 的 W2 精确实测结果是 **3.228323 和 3.341694 output tok/s**，不是用 `1/TPOT` 推算的值。它在长 Decode 的 TPOT 和完整串行吞吐上更快，但 TTFT 两块分别慢约 25.4% 和 22.9%。W2 E2E 两块分别缩短约 5.45% 和 10.05%。

### 2.2 跨 workload 解释

- W0 64/2：KT 更快；Native 两块串行吞吐分别低约 18.0% 和 3.2%。
- W1 128/16：Native TTFT 更慢，两块完整串行吞吐分别低约 7.5% 和 2.8%。
- W2 128/128：Native 的 Decode 优势超过额外 TTFT，形成 +6.1% / +13.6% 完整吞吐收益。
- W3 512/16：KT 使用明确 H513 实验 profile；Native 两块串行吞吐分别低约 16.1% 和 7.2%。
- W4 64/32：接近持平，Native 两块分别约 +1.3% 和 +4.6%。

Native 同时峰值 whole-card HBM 约 292,889–292,903 MiB（8 卡总和），KT 为约 28,813 MiB（1 卡）。因此结果表达的是“8 卡纯 NPU 与 1 卡 Ascend＋Kunpeng 异构系统的部署取舍”：更长 Decode 下 Native 更快，短输出和 Prefill-heavy 请求上 KT 的 TTFT/资源效率更有利。它不能写成 KT kernel 因果加速或等资源比较。

A 状态：`PASS_ROUND14_A_SYSTEM_COMPARISON`。

## 3. B：真实代理分布与默认边界

建立了 64 条可追溯公开代理请求，按会话拆分为 16 DEV＋48 HOLDOUT，覆盖中文、英文、代码、摘要/检索、结构化格式和多轮历史。没有业务日志，因此证据范围明确为 `PUBLIC_PROXY_DISTRIBUTION_NOT_BUSINESS_LOGS`。

### 3.1 完整块 v273

v273 完成 64/64 请求、1 次 abort control、cancel/reuse、512 输入＋256 输出边界接受，以及范围外在 scheduler 前拒绝。资源峰值：whole-card HBM 28,815 MiB、owned HBM 25,798 MiB、Host cgroup 363,646,230,528 bytes。

质量结果：

- 自动通过：31；
- 自动失败：8，全部来自英文类别；
- 待人工审阅：9，全部为开放代码答案；
- 中文、结构化、多轮、摘要/检索的冻结自动规则未出现失败。

分桶时延中位数：

| 输入桶 | 请求数 | E2E | TTFT | TPOT |
|---|---:|---:|---:|---:|
| ≤128 | 4 | 13.0604 s | 3.9425 s | 0.29412 s |
| 129–256 | 4 | 17.3315 s | 8.2020 s | 0.29404 s |
| 257–512 | 4 | 25.3741 s | 15.7081 s | 0.31160 s |

### 3.2 默认策略

H512 只能记为“执行了一块完整实验”，不能成为默认：一方面 8 个冻结英文已知答案失败，另一方面计划要求两个独立确认块，而 v269 在第 61 个请求因 host-only probe 4096-row 上限退出，不能冒充第二块。B 的 4 模型组额度已经用完。

因此：

- 默认保持 `S0_LEGACY_H129`：总 token `<129` 走 Hybrid，`>=129` 走原 legacy/fallback；
- 当前真正资格范围仍是 Round10 已证明的输入总 token `≤128`；
- H512：`NOT_DEFAULT_QUALITY_REGRESSION`；
- H768：`NOT_RUN_CONDITION_NOT_MET_AND_B_MODEL_GROUP_CAP`；
- 范围外必须显式 legacy/fallback，不得静默塞入实验 H512。

B 状态：`ROUND14_B_PARTIAL_WITH_ONE_COMPLETE_BLOCK`。

## 4. C：新 profile、热点选择与 GDN 候选

### 4.1 两块 fresh S0 profile

v274/v275 各运行 warmup、W2、W3、drain。observer 只在原事件上 record/query，没有新增每算子 synchronize；parent 与 child 不相加，host wait 与 device event 不跨时钟域相加。

两块稳定显示：

| 模块 | Extend 可归因比例 | Decode 可归因比例 | 判断 |
|---|---:|---:|---|
| GDN | 62.24% / 60.63% | 23.79% / 23.50% | 最大且两块稳定，选为唯一候选 |
| Routed Expert | 约 12.4%–13.1% | 29.56%–30.90% | 合格热点，但本轮只允许一个 |
| QSA attention | 20.23% / 21.96% | 22.70% / 22.73% | 合格热点，但已是 promoted 基线且本轮只允许一个 |
| PLE consumer | 约 0.14% Extend | 约 0.60% Decode | 不满足热点门 |
| PLE exposed wait | 低于 0.01% Extend | 约 0.04% Decode | 不应重启 overlap |

GDN 的跨块中位归因约为 Extend 61.16%、Decode 23.52%。因此在实施前冻结为 Prefill 门：两个请求块 W1 TTFT 均至少降低 15%，E2E 均至少降低 10%；组件完整 parent 两块均至少改善 15%。

### 4.2 第一实现：完整调用 Qwen3.5 parent

v280 暴露出明确语义错误：卷积状态 relative L2 为 0、recurrent state 为 0.00528，但最终输出 relative L2 为 0.97137。原因是 Qwen3.5 parent 的 RMSNorm gate 使用 swish，而本地冻结 Qwen4/Qwen3.8 语义要求 sigmoid。该实现永久拒绝，未放宽 0.03 数值门。

### 4.3 第二实现：NPU core＋冻结 Qwen4 sigmoid gate

第二实现只让冻结 SGLang NPU backend 执行 causal convolution 和 GDN recurrence，随后继续使用本地 `sigmoid_gated_rms_norm`、原 norm 权重和 out projection。候选默认 OFF。

数值证据（v282/v284/v286 重复一致）：

- 三种 layer0 路径的输出 relative L2：0.00638–0.00679；
- conv state relative L2：0；
- recurrent state relative L2：0.00476–0.00528；
- 独立 FP64 三 token oracle：eager 最大约 0.00404，candidate 最大约 0.00582；
- 全部低于冻结门 0.03。

但资源门失败：在精确预热后的同 shape、同分块、同 Decode replay 中，allocator 每轮稳定从 294,230,528 bytes 增长到 465,655,296 bytes，即 **+171,424,768 bytes**。移除探针 timing event 后增量不变，说明没有证明 production request 生命周期的 allocator 复用。终局 v286 按原门 fail-closed，物理 12 在进程退出后恢复为空。

因此：

- 组件性能 timing：未执行；资源回收门先于性能门；
- 两模型 smoke：未运行；
- `K0/KC/KC/K0` 请求 ABBA：未运行；
- 请求收益：`NOT_MEASURED_COMPONENT_GATE_FAILED`；
- 候选：不晋级、默认 OFF；
- 结论类型：资源合同拒绝，不是 `NO_GAIN`。

C 状态：`ROUND14_C_COMPONENT_REJECTED_RESOURCE_CLEANUP`。

## 5. 可选并发 C-S

未运行。单请求候选没有通过组件资源门，且 B 主线已经存在必需证据缺口。在这种状态下启动并发2不能回答候选收益，也会引入状态隔离与调度变量。没有并发吞吐、in-flight>1 或状态隔离资格声明。

## 6. 预算与安全回收

最终消耗：

| 项目 | 已用 | 总帽 |
|---|---:|---:|
| 模型组 | 14 | 24 |
| generation attempts | 238 | 664 |
| abort controls | 1 | 14 |
| 组件进程/尝试 | 6 | 24 |
| NPU device-hours | 4.16464 | 128 |

所有真实启动前均执行 fresh 2×60s admission。未执行 blind retry；每次失败都先改变对应 source/test contract 并保留新 run 身份。没有换卡、降低 NUMA/HBM 门、修改训练任务或全局 drop_caches。

终局回收：

- 物理 12/card6/chip0：`No process in device`；
- 本项目容器/进程：0；
- 外部训练容器完整 ID 保持运行且未修改；
- 最后一次定向缓存退休：全机 fd/mmap 审计无引用，179 个登记只读 `.gguf/.safetensors` resident pages 34,048 → 0；
- 终局 NUMA MemFree KiB：node2 256,307,200；node3 254,545,476；node4 205,863,080；node7 257,668,116；
- 历史 timer 保持暂停。

## 7. 为什么终态是 PARTIAL

A 必需证据完整。B 缺少第二个完整确认块，H512 有质量回归，H768 未运行。C 的关键路径调查完整，但候选没有通过 allocator 复用门，因而没有完整模型请求收益或无收益结论。按冻结终态合同，不能写 `MAINLINE_COMPLETE_GAIN`，也不能把未跑完的请求集成写成 `MAINLINE_COMPLETE_NO_NEW_GAIN`。

最终状态为：`ROUND14_PARTIAL_WITH_GAPS`。

## 8. 证据入口

- `A_RESULT.json`：A 四进程对照和完整指标；
- `B_RESULT.json`：B 完整块、质量、边界与缺口；
- `C_PROFILE_RESULT.json`：两块 profile 和热点排名；
- `C_RESULT.json`：两种 GDN 实现、数值、allocator 与去留；
- `BENCHMARK.csv`：统一已测表，未知字段保持空白；
- `ROUND14_EXECUTION_BUDGET.json`：append-only 预算事件；
- `RESOURCE_RETIREMENT_RECEIPT.json`：终局进程、NUMA 和缓存回收；
- `FINAL_STATUS.json`：机器可读终态；
- `FINAL_VERIFICATION.json`：62 项测试、编译、语义、预算、归档和远端空闲复核；
- `HASHES.json`：最终交付、控制源码、构建身份与正式证据归档的 218 项 SHA-256 闭包；本地 `.superpowers/sdd` 内部账本按仓库忽略规则不进入公开闭包。
