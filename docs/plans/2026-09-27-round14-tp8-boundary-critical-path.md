# Round14：A 同请求 TP8 对照、B 真实使用边界、C 关键路径加速

状态：ACTIVE。来源：用户 2026-09-27 Round14 执行指令。本文件是本地断点恢复入口；不改写 Round13 的 `ROUND13_PARTIAL_WITH_GAPS`，不恢复旧 timer。

## 冻结起点

`S0` 为 Round13 最终限定配置：Uniform Q32、Kunpeng C128（NUMA 2/3/4/7）、只读 MXFP4 CPU bank、BF16/NZ29 resident、chunk32、PLE ASYNC、QSA VECTORIZED Prefill 与 DIRECT_BF16_ROW Decode、TP1 串行。默认 H129 和原 fallback 保留；Dynamic、deferred、R2/R3、transpose-view、shadow320 均关闭。

从 Round13 的实际冻结快照、manifest、原始证据和源码 tuple 恢复。Git HEAD 不是运行身份的替代。oracle、candidate 和运行组合分别登记；大模型前先在冻结 CPU-only 镜像检查全部 source gate 与入口。

本轮必须回答：

1. Native TP8 old-QSA 在相同 W2（128 输入、128 实际输出）上的串行 output tok/s、E2E、TTFT、TPOT 和资源。
2. 当前 KT S0 的真实请求分布覆盖、默认 H-BOUNDED 范围及范围外行为。
3. fresh profile 后的剩余关键路径，以及一个单变量优化的真实请求收益或不晋级原因。

## A：同请求 KT / Native TP8 系统部署对照

- `K`：fresh S0，物理12。
- `N`：Round12a Native SGLang TP8、BF16、old-QSA、物理0–7、主模型 NPU 与 Host PLE，无 KT、无整层 offload。
- 顺序：`K-A → N-A → N-B → K-B`，四个独立模型进程。
- 主 workload：W2，模板后的相同 128 输入 token IDs、128 实际输出 token。
- 补充：W0=64/2、W1=128/16、W4=64/32、W3=512/16；K 的 W3 只使用明确 H513 实验 profile。
- 每 workload 1 warmup＋3 measurement；每模型组不超过 24 generation；profiler OFF。
- 固定 revision、tokenizer/template、采样、stop 和长度合同。记录真实输出、有限 logprob、质量、全部 TP rank、实际 QSA 分支、错误和资源。
- 每块报告 E2E、TTFT、TPOT、完整串行 output tok/s、requests/s、同时 HBM 峰值和 Host 峰值。
- 这是单 NPU＋C128 KT 与八 NPU BF16 Native 的系统方案比较；精度、卡数和 QSA 不同，不能写成纯 KT 因果加速。
- 物理0–7不可用时 A 等待，B/C 继续；最终缺失则保留缺口。

## B：真实代理分布与默认边界

- 建立 64 条可追溯请求：16 DEV、48 HOLDOUT，按会话拆分，覆盖中英文、代码、摘要/检索、格式和多轮历史。
- 无业务日志时明确标为公开代理分布。
- 长度桶：`<=128`、`129–256`、`257–512`、`513–768`。
- 正常质量集尊重 EOS；固定长度性能集单列。记录实际输出长度、拒绝/错误率、任务质量与分桶时延。
- 对当前 DIRECT_BF16_ROW / vectorized provider 验证 prompt、decode 步数、有效 context 与 state 矩阵；不继承旧 Round11 的形状资格。
- 保留 `S0-LEGACY` H129；新增 `H-BOUNDED`，首目标 512。只有 768 当前 shape、质量与资源全部通过才扩展到 768。
- 输出上限 256，且总 context 不超过已验证范围。范围外在预分配前明确拒绝或要求显式 legacy，不静默进入分钟级 Layerwise。
- 长范围 reference 使用已验证当前 forced-Hybrid。配置接线收益与 kernel 收益分别报告。
- 最多 4 个模型组，每组不超过 64 generation。两个独立确认块包含 HOLDOUT、cancel/reuse。
- 开放答案未完成人工审阅时标 `PENDING_REVIEW`，不声称全面产品质量。

## C：fresh profile 后只优化一个热点

- 两个 fresh S0 profile 块，每块 4 generation：warmup、W2、一个已资格 Prefill-heavy 请求、drain。
- 复用 observer/msProf，避免新增逐算子同步；采样外推与嵌套时间分开。
- 按关键路径选择一个热点，最多两种实现，最多晋级一个。
- 允许：新 Ascend C/Triton-Ascend 算子、QSA 分块/online-softmax 融合、GDN gate/recurrence 融合、Kunpeng microkernel、局部 Graph、同层精确异步/调度重排。
- 禁止把 GPU 示例速度外推到 Ascend；优先消除工作，再考虑融合与重叠。
- 独立 reference、真实权重/路由 fixture、causal/GQA、state 连续性、fallback、取消/late completion、lease 与 allocator 必须通过。
- 当前层结果未完成时后续层不得消费；禁止近似 deferred 或丢专家。
- 组件两个配对块的完整 parent 都至少改善 15% 才进入模型。
- 两模型 smoke 每边最多 8 generation；请求比较顺序 `K0/KC/KC/K0`，每组最多 32 generation，覆盖 W0–W4 与 8 条冻结真实保护请求。
- 默认 Decode 门：两块 W2 TPOT 均降低至少 10%，完整串行吞吐均提高至少 10%。
- 仅在新 profile 明确 Prefill 主导时，实施前可冻结 Prefill 门：TTFT 至少降低 15%，E2E 至少降低 10%。结果后不得改门。
- 质量不得下降；关键保护指标不得两块一致退化超过 5%。单块退化超过 10% 要独立确认；不能确认则不默认晋级。
- 性能负结果可以收口；集成或请求比较未完成不能写 `NO_GAIN`。

## 可选 C-S：并发2预演

只有单请求 A/B/C 主线收口、资源与隔离合同成立后才触发。先验证两个请求的 KV/GDN/PLE/Expert 状态、取消和 buffer reuse 不互相污染，再执行 `1/2/2/1` 四块，每块最多 32 generation。只有真实引擎 in-flight 大于1时报告并发；同时报告 aggregate tok/s 和单请求时延。

本轮不进入并发4/16、KT TP/EP/PP、HiCache、FP8 KV、Graph/MTP 全局开发或旧 NZ/Layerwise 优化链。

## 硬预算

| 阶段 | 模型组 | generation | abort | 组件 |
|---|---:|---:|---:|---:|
| A | 4 | 96 | 0 | 0 |
| B | 4 | 256 | 4 | 4 |
| C profile | 2 | 8 | 0 | 0 |
| C smoke | 2 | 16 | 2 | 0 |
| C 请求 ABBA | 4 | 128 | 4 | 12 |
| 可选并发 | 4 | 128 | 2 | 4 |
| 定向修复储备 | 4 | 32 | 2 | 4 |
| 总帽 | 24 | 664 | 14 | 24 |

单组件 1800 秒、单模型组 21600 秒、单请求 1800 秒，额外安全 drain 最多 180 秒。全轮 72 小时墙钟、128 NPU device-hours、离线 CPU 准备 12 小时。TP8 每小时计 8 device-hours。失败计费；blind retry 为0；同根因最多2次有修改验证。不挪用 B/C 保留额度。

## 资源、安全与回收

- 每次设备启动 fresh 2×60秒准入。
- 继承 Round13 更严格的实际资源 cap、每候选 NUMA 至少48 GiB MemFree、Host/HBM headroom、训练容器完整身份和未知设备进程 fail-closed。
- 不换拓扑、不降低门、不修改外部任务。
- 退出顺序：自有进程退出与 guard ACK → 原始证据归档与完整 SHA → 全机登记文件 fd/mmap 审计 → 仅对登记只读文件执行 DONTNEED → fincore 与 NUMA 回执。
- 不全局 drop_caches，不自动恢复 timer，不 push/merge/tag/reset/clean。

## 终态

- A/B 必需证据完整且 C 请求收益通过：`ROUND14_MAINLINE_COMPLETE_GAIN`。
- A/B 完成且 C 完整证明无新增收益：`ROUND14_MAINLINE_COMPLETE_NO_NEW_GAIN`，性能目标不写 PASS。
- 任一必需臂、质量或请求验证未完成：`ROUND14_PARTIAL_WITH_GAPS`，保留独立成功子结果。

## 合同来源缺口与裁定

用户点名的新版 `36_C线_关键路径加速与并发演进.md` 与 `37_执行合同_预算_自动推进与终态.md` 在本地所有 checkout、当前 GitHub 远端引用和 GitHub 精确检索中均不存在。裁定：本文件完整保留用户消息中可执行的 C 线、预算、安全和终态条款，并以该消息为最高优先级；缺失文件不阻止 T0/A/B/C 的已定义工作。若后续出现原文，只能收紧未开始的工作，不能改写已产生的原始证据。
