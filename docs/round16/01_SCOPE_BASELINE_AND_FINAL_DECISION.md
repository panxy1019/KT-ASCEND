# Round16 范围、基线和最终判断

## 1. 目标

Round16 要回答两个互相独立的问题：

1. 在 SGLang + KT 主线上，当前约 3.x tok/s 的真实请求剩余成本是否支持一个有界、精确的 A2 同层 overlap 或 A3 NPU 分段图候选？
2. 在 vLLM-Ascend 承载下，能否在不改变模型数学、专家精度和资源拓扑的前提下接入同一套 KT Expert，并且在完整请求上比当前 S0 更值得作为默认实现？

本轮没有重新打开 NZ update、R2/R3、transpose-view、shadow320、Dynamic、deferred、front-loading、W8A8、MTP、TP/EP/PP、多请求并发或长上下文。

## 2. S0 冻结基线

| 项目 | 冻结值 |
|---|---|
| 模型 | Qwen3.8 frozen revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| 逻辑专家 | 48 层，每层 512 routed experts，Top-10 |
| CPU Expert | Kunpeng C128，NUMA 2/3/4/7，4 个 pool × 32 workers |
| CPU bank | 48 层只读 MXFP4 |
| NPU resident | Uniform Q32，BF16/NZ29 |
| 调度 | chunk32，TP1，max-running=1，串行单请求 |
| GDN | Round14a promoted implementation |
| QSA | Vectorized Prefill + DIRECT_BF16_ROW Decode |
| PLE | ASYNC |
| 路径 | H129；正式输入和输出各不超过 128 token |
| 图 | S0 保留冻结行为；B 首个完整模型强制 eager |

Round15 没有晋级新的默认计算候选，所以 Round16 的父配置仍是 Round14a S0。

## 3. 结果分层

Round16 将结论分为三类，避免把组件、模型和性能混在一起：

- **功能结果**：真实构造、加载、数值、状态、取消和回收是否成立。
- **性能结果**：新鲜完整请求是否获得收益。
- **架构结果**：某条机制可移植、深度绑定、或仍缺证据。

B 的功能链路已经完整通过，但请求性能没有晋级。A 没有满足证据门的候选，因此没有进行 A 模型或请求性能实验。

## 4. 最终状态为什么是 PARTIAL

最终状态是 `ROUND16_PARTIAL_WITH_GAPS`，原因只在 A 线：

- 两个 fresh profile 块确实完成；
- 直接 NPU span 确实存在；
- 但没有直接隔离的 CPU result exposed-wait 区间；
- 也没有直接隔离的 Host launch/metadata critical 区间；
- 原生 C128 worker ABI 不提供 worker start/end callback；
- 冻结合同禁止用 parent 减 children、嵌套 span 求和或跨 Host/NPU 时钟相加填补缺口。

因此 A2/A3 只能写 `NOT_SELECTED_EVIDENCE_GAP`，不能写 `NO_GAIN`。

B 已经达到 `B16_FULL_RUNTIME_AND_REQUEST_MEASURED_NO_PROMOTION`：它不是集成缺口，也不是未测性能。

## 5. 默认承载判断

B 与 S0 的比较是系统方案比较，因为框架、调度实现和 graph 模式不同。它不是纯框架消融，也不是 KT Expert kernel 的单变量对照。

两个独立 W2 配对块均显示：

- B TPOT 更高；
- B 串行 output tokens/s 更低；
- B E2E 更长；
- W4 保护门也失败。

因此没有切换默认承载的依据。最终配置继续使用 **S0 SGLang + KT**。

## 6. 不得外推的结论

- QSA 是最大的已采样 NPU span，不等于它已被证明是整个请求的唯一瓶颈。
- B 跑通 TP1 eager，不等于 vLLM-KT 的 Graph、TP、多请求或长上下文已资格。
- 三页 KV 组件 probe 通过，不等于完整模型任意长上下文都通过。
- B 比 S0 慢，不等于 vLLM-Ascend 所有部署都慢；本结论只属于冻结配置和 workload。
- A 没有候选，不等于 A2/A3 永远无价值；当前缺的是直接关键路径证据。
