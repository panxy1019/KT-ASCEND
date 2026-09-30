# Round18 Phase I 冻结结论

- 日期：2026-09-30 UTC
- 状态：`LOCKED_NO_FURTHER_PIECEWISE_WORK`
- 后续阶段：Phase II Native expert pool

## 已验证结果

1. `native-partial-v004` 证明冻结 Ascend 原生 selector、
   `AscendUnquantizedFusedMoEMethod`、AllGather communication 和每层
   Physical-Q32/NZ29 owner 可以在 512 个逻辑专家域中只计算 resident
   occurrence。`-1` 映射在这条实际路径中跳过 nonresident occurrence；
   all-resident、all-CPU、mixed、multi-token 和 Top-K tie 用例通过独立
   checkpoint BF16 reference。最大 relative L2 为 `0.0026470785960555077`，
   输出 repeat-exact。
2. `native-hybrid-v001` 证明最小 CPU complement 与 native resident partial
   可以在 routed-expert body 内只合并一次，同时保持原
   `AscendMoERunner`。
3. `native-hybrid-full-smoke-v003`（远端快照别名
   `native-hybrid-full-v003`）在 `cudagraph_mode=NONE` 下完成完整 48 层
   eager smoke：五类 canary、32-step Decode、取消和同进程 reuse 均通过。
   这证明 Phase I eager 计算正确，不构成性能晋级证据。

## 图恢复结论

### PIECEWISE

正式停止修复。连续诊断已把失败从一个局部算子推进到另一个相互独立的
模型区域：

- `v011` 在冻结 PIECEWISE runtime 中把实际 32-row tensor 与错误的动态
  scalar/range 绑定组合，最终生成非法 reshape；
- `v013` 绕过完整 GDN boundary 后，在 Qwen HyperConnection
  `hc_combine` 又出现同类错误：实际 32-row tensor 被要求 reshape 为
  `[33, 4, 2560]`。

因此现有证据支持“冻结 PIECEWISE 栈的通用动态 scalar/shape ABI
绑定问题”，不支持继续逐算子增加 shape 补丁。它不是 GDN-only、KV-only
或 routed-expert-only 问题。特别地，不能因单-page 场景增加
`unsqueeze(0)` 就宣称 KV 或完整图问题已经解决。

### Breakable

冻结 vLLM-Ascend 栈明确禁用该能力。本轮不升级整个软件栈，也不自行实现
新的 breakable graph runtime。

### FULL / FULL_DECODE_ONLY

冻结 FULL replay 只调用已捕获的 NPU graph，不重新进入 Python runnable。
当前 CPU submit/sync 位于 quant-method `apply()` 中，因此 replay 无法为
每个 token/layer 产生新的 CPU work ID。当前 seam 与 FULL 的正确性合同
不兼容；继续强行 capture 会复用 capture-time CPU result 或迫使整段退回
eager。

## Phase I disposition

- Eager：`PASS_NATIVE_HYBRID_EAGER_CORRECTNESS`
- PIECEWISE：`LOCKED_GENERAL_DYNAMIC_SHAPE_ABI_BLOCK`
- Breakable：`UNSUPPORTED_BY_FROZEN_STACK`
- FULL-compatible：`CURRENT_CPU_EAGER_BREAK_INCOMPATIBLE`
- 请求性能：`NOT_RUN_WITHOUT_GRAPH_SAFE_CANDIDATE`
- Phase I 总结：`ROUND18_NATIVE_HYBRID_WORKS_NO_GRAPH_RECOVERY`

这些结论冻结后，不再运行 PIECEWISE 修补、PIECEWISE smoke 或 Phase I
请求 ABBA。后续设备实验仅属于 Phase II Native expert pool。

## Phase II 入口

Phase II 只在三个门上推进：

1. fresh Native route 的 working-set / transfer economics；
2. 不重建完整 parent 的固定 Physical-Q32 slot 写入；
3. 在 graph replay 之外更新同地址权重与 mapping、且 replay 消费新内容的
   graph-safe mutation。

任一门失败即形成对应的 working-set、backing-format 或 graph-compatibility
结论，不退回 Phase I。
