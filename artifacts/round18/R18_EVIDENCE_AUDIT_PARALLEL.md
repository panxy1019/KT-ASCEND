# Round18 并行证据与终态门审计

> 历史快照说明：本文件是 v013 执行前的只读阶段审计，保留当时的门禁判断，
> 不代表 Round18 最终状态。后续 v013、v016、v017、v018、Phase II 经济性与
> graph-safe gate 已补齐；当前终态请以 `FINAL_STATUS.json`、
> `CURRENT_EXECUTION.md` 和 `ROUND18_FINAL_REPORT_ZH.md` 为准。

审计时间：2026-09-30T03:21:28Z
审计性质：本地只读证据审计；没有连接设备、没有启动模型、没有访问网络。
审计范围：`artifacts/round18/`、其 `evidence/` 归档、当前 Round18 源码和测试。
快照提醒：本报告记录审计时刻的状态。`v013` 只有冻结运行脚本，审计时尚无本地 evidence 目录或结果，故仍按 `NOT_RUN` 处理。

## 1. 审计结论

Round18 已经形成两条可以独立成立的正面证据：

1. **P1 Native resident partial MoE 已通过。** 冻结的 Ascend 原生 selector 和 `AscendUnquantizedFusedMoEMethod` 在 512 个逻辑专家下能够使用 Q32 global-to-slot map；map 中 `-1` 的 nonresident occurrence 被跳过，原生 NPU 分支产生 resident-only partial routed result。该结论有 `native-partial-v004` 的真实设备证据支持。
2. **P2 Native-preserving Hybrid eager 已通过组件和完整模型 smoke。** CPU complement 与原生 Q32 partial 在 routed result 边界只合并一次，48 层仍绑定具体 `AscendMoERunner`。`native-hybrid-full-smoke-v003` 在 `cudagraph_mode=NONE` 下通过五类 canary、32-token Decode、取消与 same-process reuse。

但 Round18 **尚不能进入任何完成或性能晋级终态**：

- PIECEWISE 尚未成功完成 engine startup/profile replay，graph correctness 未通过；
- NONE 与 GRAPH 的请求级对照没有运行；
- BEST18 与冻结 S0 的 fresh ABBA 没有运行；
- Phase II 只有 Round9 路由的历史敏感性预检，没有 fresh Round18 W1/W2/W4 route trace，也没有 backing format、固定 slot、graph-safe residency 或完整请求证据；
- `CURRENT_EXECUTION.md` 和 `FINAL_STATUS.json` 都落后于实际证据，仍把 P2 full-model eager 写成 `NOT_RUN`，不能作为最新事实源。

因此，审计时唯一安全的总状态是：

```text
ROUND18 = ACTIVE
P0 = PASS_RESOLVED_NATIVE_SEAM
P1 = PASS_NATIVE_RESIDENT_PARTIAL
P2 = PASS_NATIVE_HYBRID_EAGER_FULL_MODEL_SMOKE
P3_GRAPH_RECOVERY = NOT_QUALIFIED
PHASE_I_REQUEST_PERFORMANCE = NOT_RUN
PHASE_II_EXPERT_POOL = NOT_RUN
FINAL_TERMINAL_STATUS = NOT_REACHED
```

## 2. 逐阶段证据矩阵

| 阶段 | 主要证据 | 已证明 | 未证明 / 边界 |
|---|---|---|---|
| P0 Native seam | `R18_RESOLVED_MOE_SEAM.md` | 完整 Top-K、route weight、shared/finalize 的实际所有权；Q32 owner 可在不替换 runner 的窄 seam 注入；冻结 image、vLLM/vLLM-Ascend commit 和关键源码 hash 已登记 | 不是请求正确性或性能证据 |
| P1 resident partial | `R18_NATIVE_PARTIAL_MOE.json`；`evidence/native-partial-v004/receipt.json` | logical512/physical32；all-resident、all-nonresident、mixed 1/9、5/5、9/1、多 token、ties；每次原生 selector 只调用一次；最大 relative L2 `0.0026470786 < 0.01`；20 次 lifecycle hash 稳定；allocator steady `314,576,384` bytes，peak `331,745,280` bytes | 只证明 resident partial；没有 CPU complement、完整层、完整模型或性能结论 |
| P2 Hybrid component | `R18_HYBRID_EAGER_COMPONENT.json`；`evidence/native-hybrid-v001/receipt.json` | C128/MXFP4 CPU complement 与 native Q32 partial 的 all-resident/all-CPU/mixed 合并；最大已列 relative L2 约 `0.00256`；20 次 lifecycle；CPU provider 为 `LlamafileMoEWrapper/MXFP4`，128 workers；original Top-K tensor/weight 只传递一次 | 组件通过不代表 runner 上层 shared/scaling/finalize 在全模型中没有其他接线问题，也不代表图兼容或收益 |
| P2 full-model eager | `R18_HYBRID_EAGER_RESULT.json`；`evidence/native-hybrid-full-smoke-v003/result.json`；`binding-receipt.json` | `enforce_eager=true`、graph mode NONE；48 层、每层 physical Q32、logical512；48 个实际 runner 均为 `AscendMoERunner`；五类 canary、Decode32、有限 selected logprob、取消后无 unfinished、reuse 输出 391；container exit 0 并移除 | 不是 W1/W2/W4 正式性能；没有 20 轮 full-model lifecycle；没有 graph correctness；不证明 FULL/PIECEWISE 可用；不证明优于 S0 |
| P3 PIECEWISE | `native-hybrid-piecewise-*` v004–v012 | 请求配置确实解析为 `CUDAGraphMode.PIECEWISE`；冻结栈明确记录 breakable graph 被强制关闭；多轮诊断已把 Host PLE trace、source-less shape guard、rotary/GDN symbolic shape 等阻塞逐步定位到可复现边界 | 没有一次 PIECEWISE engine startup + canary/decode/cancel/reuse PASS；无 replay generation、static buffer、state rollback、graph pool 或 close 的完整正确性凭证 |
| Phase II 路由预检 | `R18_ROUTE_WORKING_SET_HISTORICAL_PRECHECK.json` | 可复用的离线模拟工具和容量敏感性线索；例如 Q128 frequency-warm Belady 在这份历史 32-token tape 上 hit rate 约 `0.9361`，bytes/input-token 约 `603,340,800` | 数据来自 Round9 SGLang prefill，不是 fresh Round18 Native W1/W2/W4；文件自身状态明确为 `HISTORICAL_ROUTE_SENSITIVITY_NOT_ROUND18_QUALIFICATION`；不可用作经济性门、promotion 或 graph compatibility 结论 |

## 3. 运行证据和失败边界

### 3.1 P1 / P2 成功链

- `native-partial-v001`–`v003` 是保留的失败证据；`v004` 才是 P1 通过身份。
- `native-hybrid-v001` 是 P2 layer0 组件通过身份。
- full-model eager `v001` 和 `v002` 先后暴露 post-load child/module binding 与 logical512 communication 对 physical32 weights 的容量耦合；修复后的 `native-hybrid-full-smoke-v003` 才是完整模型 eager PASS 身份。
- `R18_NATIVE_PARTIAL_MOE.json` 与 `native-partial-v004/receipt.json` 字节完全相同，SHA-256 均为 `447be459...e57e`。
- `R18_HYBRID_EAGER_COMPONENT.json` 与 `native-hybrid-v001/receipt.json` 字节完全相同，SHA-256 均为 `1dffdd64...3ab1`。
- v003 的本地 evidence manifest 中 `result.json` SHA-256 为 `e26e502b...0658`，文件与 manifest 一致。

### 3.2 PIECEWISE 失败链

| Run | 结果 | 证据化根因 / 作用 |
|---|---|---|
| v004 | FAIL before graph qualification | custom-op/配置生命周期不合法，`Current vLLM config is not set`；不是图性能负结果 |
| v005 | FAIL during AOT trace | Host PLE bridge 内 `callable(getattr(...))` 无法被 Dynamo fullgraph trace；证明 CPU/Host PLE 必须成为合法 split/eager boundary |
| v006 | FAIL at registration | custom op schema 使用 `Any`，冻结 `infer_schema` 不接受；属于注册合同错误 |
| v007–v009 | FAIL during AOT guard construction | 在补齐 PLE split 后继续暴露 source-less symbolic shape / `chunk(2)` 相关 guard；后续静态 GDN 与 rotary split 使诊断前进，但仍未完成 engine startup |
| v010 | FAIL during PIECEWISE replay/profile | AOT 已进入 runnable，但 post-GDN view 使用 `[1584,128]`，实际输入为 `196608 = 1536×128`；动态 token SymInt 跨 GDN split 绑定错误 |
| v011 | 同一失败 | 即使把 runtime token 数来源改成 `z.size(0)`，仍发生相同 `[1584,128]` vs `1536×128` 错误；说明问题不是一行变量来源，而是 frozen PIECEWISE range ABI / symbolic binding across GDN boundary |
| v012 | 未进入模型 | immutable snapshot 漏带 `round18_integration/native_hybrid_live_worker.py`，34-test preflight 因 `FileNotFoundError` 失败；这是打包失败，不能当作图候选失败或新 runtime 证据 |
| v013 | 审计时 NOT_RUN | 只有 frozen run script；计划把完整原生 Qwen GDN forward 置于 opaque splitting op 后再做一次诊断。脚本存在不等于执行或通过 |

冻结栈在所有 PIECEWISE 日志中明确打印：

```text
Breakable cudagraph is force disabled on Ascend because DeepSeek V4
PIECEWISE cudagraph is not supported yet.
```

所以当前不能宣称 Breakable 已验证，也不能通过配置强开把它当作合法下一步。

### 3.3 源码身份边界

`R18_HYBRID_EAGER_RESULT.json` 冻结的 v003 source hash 中，以下四个当前文件仍与 v003 一致：

- `round18/native_hybrid_live_binding.py`：`1050ec18...dee`
- `round18/native_hybrid_quant_method.py`：`65c22b2b...932`
- `round18/native_runner_seam.py`：`2d2955bb...612`
- `round18_integration/native_hybrid_full_smoke.py`：`7a321051...c45`

当前 `round18_integration/native_hybrid_live_worker.py` 为后续 graph 诊断版本，SHA-256 是 `aebf1726...419`，与 v003 结果中冻结的 `37770c78...1f0` 不同。后续文档必须把 v003 immutable identity 与当前 graph candidate identity 分开，不能用当前 live worker 覆盖 eager PASS 的来源身份。

### 3.4 哈希审计

- v010、v011、v012 的 `SHA256SUMS.txt` 所列**证据主体文件**全部校验通过。
- 三份清单都把生成清单前的空 `SHA256SUMS.txt` 自身记录为 SHA-256 `e3b0c442...b855`；清单写入后，自引用项当然失败。这是清单生成方式问题，不是其他 evidence 文件损坏。
- 最终冻结时应生成一个不包含自身的 `EVIDENCE_SHA256.json` 或 detached manifest，避免自引用假告警。

## 4. 当前终态门逐项判断

### Phase I 允许终态

| 终态 | 当前可否写入 | 原因 |
|---|---|---|
| `ROUND18_NATIVE_HYBRID_GRAPH_PROMOTED` | 否 | graph correctness 与正式 NONE/GRAPH、S0/BEST18 对照均未运行完成 |
| `ROUND18_NATIVE_HYBRID_GRAPH_RECOVERED_BUT_NO_PROMOTION` | 否 | 尚无任何 graph PASS，更无与 S0 的请求级负结果 |
| `ROUND18_NATIVE_HYBRID_WORKS_NO_GRAPH_RECOVERY` | 暂时否 | eager 的确已工作，但 v013 仍是一个有依据且尚未取得结果的最小 graph boundary 实验；在该实验和明确 disposition 之前不能提前收口 |
| `ROUND18_NATIVE_PARTIAL_MOE_BLOCKED` | 否 | 与事实冲突；P1 已 PASS |

### Phase II 允许终态

| 终态 | 当前可否写入 | 原因 |
|---|---|---|
| `ROUND18_EXPERT_POOL_PROMOTED` | 否 | Phase II 未运行 |
| `ROUND18_EXPERT_POOL_WORKS_NO_PROMOTION` | 否 | 没有 component/full-model/request 实验 |
| `ROUND18_EXPERT_POOL_WORKING_SET_NOT_ECONOMIC` | 否 | 只有历史 Round9 route tape，不是 fresh Native route；也没有 safe-HBM Q 的正式上界输入 |
| `ROUND18_EXPERT_POOL_GRAPH_COMPAT_BLOCKED` | 否 | graph-safe residency cheap probe 未运行 |

### 总终态

- `ROUND18_COMPLETE_NO_PROMOTION`：不可写。两阶段都没有完整负结论。
- `ROUND18_PARTIAL_WITH_EXPLICIT_GAPS`：只有在执行链明确停止、并冻结必要缺口时才可写。审计时任务仍 ACTIVE，不能提前把中间态改成终态。

## 5. 下一条最短实验

最短、信息增益最高的下一步是**只运行已经冻结的 v013 一次**：

1. 先通过本地 34-test preflight，确认 v012 的 snapshot omission 已修复；
2. 将冻结的完整 Ascend Qwen GDN forward 作为 opaque splitting op 留在 PIECEWISE 图外，周围模型仍按 PIECEWISE 编译；
3. 只判定 engine startup/profile、resolved PIECEWISE、首次合法 replay 及既有 8-request smoke，不在同一 run 混入正式性能；
4. 记录 GDN custom op 的实际调用次数、输出/state 一致性，以及 graph 是否真正 capture/replay。

v013 的分支决策应在运行前固定：

- **若 PASS**：先生成 `R18_GRAPH_MODE_MATRIX.json` 与 `R18_GRAPH_CORRECTNESS.json`，再用单个 profiler-OFF fresh process 做 W2 三次探索筛选；达到相对 Hybrid-NONE 稳定 ≥5% 才进入正式 NONE/GRAPH 四进程。
- **若仍在同一 GDN symbolic ABI 失败**：停止继续做同类 source/view 微补丁，记录 frozen PIECEWISE 对当前 Qwen GDN 的 `GRAPH_RECOVERY_BLOCKED_AT_GDN_RANGE_ABI`。由于 breakable 被冻结栈强制关闭，不应升级软件栈或手写新的大型 graph runtime；转入 Phase II P7–P10 cheap gates。
- **若越过 GDN 后暴露一个新的普通非 traceable Host boundary**：只有它能通过现有原生 splitting-op 机制形成一个明确边界时才允许一次新修复；若演变为逐模块重建 graph runtime，应按合同转 Phase II。

如需区分“KT CPU eager break 导致 PIECEWISE 失败”与“冻结 Native Qwen 本身就有同一 GDN ABI 问题”，v013 失败后的最小控制实验是同一镜像、同一 Qwen、无 KT bridge 的 Native PIECEWISE startup/profile probe。该控制只回答归因，不提供 KT graph PASS，也不替代请求比较。

## 6. 必需交付缺口

审计时 `artifacts/round18/` 的合同文件状态如下：

| 文件 | 状态 |
|---|---|
| `R18_RESOLVED_MOE_SEAM.md` | PRESENT |
| `R18_NATIVE_PARTIAL_MOE.json` | PRESENT |
| `R18_HYBRID_EAGER_RESULT.json` | PRESENT |
| `R18_GRAPH_MODE_MATRIX.json` | MISSING |
| `R18_GRAPH_CORRECTNESS.json` | MISSING |
| `R18_GRAPH_REQUEST_COMPARISON.json` | MISSING |
| `R18_ROUTE_WORKING_SET.json` | MISSING；历史 precheck 不能替代 |
| `R18_BACKING_FORMAT_MATRIX.json` | MISSING |
| `R18_EXPERT_POOL_COMPONENT.json` | MISSING |
| `R18_EXPERT_POOL_GRAPH_COMPAT.json` | MISSING |
| `R18_EXPERT_POOL_REQUEST_COMPARISON.json` | MISSING |
| `ROUND18_BENCHMARK.csv` | MISSING |
| `ROUND18_RESOURCE_RETIREMENT.json` | MISSING；各 run 有局部 cache receipt，但没有全轮统一回执 |
| `ROUND18_FINAL_REPORT_ZH.md` | MISSING |
| `FINAL_STATUS.json` | PRESENT BUT STALE |
| `EVIDENCE_SHA256.json` | MISSING |

此外还需要更新但本并行审计按约束未修改的文件：

- `CURRENT_EXECUTION.md`：仍停在 `P2_NATIVE_HYBRID_COMPONENT_READY`，未反映 P2 component、full-model eager PASS 及 P3 graph diagnostics；
- `FINAL_STATUS.json`：仍写 `hybrid_eager=NOT_RUN`、`graph_recovery=NOT_RUN`，与已有 evidence 不一致。

## 7. 绝不能误写的结论

1. **不能写“Round18 已完成”。** graph、请求性能和 Phase II 均未闭环。
2. **不能写“Native Hybrid 已晋级”。** 当前只完成 eager correctness smoke，没有 fresh W1/W2/W4 性能门。
3. **不能写“PIECEWISE 或所有 Native Graph 永久不可用”。** 当前只证明冻结栈下、当前 Qwen/GDN 边界的多次尝试尚未通过；v013 仍未取得结果，FULL-compatible 也未有合法证据。
4. **不能写“CPU eager break 已被证明是 graph 失败的决定性边界”。** 当前最深失败发生在 Qwen GDN PIECEWISE symbolic shape ABI，尚未用 no-KT Native PIECEWISE control 排除基础栈/模型问题。
5. **不能把 v010/v011 的 `[1584,128]` 错误解释成单 page KV 已解决或只差 `unsqueeze(0)`。** 现有证据指向跨 GDN splitting op 的动态 token SymInt 绑定；报告中 `kv_contract_claimed=false`，没有证明多 page、完整有效 context 或 KV 状态矩阵资格。
6. **不能写“Breakable 已测试或可开启”。** 冻结 Ascend 栈明确 force-disabled。
7. **不能写“FULL_DECODE_ONLY 已恢复”。** 没有运行证据。
8. **不能把 v012 写成 graph 数值/性能失败。** 它在测试打包阶段即失败，未加载模型。
9. **不能用历史 route precheck 宣称 expert pool economic 或不 economic。** 它不是 fresh Native W1/W2/W4 route，也没有 graph-safe residency 和真实 transfer cost。
10. **不能把组件 20 轮写成 full-model 20 轮。** P1/P2 component 有 20-round reuse；full-model v003 是 canary/decode/cancel/reuse smoke。
11. **不能把 selected logprob 有限写成跨框架/跨精度等价。** v003 证明的是本轮冻结 canary 合同和有限性，不是全面质量或 bit-exact。
12. **不能用 `1/TPOT` 代替完整串行 throughput。** 当前更没有正式 TPOT/throughput 数据。
13. **不能用当前 live worker 的 hash 代表 v003。** v003 的 eager identity和后续 graph candidate identity已经不同。
14. **不能把证据清单的自引用失败写成 evidence corruption。** 除清单自身外，v010–v012 所列主体均通过 SHA 校验；最终应改用 detached manifest。

## 8. 审计执行与验证

本审计只执行本地、无设备检查：

- `python3 -m unittest discover -s tests/round18 -p 'test_*.py'`：34 PASS；
- `python3 -m py_compile round18/*.py round18_integration/*.py`：PASS；
- `git diff --check`：PASS；
- v010/v011/v012 detached body hash verification：除清单自引用项外全部 PASS；
- P1/P2 canonical artifact 与原 evidence receipt 的 byte comparison：PASS。

本报告没有改动 `CURRENT_EXECUTION.md`、`FINAL_STATUS.json` 或任何主代码，也没有创建、停止或检查远端进程。
