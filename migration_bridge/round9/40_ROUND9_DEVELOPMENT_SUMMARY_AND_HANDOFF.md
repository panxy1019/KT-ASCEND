# KT-ASCEND Round9 开发总结与后续交接

更新日期：2026-09-21

代码分支：`codex/round9-heterogeneous-working-set`

本轮最终资源矩阵提交：`25ccdda5d`

执行主机：`110.123.0.3`

执行设备：Ascend 910 物理卡 15（本轮按独占模式准入）

## 1. 本文目的

本文是 Round9 当前开发状态的总入口，供后续开发、复查和跨会话接手使用。更细的实现说明和问题复盘分别见：

- [41_ROUND9_IMPLEMENTED_CAPABILITIES_AND_RESULTS.md](41_ROUND9_IMPLEMENTED_CAPABILITIES_AND_RESULTS.md)：实现了什么、如何工作、实际效果如何。
- [42_ROUND9_PROBLEMS_AND_RESOLUTIONS.md](42_ROUND9_PROBLEMS_AND_RESOLUTIONS.md)：本轮遇到的每个主要问题、根因、修复和验证证据。
- [38_CANDIDATE_FULL_MODEL_TRIALS.md](38_CANDIDATE_FULL_MODEL_TRIALS.md)：全模型试验的逐次执行记录。
- [39_CANDIDATE_RESOURCE_MATRIX_DELIVERY.md](39_CANDIDATE_RESOURCE_MATRIX_DELIVERY.md)：六个最终组合的资源矩阵结果。
- [CURRENT_EXECUTION.md](CURRENT_EXECUTION.md)：按时间倒序记录的执行状态。

本总结聚焦用户调整方向之后的“开发优先”阶段，即 C128、CPU 低比特和 request-boundary Dynamic 的框架、算子、设备门禁和完整模型资源验证。Round9 更早的模型下载、源码语义审计和基础模型移植仍保留在 `00` 至 `34` 号文档中，不在这里重复展开。

## 2. 目标与约束

本轮目标是为 Qwen3.8-Flash-Next 在 KT-ASCEND 上建立三类候选能力：

1. C128：将 CPU 专家计算扩展为四个 32 worker NUMA 子池，总计 128 个物理 worker。
2. CPU 低比特：支持 Q8_0 和 MXFP4 的 CPU 专家权重，NPU 提升仍使用原始 BF16 权重。
3. Dynamic：在串行 TP1、batch1 的请求边界完成专家迁移，保证一个请求只观察一个一致代次。

整个开发和验证过程遵守以下约束：

- 不恢复 Round8 已取消的性能测试。
- 不继承旧模型、旧候选或不完整 256 题基线的资格。
- 不修改冻结基线运行时；候选实现使用隔离 overlay、隔离 native build 和独立证据。
- 不自动恢复 Round7 或 Round9 定时任务。
- 不合并 `main`，不打 tag。
- 资源结论只适用于本次独占物理卡 15 的准入；后续共享使用必须重新做 CPU、NUMA、HBM 和外部进程准入。

## 3. 当前结论

Round9 的 C128、Q8_0、MXFP4 和 request-boundary Dynamic 框架已经从 host-only 原型推进到真实算子、真实设备迁移和完整模型资源矩阵。

六个最终组合全部完成完整模型资源/系统门禁，其中五个组合同时通过四项严格内容 canary。BF16 Dynamic 的资源/系统轴通过，但第四项严格 JSON canary 失败，所以它的候选整体状态仍为失败。

| CPU 专家存储 | 放置策略 | 资源/系统 | 严格内容 | 候选整体结果 |
|---|---|---|---|---|
| BF16 | static | PASS | PASS 4/4 | PASS，限定为功能/资源范围 |
| MXFP4 | static | PASS | PASS 4/4 | PASS，限定为功能/资源范围 |
| MXFP4 | request_boundary | PASS | PASS 4/4 | PASS，限定为功能/资源范围 |
| Q8_0 | static | PASS | PASS 4/4 | PASS，限定为功能/资源范围 |
| Q8_0 | request_boundary | PASS | PASS 4/4 | PASS，限定为功能/资源范围 |
| BF16 | request_boundary | PASS | FAIL 3/4 | `FAIL_CANDIDATE_CONTENT_CANARY` |

矩阵顶层状态为 `COMPLETED_WITH_CANDIDATE_FAILURES`，矩阵进程按设计以非零状态退出。不能把“六个组合都运行完成”表述为“六个组合都通过”。

## 4. 本轮完成的开发阶段

### 4.1 候选框架阶段

建立了独立候选配置和 capability contract，明确校验 worker 数量、NUMA 节点、CPU 专家存储类型、放置策略、native 能力、银行身份和 Dynamic 设备门禁。任何未支持组合都必须显式拒绝，不能静默回退到旧精度、旧 worker 数或静态模式。

初始 host/ARM CPU 验证覆盖配置、分区、codec、GGUF 写入与读取、Dynamic 状态机、Tensor adapter、隔离 overlay 安装和工作流。这个阶段只证明框架和 CPU seam，未把 host 测试包装成设备资格。

### 4.2 C128 和 native 算子阶段

完成四个 32 worker 子池及四个 160 宽中间切片，修复 Q8_0 ARM 激活布局、160/320 尾部支持和低比特 F32 TP partial 合并。最终验证覆盖 C64/C128、Q8_0/MXFP4、多个 token 长度和重复执行。

实际调度器线程证据显示：完整模型每次运行均有 128 个单 CPU 绑定线程，且对应 128 个不同 CPU，不使用 SMT sibling 冒充物理 worker。

### 4.3 完整低比特银行阶段

生成并验证 48 层完整 BF16、Q8_0 和 MXFP4 CPU 专家银行。验证不只检查目录或 catalog，而是重新读取每个文件、计算完整 SHA-256、检查 GGUF tensor 类型和形状、确认文件集合及原始模型来源。

Q8_0 和 MXFP4 只用于 CPU routed expert backing。Dynamic 向 NPU 提升专家时使用原始 BF16 来源，不从低比特银行反量化后提升。

### 4.4 Dynamic 请求边界阶段

建立请求 lease、generation、route observation、完整计划、设备复制、map/mask 发布和失败毒化状态机。迁移只在请求相关 KV/GDN/PLE 等消费者完成清理后提交；复制或发布失败会进入 `BROKEN_RELOAD_REQUIRED`，不会尝试仅回滚元数据来掩盖设备权重已被覆盖的问题。

设备门禁最终覆盖 64 个 resident expert 的反转与恢复、三代 promote/demote/promote、9 组 KTEP 测试和三次重复。完整模型的三个 Dynamic 组合各记录 18 个连续 generation，并覆盖全部 16 个测试请求和 2 个启动健康请求。

### 4.5 完整模型资源矩阵阶段

每个最终组合均独立完成：

1. 模型、候选 overlay、native runtime、profile、48 层银行和设备门禁的 SHA 身份绑定。
2. 128 物理 worker、NUMA 2/3/6/7、136 CPU quota、640 GiB RAM、物理卡 15 和空闲 HBM 的新鲜准入。
3. 首次真实 HTTP prefill/decode。
4. 8 次系统请求；static 要求成对输出轨迹严格一致，Dynamic 要求有限输出和每个请求的边界事务完成。
5. 四项固定已知答案 canary。
6. 实际 3840 输入 token 加 256 输出 token 的 4096 token 资源请求。
7. 主动中止及中止后的状态复用。
8. 受控停止、监控峰值封存、证据哈希和 Dynamic 事务审计。

## 5. 资源配置与实测峰值

独占模式采用：

- 4 个 NUMA 子池，每个 32 worker。
- 128 个 worker 加 8 个辅助 CPU，CPU quota 为 136。
- 容器 RAM 上限 640 GiB，无额外 swap 预算。
- PyTorch candidate allocator 上限 43,008 MiB（42 GiB）。
- owned process HBM guard 45,056 MiB（44 GiB）。
- admission model HBM budget 46,080 MiB（45 GiB）。
- 额外保留 8,192 MiB，并叠加启动时实际板卡系统基线。

| 组合 | Owned HBM 峰值 | Board HBM 峰值 | Host cgroup 峰值 |
|---|---:|---:|---:|
| BF16 static v10 | 39,856 MiB | 42,624 MiB | 309,354,479,616 bytes |
| MXFP4 static v19 | 39,854 MiB | 42,622 MiB | 115,188,797,440 bytes |
| MXFP4 Dynamic v20 | 41,060 MiB | 43,828 MiB | 140,692,103,168 bytes |
| Q8_0 static v21 | 39,854 MiB | 42,621 MiB | 268,915,621,888 bytes |
| Q8_0 Dynamic v22 | 41,060 MiB | 43,827 MiB | 269,033,795,584 bytes |
| BF16 Dynamic v23 | 41,062 MiB | 43,829 MiB | 253,952,180,224 bytes |

这些数值是各自受控停止之前的实测峰值。宿主 cgroup 数值包含 CPU 银行和文件页，不可解释为 NPU HBM，也不适合直接跨试验比较性能或效率。

## 6. 当前代码和证据入口

主要实现：

- `kt-kernel/operators/llamafile/tp_slices.hpp`：多池切片和尾部 capability。
- `kt-kernel/operators/llamafile/moe.hpp`：低比特 grouped/forward-many 路径。
- `kt-kernel/operators/moe-tp.hpp`：TP partial 和归并。
- `kt-kernel/python/utils/llamafile.py`：Python/native capability 对接。
- `migration_bridge/round9/runtime/candidate_config.py`：候选 profile 和 capability 校验。
- `migration_bridge/round9/runtime/candidate_bank.py`：48 层银行和来源身份验证。
- `migration_bridge/round9/runtime/dynamic.py`：generation 状态机。
- `migration_bridge/round9/runtime/dynamic_adapter.py`：批量原始 BF16 提升和原位 KTEP map/mask 更新。
- `migration_bridge/round9/runtime/dynamic_boundary.py`：SGLang 请求清理边界接线。
- `migration_bridge/round9/runtime/expert_source.py`：有界、只读的原始 BF16 专家来源。
- `migration_bridge/round9/tools/launch_candidate_model.py`：候选完整模型启动和容器资源限制。
- `migration_bridge/round9/tools/monitor_owned_server.py`：只停止本轮 owned container 的资源监控。
- `migration_bridge/round9/tools/finalize_candidate_trial.py`：资源轴与内容轴分离的结果封存。

最终证据：

- [evidence/physical15/20260920-candidate-matrix-v4/delivery-index.json](evidence/physical15/20260920-candidate-matrix-v4/delivery-index.json)：六份 summary、audit、矩阵脚本、审计程序和最终设备快照的绑定索引。
- [evidence/physical15/20260920-candidate-matrix-v4/status.json](evidence/physical15/20260920-candidate-matrix-v4/status.json)：最终矩阵状态。
- [evidence/physical15/20260920-candidate-matrix-v4/independent-review.json](evidence/physical15/20260920-candidate-matrix-v4/independent-review.json)：独立只读复核结果。
- [evidence/physical15/20260920-candidate-matrix-v4/final-device-state.json](evidence/physical15/20260920-candidate-matrix-v4/final-device-state.json)：停止后的设备和进程快照。

## 7. 当前未取得的资格

以下项目仍未完成，后续不能从本轮结果推断：

- 完整 256 题精度基线和候选对比。
- 低比特相对 BF16 的模型级质量等价。
- 候选进程的 clean restart 和重启后等价。
- Dynamic 与相同放置、相同请求历史的 fresh-static 完整模型数值 oracle。
- BF16 Dynamic 严格 JSON 内容 canary 通过。
- TTFT、TPS、延迟或任何性能排序。
- 生产部署、长期稳定性和共享卡资格。

## 8. 最终停止状态

本轮一次性矩阵已经结束。停止后采样确认：

- 物理卡 15 无运行 NPU 进程，HBM 回落至 2,878 MiB 系统基线。
- 没有 Round9 candidate container、server、monitor 或 matrix 进程残留。
- `mzy-train` 和 `kt-r7-a3-return-v1` 未被本轮停止或修改。
- Round7 和 Round9 自动任务仍为 `PAUSED`。

## 9. 后续建议顺序

1. 为 BF16 Dynamic 建立“相同初始权重、相同 resident placement、相同请求历史”的 fresh-static 对照，判断代码围栏差异是否与迁移数值路径相关。
2. 在不放宽严格 canary 的前提下决定是否需要结构化输出约束；如引入 grammar，应作为新的功能路径单独验证，不能重写本轮历史结果。
3. 完成冻结 baseline 后 128 题，形成完整 256 题基线，再对通过资源门禁的候选做同一题集质量评估。
4. 进行候选 clean restart 和重新加载身份验证。
5. 若用户重新要求共享物理卡 15，先重新测量训练峰值并重新计算 allocator、owned guard、model budget 和 board reserve；不得直接沿用本轮独占参数。
6. 性能工作只有在用户再次授权后才能开始。

## 10. 交接时必须保留的判断

- 五个候选在限定功能/资源范围内 PASS；BF16 Dynamic 候选整体 FAIL。
- BF16 Dynamic 的第四项回答在语义上包含正确对象，但不满足严格 JSON 格式；这仍是内容失败。
- Dynamic 的有限输出、请求清理和复用成功不等于完整模型数值等价。
- 低比特 native 局部误差通过门禁不等于模型质量等价。
- 所有新测试都应使用新的证据目录，不能覆盖现有原始结果。
- 不自动恢复定时任务，不继承旧资格，不把资源测试改写成性能结论。
