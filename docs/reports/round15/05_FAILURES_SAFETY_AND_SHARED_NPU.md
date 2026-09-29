# Round15 故障、安全闭环与共享 NPU 策略

## 1. 故障不是被删除的噪声

Round15 保留了所有消耗预算或改变下一步判断的失败。控制错误可以修复，但原 FAIL 不改名；资源失败不能写成性能负结果；候选未命中不能拿 eager fallback 的性能冒充候选。

| Run | 类型 | 事实 | 后续处理 |
|---|---|---|---|
| v331 | 控制，模型未启动 | 继承预算 stage gate 拒绝 Round15 | 修正 Round15 专用合同；容器精确删除 |
| v333 | 请求后控制失败 | 完成四请求，但缺五桶 producer | 补接已有 census；不接纳该块为 P0 PASS |
| v335 | 测量有效、supervisor FAIL | profiler 名称、训练 PID 识别、已删除容器复查错误 | 严格 salvage 原始测量；原 FAIL 保留 |
| v348 | 模型前控制失败 | mutable manifest 集合不一致 | 修 builder/source gate，无设备测量 |
| v352 | 资源失败 | cold load 时 NUMA4 瞬时低于门槛 | guard fail-closed、清理；不写 NO_GAIN |
| v355 | 模型前控制失败 | 旧 Layerwise source gate 拒绝已批准 KTEP patch | 只修 gate，保留失败 |
| v356 | 集成失败 | 生产 route IDs 为 int32，图策略只接 int64，真实命中 0 | 冻结 int32 合同，重新做组件 |
| v358 | 数值自检失败 | capture 后未先 replay，首层 relative L2=1.0 | 消费前 fail-stop；写回归，最小修复 |

## 2. 数值和状态安全

A 线要求 selected set exact、downstream relative L2、重复 hash、变化输入、20 轮生命周期和未资格分支执行前 fallback。B 线要求 graph/eager exact、20 个变化输出、固定 buffer 身份、首次 capture 自检、request epoch 不生成新图以及 reset 后资源回到上界。

v358 的关键安全点是错误在 layer0 的 self-check 中被阻断，没有向后续层发布。修复没有通过放宽误差或跳过 self-check；v360 的 48 个首次 capture 检查最大 relative L2 为 0。

取消请求时，runtime 停止新增候选工作，等待已有工作退出后回收；同进程 reuse 用新的请求身份重新验证输出。A 和 v360 均通过 cancel/reuse。

## 3. 共享 NPU 策略

本轮采用受控共卡、身份隔离、资源预留、设备工作串行提交：

- KT 只使用物理 NPU 12（`card 6 / chip 0`）和 NUMA 2/3/4/7。
- 已登记训练容器的完整 ID/cgroup 是共享身份依据；训练 PID 可以变化，但必须属于该 cgroup。
- 任何未知 NPU PID 或未知容器 fail-closed，不停止、修改或挤占外部任务。
- 每次组件或模型启动前做 fresh 两轮各 60 秒的 CPU/NUMA、Host 内存、HBM、进程和 cgroup 准入。
- 每个候选 NUMA node 启动前至少保留 48 GiB；整卡 HBM 继续使用冻结总帽并保留至少 8704 MiB headroom。
- 本项目同一时间只运行一个自有设备进程，不并行冷加载两个模型或组件。
- guard 只监督本项目容器、owned HBM、Host cgroup、OOM、退出和 cleanup；不把共享训练计入 owned。

在 Round15 最终只读核验中，A3 出现新的外部 `qwen38-vllm-*` 容器。由于它们不属于 Round15，执行器没有停止或修改它们，也没有再提交设备实验。

## 4. 容器和进程回收顺序

每个实验块都遵守：

```text
请求/组件完成或 fail-stop
  → guard 取得自有进程退出 ACK
  → 删除精确的自有容器
  → 归档原始结果、supervisor、admission 和事件
  → 校验本地/远端 SHA
  → 审计登记只读文件的全机 fd 与 mmap
  → POSIX_FADV_DONTNEED
  → fincore 和 NUMA 回执
```

没有执行全局 `drop_caches`。缓存退休只作用于已登记、只读、无人引用的 `.gguf` 和 `.safetensors`；必须放在完整 SHA 之后，避免校验再次把几百 GiB 文件读回 page cache。

v357、v358、v360 的 post-archive 退休分别有独立 receipt。v358 释放约 64.18 GB，v360 的启动前模型定向退休释放约 248.20 GB，结束后又释放约 64.18 GB。字节数是 file cache resident bytes，不是模型运行内存节省或 NPU HBM。

## 5. 不得从安全证据推导的结论

- guard PASS 不等于性能候选晋级。
- 当前设备上没有 Round15 自有进程，不等于下一次启动可以跳过 admission。
- v356 输出正确不等于 Graph 正确，因为它走的是 eager fallback。
- v357 组件 exact 不等于完整模型所有状态天然安全；v360 的真实捕获、自检、取消和 reuse 才补上请求证据。
- 定向 DONTNEED 释放 page cache 不代表降低模型工作集；下次读取仍会重新产生 cache。

资源与预算原始记录见 [ROUND15_EXECUTION_BUDGET.json](../../../artifacts/round15/ROUND15_EXECUTION_BUDGET.json) 和 [cache-retirements](../../../artifacts/round15/cache-retirements/)。
