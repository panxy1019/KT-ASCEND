# Round 4A.5：并行执行 A/B（阶段性结论）

状态：`ROOT_CAUSE_NOT_YET_PROVEN`

## 已排除的单变量解释

在 `{1,17}` 最小 failing set、`v_en_01`、10 次 64-token sequential-control 下：

- 为每个 wrapper 创建独立 CPUInfer/WorkerPool 后仍有 8 个 output hash；
- 为每个 LLAMAFILE wrapper 私有化 scratch 后仍有 3 个 output hash；
- 为每个 TP MoE wrapper 私有化 merge-output storage 后仍有 2 个 output hash。
- 为每个 SGLang CPU-enabled layer 分配独立 NPU staging buffer 后，full P2
  `v_en_01` 仍有 7 个 output hash。

这些 buffer/ownership 路径存在实现风险，但都不能单独解释 P2 failure。因此它们不能作为
production fix。

## CPU worker-count A/B

将诊断服务中的 `KT_CPUINFER_THREADS` 从冻结值 16 降至 1：

| Placement / prompt | Result | Unique hashes |
|---|---|---:|
| `{1,17}` / `v_en_01` | exact, 10 repeats | 1 |
| full P2 / `v_en_01` | exact, 10 repeats | 1 |
| full P2 / `v_struct_01` | nondeterministic, 10 repeats | 2 |

在 `{1,17}` 上进一步限定 worker threshold：

| CPUInfer workers | Result | Unique hashes |
|---:|---|---:|
| 1 | exact, 10 repeats | 1 |
| 2 | exact, 10 repeats | 1 |
| 4 | exact, 10 repeats | 1 |
| 8 | nondeterministic, 10 repeats | 2 |

这证明 CPU worker parallelism 是 `v_en_01` failure 的必要触发条件，但不是 full P2
blocker 的唯一原因。`{1,17}` 的首次失败并发区间已收缩为 5--8 workers。较低 worker
count 仅是定位控制：它改变冻结的 CPUInfer worker count，不能作为 P2 production fix，
也不能据此进入 P3。

## 下一步

以捕获的真实 Layer 1/17 CPU-boundary tensors 直接交替 replay LLAMAFILE 100 次，在 8 workers
下仍是 exact。故固定 CPU inputs 上的 CPU kernel 本身不复现；问题必须依赖完整 SGLang/CANN
request lifecycle、跨 forward 状态或随 decode history 演进的交互。

下一步需要在完整失败 request 中用轻量 C++ generation/coverage instrumentation 标记每个 CPU
task 与 H2D consumer，分别对 `v_en_01` 与 `v_struct_01` 捕获实际分叉 forward。重点检查跨
forward task completion、buffer generation 和 write coverage，而不是再修改固定 CPU kernel。
