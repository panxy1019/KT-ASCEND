# Round 4A.5：P2 首次差异捕获

状态：`FIRST_DIVERGENCE_CAPTURED`（尚未完成根因归因）

## 冻结条件

- P2 placement、GGUF、模型 revision、TP1、BF16 与 CPU/NPU ownership 均未改变；
- NPU0 only，Graph/Deferred/Dynamic/MTP/Speculative 均关闭；
- `SGLANG_KT_HYBRID_NO_CPU_STREAM=1`，因此这不是 dual-stream overlap-only 测试；
- 为使两次相同 history 都实际执行完整 prefill，诊断服务额外使用
  `--disable-radix-cache`。这只影响 cache reuse，不是 production fix 或 P2 验收配置；
- `v_en_01` 的稳定首个生成 token `185` 被显式追加至原始 `input_ids`。两次均对
  这个固定的 8-token teacher-forced history 请求一个 token；每个 request 内部有
  prefill/decode 两次 forward，因此比较每层 pass 0 与 pass 2（两个 8-token prefill）。

## 前向证据

比较发现的第一个不同项是：

```text
layer = 26
stage = cpu_output
```

Layer 1、9、17 的 captured CPU input、router IDs/weights、CPU output、NPU partial 与
merged output 均为 exact byte match。Layer 26 的 evidence 为：

| Stage | Exact |
|---|---:|
| hidden_states | yes |
| topk_ids / topk_weights | yes / yes |
| NPU premerge partial | yes |
| CPU output | no |
| merged routed output | no |

所有已捕获值均 finite。故当前证据排除了在 Layer 26 CPU execution 之前的 layer
input、routing 和 NPU grouped-MoE partial；它没有单独证明 CPU kernel、CPU task
lifetime、host output buffer 或 H2D ownership 中的任何一种就是根因。

## H1：shared CPUInfer A/B

新增的默认关闭开关 `KT_DEBUG_PER_WRAPPER_CPUINFER=1` 为四个 wrapper 分别创建
CPUInfer/WorkerPool。日志确认创建了四个不同实例。保持其余 P2 sequential-control
条件不变，对 `v_en_01` 进行 10 次 64-token 重复：仍得到 8 个 output hash，
`repeat_exact=false`。

因此“多个 wrapper 共享 CPUInfer/task queue”不是充分根因，H1 作为单变量解释被拒绝。
该开关仅用于诊断，不是 production fix。

## 下一步

对 Layer 26 依次做 default-off 的单变量 A/B：per-forward CPU input/output buffer、
output poison/zero/write-coverage assertion，以及 CPU task completion 到 H2D consumer
的 generation/ownership 记录。任何同步只用于定位，不能作为永久修复。
