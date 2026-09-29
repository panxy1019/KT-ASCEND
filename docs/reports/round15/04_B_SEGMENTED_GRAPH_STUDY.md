# Round15 B 线：Decode batch1 NPU 分段图

## 1. 目标和非目标

B 线试图减少一个固定 NPU 子调用的 Host 下发和算子间空隙。它不是整模型 Graph，也不是把异构执行硬塞入一个静态图。设计首先从依赖关系出发：只有当前 token 的 resident NPU MoE 计算在输入、route 和权重就绪后可以作为一个独立 segment；CPU Expert、Host gather、状态和取消仍然需要 Python/runtime 管理。

最终捕获边界为：

```text
UnquantizedFusedMoEMethod.apply
  └─ resident Q32 BF16/NZ29 NPU MoE subcall only
```

明确排除：CPU Expert submit/wait/merge、PLE、QSA、GDN、KV state、request epoch、cancellation、scheduler 和 Prefill。没有跨层 deferred，没有提前消费 CPU 结果，没有删除同步。

## 2. 图输入与 fail-closed 合同

首版只接受以下精确组合：

- `forward_mode == DECODE`；
- batch1，hidden shape `(1, 2560)`；
- Top-10 route；
- hidden BF16、weights FP32；
- 生产 `topk_ids == torch.int32`；
- TP1、Q32、`UnquantizedFusedMoEMethod`；
- 无 bias，返回 FP32 contribution；
- `max_running_requests == 1`；
- layer 0–47。

任何 dtype、shape、provider、层范围、并发或模式漂移都在 capture/replay 之前 fallback。replay 一旦开始，不允许对同一 token直接再跑 eager，因为图中间可能已经执行过有副作用的设备工作。

每层图对象持有固定输入 buffer、route IDs、weights 和 output。每次 replay 都复制当前 token 的真实数据；capture 时的路由不能冻结成常量。图缓存不以 request epoch 建新实例，避免每请求泄漏一套图。

详细合同见 [GRAPH_CONTRACT.md](../../../artifacts/round15/GRAPH_CONTRACT.md) 和 [round15_b_segment_policy.py](../../../artifacts/round15/round15_b_segment_policy.py)。

## 3. 组件验证

v347 首先证明 torch_npu 2.9 的 NPUGraph API 能捕获该 resident MoE 区域，但使用的是探索性 int64 route IDs。v356 的真实模型随后揭示生产 Decode 使用 int32，因此 v347 只能作为 API feasibility，不能代表生产 dtype。

v357 以 int32 重建组件并完成 20 个变化输入周期：

| 项目 | 结果 |
|---|---:|
| 最大 relative L2 | 0.0 |
| 最大 absolute error | 0.0 |
| 变化输出 hash | 20/20 |
| repeat hash | exact |
| eager Host 中位数 | 0.482993 ms |
| graph Host 中位数 | 0.346946 ms |
| Host 改善 | 28.17% |
| eager device 中位数 | 0.363560 ms |
| graph device 中位数 | 0.241120 ms |
| Device 改善 | 33.68% |
| reset 后 allocated 增长 | 4096 bytes |

候选计时包含每次 replay 前对固定输入 buffer 的 copy。v357 是一个生产 dtype 组件进程，不被写成两块组件 ABBA。

## 4. 全模型集成中的两个关键失败

### v356：图开关打开但实际零命中

v356 完成 14 个请求，质量和取消/复用通过，但 capture/replay 都是 0。根因是策略只接受 int64，生产 route IDs 是 int32。该运行只证明 side-effect-free eager fallback 正确，其性能不属于候选。

修复方式是把生产合同冻结为 `EXACT_INT32_ONLY`，并让 int64 fail-closed，而不是同时兼容两个未证明 dtype。

### v358：首次 capture 输出未经过 replay

v358 在第一个 Decode、layer0 的 capture self-check 中出现 relative L2 `1.0`、max abs `0.0085735321`。自检在输出被下游消费前终止请求，错误结果没有发布。

根因是组件路径在 capture 后总会执行 `graph.replay()` 再读取 output，而全模型首次 capture 路径直接读取 `graph.output`。NPUGraph capture 记录图，不保证 capture 阶段的 output 已包含可消费结果。

修复先写回归测试，再把首次候选结果改为：

```python
candidate = graph.replay(dispatch_output)
```

然后同步、自检、发布。修复没有扩大图边界或放宽误差。

## 5. v360 真实模型结果

v360 是修复后的最终候选：

- 14/14 generation attempts 完成；
- 五类 canary、多步 Decode、取消、复用通过；
- 13 组可对齐输出 token 与 v354 完全相同；
- logprob 有限；
- 48 层 capture，48 个 capture event；
- 48 层 replay，192 个 replay marker；
- capture self-check 最大 relative L2 为 0；
- Prefill 保持 eager；
- guard ACK、容器删除和缓存退休通过。

因此 B 的“实现是否真正接通”答案是肯定的。请求性能如下：

| Workload | 指标 | v354 eager | v360 graph | 变化 |
|---|---:|---:|---:|---:|
| W1 128/16 | TTFT | 2.9001 s | 2.7849 s | 改善 3.97% |
| W1 128/16 | TPOT | 0.27181 s | 0.26438 s | 改善 2.74% |
| W1 128/16 | E2E | 6.9774 s | 6.7507 s | 改善 3.25% |
| W1 128/16 | 串行吞吐 | 2.2931 tok/s | 2.3701 tok/s | 改善 3.36% |
| W2 128/128 | TTFT | 2.7342 s | 2.8789 s | 退化 5.29% |
| W2 128/128 | TPOT | 0.26878 s | 0.26555 s | 改善 1.20% |
| W2 128/128 | E2E | 36.8694 s | 36.6035 s | 改善 0.72% |
| W2 128/128 | 串行吞吐 | 3.4717 tok/s | 3.4969 tok/s | 改善 0.73% |

图池使 owned 和 whole-card HBM 峰值各增加 984 MiB。当前捕获区域在 P0 中每次约 0.4 ms，组件节省约 0.12–0.14 ms；它在整个 token critical path 中占比有限，所以局部 28–34% 改善只转化为约 1% 的 W2 TPOT/吞吐变化。

## 6. 去留结论

```text
ROUND15_B_DIRECTIONAL_SMALL_GAIN_BELOW_PROMOTION_GATE
```

B 被判定为“实现正确、真实命中、请求小幅正向，但不足以晋级”。原 Decode 请求门要求 W2 TPOT 和完整串行吞吐都改善至少 10%，且保护指标不显著退化；v360 远低于收益门，W2 TTFT 还退化超过 5%，并增加 HBM。因此默认 Graph 仍为 OFF。

机器可读证据：

- [B_COMPONENT_RESULT.json](../../../artifacts/round15/B_COMPONENT_RESULT.json)
- [B_GRAPH_BOUNDARY_AND_RESULT.json](../../../artifacts/round15/B_GRAPH_BOUNDARY_AND_RESULT.json)
- [B_REQUEST_DIRECTIONAL_RESULT.json](../../../artifacts/round15/B_REQUEST_DIRECTIONAL_RESULT.json)
- [图实现](../../../artifacts/round15/round15_b_resident_graph.py)
- [首次 replay 回归](../../../artifacts/round15/test_round15_b_first_capture_replay.py)
