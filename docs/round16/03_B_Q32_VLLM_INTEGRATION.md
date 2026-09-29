# B 线：Physical-Q32、选择性加载与 Hybrid runner

## 1. 原阻塞

默认 EP1 将“本 rank 负责的专家集合”和“设备实际驻留的专家集合”视为同一个集合。对冻结模型而言：

- router/checkpoint 的逻辑专家数是 512；
- 当前进程仍然对全部 512 专家的模型语义负责；
- 但 NPU 每层只应物化 32 个 BF16/NZ29 resident experts；
- 其余 route occurrence 应由 C128/MXFP4 计算。

只把共享 `num_local_experts` 改成 32 会破坏 router、checkpoint、placement 和 runner 的共同假设；先构造 512 再删除又会造成不允许的 NPU 峰值。

## 2. 最终结构

本轮实现用同一份 immutable resident plan 驱动三个职责：

```text
512-domain router / Top-10
          |
          v
ResidentPlan[layer]
  - resident_global_ids[32]
  - global_to_slot[512]
  - slot_to_global[32]
          |
          +--> Physical-Q32 owner / selective loader / BF16 NZ post-load
          |
          +--> Hybrid runner
                 +--> resident occurrence -> NPU 0..31 physical slots
                 +--> nonresident occurrence -> C128/MXFP4 logical IDs
                 +--> same-layer exactly-once merge
```

vLLM 继续唯一持有 request identity、KV、QSA/GDN/PLE state、batch、cancel 和 token 提交。KT bridge 持有 expert compute、临时 hidden/route lease、CPU completion 和输入输出 buffer。

## 3. V1：真实构造前 Q32

真实镜像中的单层结果：

| 项目 | 结果 |
|---|---:|
| 逻辑专家 | 512 |
| NPU 物理槽 | 32 |
| `w13` raw shape | `[32, 1280, 2560]` |
| `w2` raw shape | `[32, 2560, 640]` |
| payload | 314,572,800 bytes |
| allocator peak delta | 314,576,384 bytes |

构造在会分配权重的父路径之前绑定物理视图，没有创建 512-expert routed payload。

证据：[V1 receipt](../../artifacts/round16-b/evidence/20260929-round16-b-q32-v1-device-v003/receipt.json)。

## 4. V2：选择性加载与布局

checkpoint 保持原始 512 逻辑专家。loader 在 NPU 传输前按 global expert ID 选择 32 行：

- gate/up packed source shape：`[512, 1280, 2560]`；
- down source shape：`[512, 2560, 640]`；
- 32×3 个 projection 全覆盖；
- gate、up、down 的 Q32 目标值 exact；
- 只对 Q32 storage 执行 Ascend post-load；
- 最终 tensor 不引用一个隐藏的 512-row NPU storage；
- allocator peak 约为两份 Q32 payload，而不是一层全 512 payload。

证据：[V2 receipt](../../artifacts/round16-b/evidence/20260929-round16-b-q32-v2-device-v007/receipt.json)。

## 5. V3：真实 Hybrid MoE

V3 使用真实 C128/MXFP4 bank 和 BF16/NZ29 resident provider，覆盖：

- all-NPU Top-10；
- all-CPU Top-10；
- resident/nonresident 交错的 mixed Top-10；
- 一次 selector、无 re-Top-K；
- 分支内不重新归一化；
- route weight 只应用一次；
- CPU/NPU occurrence 互斥且并集完整；
- routed scaling/shared/residual 由上游唯一 owner 处理。

数值结果：

| Case | merged relative L2 | repeat hash |
|---|---:|---|
| all-NPU | 0.00018987 | exact |
| all-CPU | 0.00264161 | exact |
| mixed | 0.00172825 | exact |

另外完成 20 轮 lifecycle、取消 drain、晚到结果丢弃和 allocator 稳态检查。

证据：[V3 receipt](../../artifacts/round16-b/evidence/20260929-round16-b-q32-v3-device-v007/receipt.json)。

## 6. V4：48 层同时驻留

V4 不是“逐层构造后立即删除”。它同时保留：

- 48 个 distinct owner；
- 96 个 distinct storage；
- 每层约 300 MiB routed payload；
- 总 nominal payload 14.0625 GiB；
- allocator peak 15,099,666,432 bytes。

关闭后 allocator 回到组件基线附近。这证明全模型 routed resident capacity 可以在单卡合同内成立。

证据：[V4 receipt](../../artifacts/round16-b/evidence/20260929-round16-b-q32-v4-device-v001/receipt.json)。

## 7. V5：完整模型

最终 B 模型：

- TP1，max-running=1；
- max-model-len=256；
- enforce-eager=true；
- graph mode NONE；
- 48 个 MXFP4 bank；
- 48 层 Q32 resident owner；
- Host PLE bridge；
- 原生 vLLM-Ascend QSA/GDN/状态管理。

完整 smoke 覆盖五类 exact canary、32-token Decode、取消和同进程 reuse。所有选中 logprob 有限，取消后没有 unfinished request。

证据：[V5 smoke](../../artifacts/round16-b/evidence/20260929-round16-b-v5-smoke-v023/result.json)。

## 8. 图与原生融合边界

B 首版优先证明 eager 语义，未将 CPU submit/wait 包装为可捕获 NPU 算子。最终配置明确为 `CUDAGraphMode.NONE`：

- 桥外原生 vLLM-Ascend 主干仍在；
- CPU/NPU Expert bridge 是真实执行；
- graph replay 覆盖率为 0；
- 不继承历史 Native FULL_DECODE_ONLY 性能标签；
- 不能把 B 的结果描述为保留全部原生融合的比较。

## 9. 去留

B 已证明架构可行，但 W2 Decode 和 W4 保护 workload 均比 S0 慢。最终 disposition 是 `B16_MEASURED_NO_PROMOTION`，保留代码和证据，不更换默认承载。
