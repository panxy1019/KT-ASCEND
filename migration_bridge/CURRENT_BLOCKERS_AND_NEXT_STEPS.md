# KTransformers × Kunpeng × Ascend 当前问题与下一步

> 文档状态：Round 4A 最终冻结后的风险清单
> 更新时间：2026-09-12
> 当前结论：P1/P2/P3 在冻结 TP1 static-placement scope 内无剩余 blocker

## 1. 当前 blocker 判断

```text
P1 correctness blocker = NONE
P2 correctness blocker = NONE
P3 correctness blocker = NONE
Round 4A final blocker = NONE
```

`DEEPSEEK_V2_LITE_TP1_MULTI_PLACEMENT = A3_VERIFIED_READY`。

这里的 `NONE` 仅针对以下冻结边界：DeepSeek-V2-Lite revision `604d5664`、
Kunpeng 920、单张 NPU0、TP=1、batch=1、static placement、LLAMAFILE /
CPUInfer、Ascend BF16 unquantized、Graph/Deferred/Dynamic/MTP/Speculative OFF。

## 2. 已关闭问题

### 2.1 Cross-backend numerical equivalence

原始 45/45 raw token exact gate 曾经抓出 BF16-visible boundary 和 Hybrid
partial accumulation 的真实缺陷。修复后仍存在 near-tie backend reduction flip，
且 GEMM backend replacement 无法单调关闭。Round 4A.4 建立并 held-out 验证了
pairwise-order contract：

- system semantics exact；
- same-path deterministic exact；
- CPU-not-hit exact；
- expert rel-L2 `<= 1e-2`；
- `B_pair=2.1875`，K=32；
- stable top1 100% exact；
- ambiguous Hybrid top1 必须属于 baseline ambiguity set；
- downstream quality 无统计显著退化。

该合同 canonical SHA 为
`223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717`，
P2/P3 未重新拟合。

### 2.2 P2 multi-layer nondeterminism

根因已归因到 raw ACL pointer-based D2H/H2D 绕过 torch/torch_npu stream 与
Tensor lifetime 模型。commit `00c7acfe...` 使用 framework-managed `copy_` 后，
P2 same-process、clean restart、pairwise、coverage、quality、lifecycle 和 regression
全部通过。immutable P2 tag：

`round4a5-p2-a3-verified^{}` =
`d4830525c6806d5d587b3096634614aad7fef181`。

### 2.3 P3 scale validation

P3 在 8 layers × 4 experts 上通过全部门禁：117-request determinism、clean
restart、512-position pairwise、32/32 expert coverage、CPU-not-hit、H2+F free
generation、256-example quality、2,304-token stability 和 retained regression。
没有出现新的 multi-layer interaction 或 transfer-lifetime failure。

## 3. 当前未资格验证的风险面

| 风险面 | 当前状态 | 为什么不能沿用 Round 4A 结论 |
|---|---|---|
| upstream Ascend alignment | NOT STARTED | 新 runtime architecture 会改变 attribution lineage |
| TP2/HCCL/multi-NPU | NOT QUALIFIED | 新通信、分片与设备 ordering 风险 |
| Graph/TorchAir | NOT QUALIFIED | 新 capture/lifetime/compile 语义 |
| Deferred/Dynamic placement | NOT QUALIFIED | 新调度、迁移和 buffer lifetime 风险 |
| MTP/Speculative | NOT QUALIFIED | 新 token-history 和 verification 语义 |
| W8A8/MXFP4 | NOT QUALIFIED | 新量化数值合同 |
| KML production backend | NOT QUALIFIED | 新 GEMM backend、构建和性能依赖 |
| NUMA/concurrency tuning | NOT QUALIFIED | 新线程、共享状态和资源竞争风险 |
| multi-node serving | NOT QUALIFIED | 新网络、进程和 failure-domain 风险 |

这些项目不是已知 correctness failure，但都需要独立 evidence chain，不能用
Round 4A tag 自动宣称 READY。

## 4. 下一步：Round 5 Upstream Ascend Alignment

建议严格执行：

1. 从 `round4a-tp1-multiplacement-a3-verified` 创建独立分支；
2. 冻结最新 official KTransformers/SGLang Ascend SHA；
3. 对比 official callback-worker/synchronous-submit 与 local framework-managed
   Tensor transfer 的 stream、lifetime、ownership 和 failure semantics；
4. 形成 adopt/rebase/cherry-pick-concepts/manual-port/retain-local ADR；
5. 只做最小 source change，保留 stable/ambiguous、transfer determinism 和
   CPU-not-hit regressions；
6. 用新 lineage 重新跑 P1 compact，再跑 P2/P3 compact；
7. 若 architecture materially changes，再跑完整 pairwise、quality 和 stability。

## 5. Round 5 hard stop 建议

出现以下任一情况，应停止 alignment 并先归因：

- production transfer 再次使用绕过 framework dependency 的 raw pointer copy；
- same-path 或 clean-restart token nondeterminism；
- CPU-not-hit 污染；
- stable-region token flip、ambiguity escape 或 `B_pair` overflow；
- selected-layer/expert coverage collapse；
- downstream quality regression；
- RSS/HBM per-request monotonic unbounded growth；
- child/parent/model/tokenizer/GGUF/placement provenance 无法对齐。

禁止通过扩大 `B_pair`、更换 corpus、减少 CPU layers/experts、选择低频 expert、
降低 token budget 或改用 sequential-control 来掩盖 alignment regression。

## 6. 不可变基线

以下引用必须保持不可移动：

- `round4a5-p2-a3-verified`；
- `round4a-tp1-multiplacement-a3-verified`；
- P3 placement SHA
  `77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d`；
- P3 GGUF SHA
  `992d32e0de59e135464f77fc4869e810f3d71b3ec01f6ed0996cbeaa58e49df4`；
- pairwise contract SHA
  `223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717`；
- quality manifest SHA
  `c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce`。

历史问题细节保存在 `round4a1/`、`round4a2/`、`round4a3/`、`round4a4/`
和 `round4a5/` 专项文档中；本文件只表达当前 blocker 状态。
