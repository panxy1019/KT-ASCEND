# Round18 Phase II：NZ29 固定槽写入源码审计

日期：2026-09-30
状态：`SOURCE_AUDIT_COMPLETE`
设备范围：本审计没有启动新 NPU、容器或远端任务；设备事实仅引用已归档 v018 单层探针。

## 结论

对当前 Round18/S0 的权重表示——每层两个连续 `Physical-Q32` parent，`w13_weight` 与 `w2_weight` 都是 Ascend format 29（FRACTAL_NZ）——没有找到一个受支持、且能证明只覆写一个 expert slot 而不整理/写回完整 Q32 parent 的接口。

因此，当前 parent 上的 direct fixed-slot 路线应当停止。`destination.index_copy_(0, index, staged)` 在逻辑上能正确更新所选槽，历史设备探针也验证过数值，但固定版本实现会在 NZ metadata 不匹配时先生成 matched/contiguous parent，逐 index 做 select/copy，最后 `format_fresh_view(self, result)` 调用 `self.copy_(result)`。Round9 原始 trace 对每个 projection update 都观察到了完整 32-slot parent 的 `format_contiguous` 和完整 parent→parent 的最终写回。这不符合 Native expert pool 的“不重建完整 parent”门。

冻结 vLLM-Ascend 中确实存在另一条更窄的现成机制：在 fused-MC2 + MegaMoE 或 Dynamic EPLB 表示下，BF16 routed weights 被拆为 per-expert tensor list，EPLB 为每个 expert 建立 `torch.empty_like()` 的同格式 buffer，并用 `expert_tensor.copy_(buffer_tensor)` 原位替换单 expert。该表示从结构上没有 Q32 parent，因此可以成为下一次 cheap component probe 的对象。

它不是当前 S0 parent 的“写入 API”：启用它会改变 routed-weight owner 和 Native consumer 输入表示；现有 Dynamic EPLB 还明确拒绝 EP1。没有真实 TP1/Q32 device consumer 证据前，不能把这条 seam 写成 Round18 expert pool 已可用。

## 当前 Q32 parent 为什么被阻断

真实 v018 单层 owner 观察到：

- `w13_weight`: BF16、shape `[32, 2560, 1280]`、format 29；
- `w2_weight`: BF16、shape `[32, 640, 2560]`、format 29；
- `mutation_started=false`；
- 状态为 `FAIL_FIXED_SLOT_POSTLOAD_FORMAT_INELIGIBLE`。

这次探针在任何写入之前 fail-closed，因而没有污染 parent。它证明当前生产表示确实是 NZ29，也证明现有 Round18 writer 只资格化 ND/BF16，不能直接套在 NZ29 上；它没有证明所有未来 writer 都不可能。

固定 op-plugin 的 `index_copy_` 路径为：

```text
self (NZ29 parent)
  -> check_match(self)
  -> format_contiguous(self) when metadata/layout does not match
  -> for each index: select(destination) + copy_(source)
  -> format_fresh_view(self, result)
  -> self.copy_(result)
```

Round9 的设备 trace 与此源码一致：每个 projection update 的 `format_contiguous` shape 是完整 32-slot parent，最终 `format_fresh_view` 也是完整 parent→parent。它没有证明每个 slot 都重新转换一次完整 parent；它证明每次 projection update 的 parent 级整理与写回不能从成本边界中删除。

`scatter_update_` 和 `npu_scatter_nd_update_` 也不能补这个缺口。冻结 image 的接口文档只声明 ND 输入/更新格式，没有 NZ29 parent 的支持合同。`npu_format_cast(..., 29)` 是 whole-tensor layout conversion，并不提供“预处理一个 expert blob 后按物理 NZ offset DMA 到 parent slot”的公开接口。猜测 NZ 物理 offset、raw pointer 或 `as_strided` 仍不合格。

## 可用但属于另一种表示的 per-expert seam

冻结 vLLM-Ascend 在 unquantized post-load 中只在 `enable_fused_mc2` 且 `use_megamoe or dynamic_eplb` 时把 parent `unbind(dim=0)` 后 clone 成 `w13_weight_list/w2_weight_list`，随后删除 parent。Native fused MoE 也显式接受 tensor list。

EPLB adaptor：

- 从 tensor list 或 parent 的第一个 expert 取得真实 expert shape；
- 为每个 buffer 创建 `torch.empty_like(expert_tensor)`；
- 以 `expert_tensor.copy_(buffer_tensor)` 更新一个 physical expert；
- 通过已有 `log2phy.copy_` 更新映射。

这说明“同格式、per-expert 独立 storage 的 fixed-slot write”在冻结源码里有明确先例。它没有证明以下事项：

- TP1/EP1 能直接启用现有 Dynamic EPLB；冻结源码在 `ep_size == 1` 时断言 Dynamic EPLB 必须关闭；
- 当前 Q32 parent 能无转换地改成 tensor list；
- Native partial Q32 kernel、shared expert/finalize 和 graph 仍保持同一计算身份；
- Host BF16 backing 能直接生成与 consumer 一致的 per-expert NZ blob；
- mapping publish 与 gate/up/down 全部写完之间已有原子提交合同。

## Offloader 不是 expert-pool slot writer

冻结 Ascend prefetch offloader 会检测参数是否为 NZ，并把静态 buffer pool 中与该参数同 key、同完整 shape 的 buffer 整体 cast 为 NZ。这解决的是整 parameter 的 offload/prefetch 格式一致性；源码没有按 logical expert ID 管理固定槽，也没有提供 current-token route miss planner。因此它不能作为 Round18 fixed expert pool 的现成实现。

## 建议的唯一下一次 cheap device probe

如果主线仍要继续验证 backing-format 路线，应把探针对象改为 per-expert tensor-list 表示，而不是再对当前 NZ29 parent 尝试 `index_copy_`、slice `copy_` 或 scatter：

1. 只构造一个 Native unquantized MoE layer 和一个 per-expert physical slot；不启动完整模型、Graph 或 Dynamic EPLB planner。
2. 用冻结 post-load 路径生成 expert A 与 expert B 的同格式 per-expert tensor，记录 format、shape、storage bytes 和 `data_ptr`。
3. 用 `empty_like`/same-format buffer 完成 A→B→A，写入后显式 barrier；每次都保持 destination `data_ptr` 不变。
4. 让 Native consumer 分别计算 A、B、A，并与独立 eager BF16 reference 比较；同时证明没有 parent tensor、whole-parent cast 或 hidden parent alias。
5. 记录 allocator、copy/publish wall、generation、失败中止和 close。通过后才能讨论 per-expert backing blob 和 graph-safe publication。

若冻结 TP1 Native consumer不能接受该 tensor-list表示，或只能通过启用 EP/FUSED_MC2 改变当前系统计算路径，该路线应记录为 `PER_EXPERT_TENSOR_LIST_CONSUMER_NOT_AVAILABLE_FOR_TP1`。这不是“性能无收益”，而是当前栈没有合格的 parent-preserving slot representation。

## 证据边界

机器可读结论：[R18_NZ29_FIXED_SLOT_SOURCE_AUDIT.json](R18_NZ29_FIXED_SLOT_SOURCE_AUDIT.json)。

主要本地证据：

- v018 receipt：`evidence/native-expert-pool-fixed-slot-v018/fixed-slot-receipt.json`；
- 冻结 vLLM-Ascend `fused_moe.py`、EPLB adaptor/loader/utils、Ascend prefetch offloader；
- Round9 `NZ_UPDATE_CAUSAL_FINDINGS.md` 与冻结 `installed-IndexCopyKernelNpu.cpp`、`installed-NpuUtils.cpp`；
- 所有路径及 SHA-256 已写入机器可读审计结果。

公开主线只能作为机制背景。当前结论绑定冻结源码 hash 和 v018 receipt，不把上游新版本的 per-expert NZ tensor-list 支持反推成当前镜像资格。
