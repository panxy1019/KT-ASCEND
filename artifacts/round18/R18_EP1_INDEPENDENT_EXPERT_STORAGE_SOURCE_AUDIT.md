# Round18 Phase II：EP1 独立 Expert Storage / List 源码审计

日期：2026-09-30
状态：`SOURCE_AUDIT_COMPLETE_DEVICE_NOT_RUN`
源码审计时结论：`PROCEED_TO_ONE_BOUNDED_EP1_LIST_CONSUMER_PROBE`

终态说明：后续已经实现并离线验证 parentless independent-list 探针，但 fresh
route economics 和 current-token device planner gate 阻止了本轮设备执行。该探针
保留为未来入口，状态为 `OFFLINE_READY_DEVICE_NOT_RUN`；详见
`R18_EP1_INDEPENDENT_LIST_PROBE_CONTRACT.json` 和 `FINAL_STATUS.json`。

本审计只读取冻结的 vLLM-Ascend / vLLM 源码与 Round18 已归档证据，没有启动设备、容器或模型，也没有修改 PIECEWISE、CURRENT 或 FINAL 状态。

## 1. 直接结论

冻结栈中存在一个比 Q32 NZ29 parent 原位更新更合适的相邻实现边界：BF16 routed expert 的运行时已经能够把 `w13_weight_list` / `w2_weight_list` 作为逐 expert 独立 NPU tensor 列表传给原生 `AscendMoERunner`，通用未量化 MLP 最终把该列表直接交给 `torch_npu.npu_grouped_matmul`。因此，“保留原生 runner、每个 physical slot 使用独立 storage、只替换 slot 内容”在源码层面是一个有依据的最小探针，而不是要求重新构造一套 Hybrid runner。

但现有 Dynamic EPLB 不能直接用于 TP1/EP1 Q32 pool，原因有四个：

1. `init_eplb_config()` 在 `ep_size == 1` 且启用 Dynamic EPLB 时直接断言拒绝；model runner 也只在 `dynamic_eplb` 为真时创建 loader/updator。
2. 现有 loader 只实现 NPU rank 间 D2D `isend/irecv`，没有 Host backing、pinned buffer 或 H2D 路径。
3. 它发布到设备的是 `log2phy`，而 Round18 Q32 partial 已验证的稀疏语义依赖 512-entry `expert_map` 中的 `-1`。`log2phy` 的生成逻辑假定所有 logical experts 在 EP ranks 的并集中都有物理副本，不能等价表示“当前 pool 未驻留”。
4. 当前 update 顺序是 CPU shadow expert map → device `log2phy` → weight copy。源码没有两阶段 weight-ready 后再原子发布 Q32 `expert_map` 的合同。

因此最小路线是：复用原生 list consumer 和 FULL replay 外部 hook 的时序，新增一个 model-scoped EP1 pool adaptor；不要开启或伪装现有 Dynamic EPLB，也不要使用其 `log2phy` 代替 Q32 `expert_map`。

## 2. EP1 拒绝点

冻结文件 `vllm_ascend/eplb/core/eplb_utils.py`：

- 第 73–84 行读取 `ep_size`，在 `ep_size == 1` 时要求 `dynamic_eplb == false`，随后返回空 placement/map/log2phy。
- 第 98–116 行只在 EP placement 模型中建立每 rank `expert_map` 和 device `log2phy`。

冻结 `model_runner_v1.py`：

- 第 611–627 行只有 `dynamic_eplb` 为真时才创建 `D2DExpertWeightLoader`、EPLB subprocess 和 `EplbUpdator`。
- 因而只绕开 `init_eplb_config()` 的断言仍不够；还会把 EP-only planner、process、communication group 和 load collection 一并带入 TP1/EP1。

这个拒绝是“现有 EPLB 控制面不支持 EP1”，不是原生 MoE kernel 拒绝 EP1，也不是独立 tensor list 的 consumer 拒绝 EP1。

## 3. 独立 expert tensor/list 的真实消费者

### 3.1 BF16 list 是冻结栈已有的一等运行时表示

`vllm_ascend/ops/fused_moe/fused_moe.py`：

- 第 132–161 行先把 BF16 parent 转置并按配置转为消费格式；当 fused MC2 且 MegaMoE 或 Dynamic EPLB 启用时，第 153–157 行将 expert 轴 `unbind` 后逐项 `clone()` 成 `w13_weight_list` / `w2_weight_list`，再删除 parent。
- 第 256–285 行无论 fused MC2 还是通用通信路径都会优先读取 list；通用路径也会把 list 继续传入 `build_fused_experts_input()`。
- 第 287–315 行把这些 weight objects 交给未修改的 native MoE communication/MLP 路径。

`vllm_ascend/ops/fused_moe/moe_mlp.py`：

- 第 671–688 行的 `unquant_apply_mlp()` 明确接受 `list[Tensor] | Tensor`。
- 第 693–701 行把 w1 经 `_as_grouped_matmul_weights()` 作为 list 交给 `npu_grouped_matmul`。
- 第 774–782 行对 w2 使用相同的 list consumer。
- 第 796–842 行的统一 MLP 入口在 BF16 非量化路径直接进入上述函数。

因此原生 `AscendMoERunner` 不需要被替换。需要变化的是 routed weight owner / post-load 表示及 pool publication，而不是 runner 的 router/shared/finalize 组织。

### 3.2 当前 list 构造仍不满足“从不创建 Q32 parent”

现有 list 是从已构造、已加载、已做 post-load 的 parent `unbind().clone()` 得到。它证明 consumer 和逐 expert mutation 模式存在，却不能直接证明新 pool 已避免 parent 分配。

要满足 Round18 Phase II 容量合同，需要一个轻量的 RoutedExperts/quant-method owner：

- 构造时直接创建 Q 个独立 w13/w2 slot tensor；
- selective loader 或 backing loader 直接写这些 slot；
- 覆盖/替代现有 `process_weights_after_loading()` 中依赖 parent 的部分；
- 不创建 `[Q, ...]` NZ29 parent 后再拆 list，更不能创建 512-row parent。

这仍是 weight owner/quant-method seam，不是 runner 重写。

## 4. ND / NZ / list 支持边界

源码能够证明：

- list 接口被通用 BF16 grouped-matmul 路径消费；
- EPLB adaptor 会为每个 expert tensor 创建 `empty_like` buffer，并用 `expert_tensor.copy_(buffer_tensor)` 原位替换；
- fused MC2 非 MegaMoE 路径在拆 list 前会把 parent 转为 NZ；MegaMoE 分支可从连续 tensor 拆 list。

源码不能证明：

- 当前 TP1/EP1 通用路径在本机 CANN 上同时接受独立 ND list 与独立 NZ list；
- 从 Host 直接构造的 blob 能携带与 native post-load 相同的 NZ metadata；
- FULL capture 是否会读取 replay 前更新后的 list element 内容。

Round18 `native-expert-pool-fixed-slot-v018` 已证明当前 Q32 parent 是 NZ29，且现有 guard 拒绝对它做 row-level mutation。这个 FAIL 只否定“在 Q32 parent 的子 view 上直接 copy”，不否定独立 per-expert list storage。

所以设备探针必须把 ND-list 与 native-consumable list format 分开记录。若 ND list 被通用 grouped-matmul 接受，首版可以避免 NZ parent。若只接受 NZ list，则需要直接创建/加载单 expert 的 native-consumable storage；若这一步仍要求完整 parent format-cast，路线进入深重构边界。

## 5. `expert_map` 与 `log2phy` 不能混用

`fused_moe.py` 第 193–214 行先在完整 logical domain 产生 Top-K；第 287–315 行同时传入 `expert_map` 和 `log2phy`。

通用 communication path 在 `moe_comm_method.py` 第 151–160 行看到 `log2phy` 时，会先执行 `routed_topk_ids = log2phy[topk_ids]`。`generate_log2phy_map()`（`eplb_utils.py` 第 119–145 行）只收集各 EP rank 中存在的 logical experts，并创建长度为 `len(log2phy_map)` 的 dense tensor。该算法要求所有 logical IDs 在 EP ranks 联合 placement 中存在；它没有 `-1 = pool miss` 的语义。

Q32 pool 的正确 runtime map 应继续使用已通过 `native-partial-v004` 的 512-entry `expert_map`：

```text
resident logical ID  -> 0..Q-1 physical slot
nonresident logical  -> -1
```

现有 EPLB adaptor 的 `do_update_expert_map()` 只更新 CPU shadow，`do_update_log2phy_map()` 才原位更新 device tensor。它没有原位发布 layer `_expert_map` 的函数。因此不能直接复用现有 EPLB publication 并声称 Q32 map 已更新。

## 6. D2D / Host backing 与 hook 顺序

### 6.1 现有 loader 只有 D2D

`D2DExpertWeightLoader.generate_expert_d2d_transfer_task()` 从当前 NPU expert tensor 建立 `dist.isend`，向 `empty_like` NPU buffer 建立 `dist.irecv`；`asyn_expert_weight_transfer()` 调用 `batch_isend_irecv()`。源码中没有 Host canonical backing、pin/register、H2D 或 file/blob reader。

Phase II Host backing 因而需要新的 bounded transfer source，但目标仍可复用逐 expert `copy_` / fixed-address slot 思路。

### 6.2 FULL replay 外部 hook 的顺序

真实 execute 顺序在 `model_runner_v1.py` 中是：

```text
forward_before()              # 2223-2224；创建并启动 D2D task
  -> _model_forward()         # 2269-2271；可能进入 FULL replay
  -> sampling/postprocess
forward_end()                 # 2514-2515；wait、publish、copy
```

下一次 model call 再执行新的 `forward_before()`。dummy/capture 路径同样在 `_model_forward()` 外围调用 before/end（第 3600–3601、3864–3871、3894–3895 行）。

`acl_graph.py` 第 204–207 行仅在 capture 时调用 Python runnable；第 274–289 行 replay 只调用 `entry.aclgraph.replay()`。FULL replay 前第 285–288 行会同步 current stream。

这说明固定地址的 weight/map mutation 可以放在 model graph 外、影响下一次 replay；但当前 EPLB 是“本轮结束更新，下一轮使用”的控制面，不提供当前 token route miss 的同轮填充。Host backing 若使用独立 copy stream，还必须显式用 event 让 replay current stream 等待，不能只依赖 current-stream synchronize。

## 7. 现有 publication 的原子性缺口

`eplb_device_transfer_loader.py` 第 102–121 行执行：

1. 等待 D2D requests；
2. 更新 CPU shadow expert map；
3. 更新 device log2phy；
4. 最后逐 expert copy weights。

即使相同 NPU stream 的排队顺序最终可能保证下一轮看到完整内容，源码也没有“所有 gate/up/down slot ready → event complete → device Q32 expert_map publish”的两阶段合同。Round18 pool 应反转可见性顺序：先写完全部 projection 并确认 ready，再对同地址 512-entry `_expert_map` 做一次 publication；失败时 map 保持旧 generation。

## 8. 最小可行设备探针

只做一个 single-layer、TP1/EP1、BF16、runner 不变的 probe：

1. 构造两个或少量 independent physical expert slots，直接形成 `w13_weight_list` / `w2_weight_list`；运行时不得存在 Q32 或 512-row routed parent。
2. 先用 eager 原生 `AscendMoERunner` 验证 list consumer：all-resident、all-miss、mixed，输出对独立 BF16 reference。
3. 分别记录 ND list 和 native post-load-compatible list 的 NPU format。只晋级实际被 grouped-matmul 接受的格式。
4. 固定 slot 地址，执行 A → B → A：对目标 slot 完整写 w13/w2，等待 copy ready 后，原位更新同一个 512-entry `_expert_map` tensor；不得使用 `log2phy` 冒充。
5. eager 通过后才捕获一个 native MoE FULL component graph。capture output 不作为首个真实结果；合法 replay A 后，在 graph 外完成 B copy + map publication，再 replay，并与 eager B、stale-A、unchanged-slot controls 比较。
6. 每次记录 list element `data_ptr`、map `data_ptr`、format、copy/publish event、generation、output hash/L2、allocator、capture/replay count；注入 w13 写完但 w2 未完成的故障，确认旧 map/generation 仍可见。

该 probe 若通过，只证明 fixed-slot mutation + native list consumer + graph visibility。它仍不解决生产的 current-token route planner。

## 9. 深重构边界

出现任一情况即停止轻薄 adapter 路线并记录明确边界：

- 通用 TP1 BF16 grouped-matmul 拒绝 independent list，或只接受由完整 parent post-load 派生的私有格式；
- 创建单 expert native-consumable storage 必须执行完整 Q-parent/512-parent format rebuild；
- FULL capture 固化 list element storage 内容或 map，replay 不观察同地址更新；
- 必须替换 `AscendMoERunner` 才能传 list；
- current-token miss 必须 D2H route → Python planner → H2D 后恢复，导致 FULL execution 退回细粒度 eager。

前三项属于 native kernel/storage/graph runtime 深边界；最后一项属于 planner/graph split 深边界。只有 bounded probe 的真实 FAIL 才能把相应项标为 blocked。

## 10. 审计结论

源码支持继续做一次有界的 `EP1_INDEPENDENT_LIST_CONSUMER_AND_GRAPH_MUTATION_PROBE`。它比继续攻击 NZ29 Q32 parent 更符合 frozen stack 已有机制，且不需要替换 `AscendMoERunner`。

当前不能写：

- EP1 Dynamic EPLB 已支持；
- Host backing 已支持；
- ND 或独立 NZ list 已在设备通过；
- `log2phy` 等价于 Q32 `expert_map`；
- FULL graph 已观察到新 slot 内容；
- production expert pool 已解决 current-token planning。

这些字段在设备证据出现前均保持 `NOT_PROVED`。
