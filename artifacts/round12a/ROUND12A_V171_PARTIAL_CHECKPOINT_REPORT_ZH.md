# Round12a 最终报告：tied 语义已验证，资源门未通过，公平 KT/non-KT 比较未启动

## 1. 终态

Round12a 的真实终态是 **`ROUND12A_PARTIAL_WITH_GAPS`**。

OffloaderV1 的 tied-parameter-aware 改法已经得到离线语义证明和一次真实 Ascend 组件的数值证明：v171 连续 7 次调用都与 offload 前的 NPU reference bit-exact，最大 relative L2 为 0；Parameter 与 persistent Buffer 的 tied identity 在每次 `functional_call` 内都保持；CPU 参数字节计数、内容和 residency 也符合合同。

该候选仍不能标为组件 PASS。v171 清理后，进程内 NPU allocator 从基线 0 回到 4,608 bytes，而冻结门槛要求精确回到基线。Round12a 的 2 个组件进程预算已由 v170/v171 用完，且合同明确规定组件失败时关闭模型阶段。因此没有启动首个 non-KT 模型请求，也没有提交 KT/non-KT/non-KT/KT 四进程比较。

所以本轮**不能回答 KT-style heterogeneous execution 在 Ascend + Kunpeng 上“值得”或“不值得”**。当前结果类别是 **non-KT 集成未完成**，不是 non-KT 性能负结果；任何 KT/non-KT 加速比、无限加速或系统价值结论都没有有效证据。

## 2. 冻结起点与唯一代码候选

Round12a 从 Round12 的冻结状态继续：模型 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`、默认 Hybrid 阈值 129、chunk32、PLE ASYNC、Dynamic OFF、TP1、串行单请求。未来两臂均要求使用 Round12 已晋级的 QSA：

- QSA source：`0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35`
- Qwen consumer：`d4583c5abe9791b09c48d1dd8f8491a2988eb5147d45df188e0e48908c699f1f`

OffloaderV1 基线 source 为 `98a37555dd1fb6bda68dc5485ca0999dce75a5c43a6df62f86a16f740532442e`。唯一候选 source 为 `8b68d4012a9bd0fab3bab70ab28db1d7c09b6242af912066d2aedd3f3a3c9006`，改动仅为：

1. 使用 `module.state_dict(keep_vars=True)`；
2. 对同一个 tensor object 只保留第一个登记名字；
3. 每个唯一 object 只复制一次到设备；
4. 保留 PyTorch 默认的 `functional_call(..., tie_weights=True)`。

没有使用 `tie_weights=False`。同冻结镜像中的离线证明显示，这个替代项虽然在小样本中没有改变输出数字，却会破坏 forward 内 Parameter 和 Buffer 的 alias identity。

## 3. v169：安全 guard 的预设备失败

v169 完成 fresh 2×60 秒准入，但旧 Round12 guard 的精确白名单不接受 `kt-round12a-*` 名字。controller 在 `docker start` 前失败，没有启动设备进程，因此不计组件预算。

修复按测试先行完成：白名单只新增以下 Round12a 自有形态，没有接受任意名字：

- `kt-round12a-tied-offloader-vN`
- `kt-round12a-nonkt-smoke-vN`
- `kt-round12a-{kt|nonkt}-{a|b}-vN`

修复后的 guard 被纳入后续不可变 snapshot manifest，不能再从外部任意路径注入。v169 原始 admission 和 controller 失败证据已逐文件与远端 SHA 对齐并在本地设为只读。

## 4. v170：探针构造在候选调用前破坏 tied identity

v170 通过独立准入并实际启动第一个 NPU 组件进程。它在调用 OffloaderV1 前失败：探针先在 CPU 上把同一对象登记为两个名字，再对整个模块执行 `.to("npu:0")`；本地 NPU provider 的 module `_apply` 路径使两个登记项成为不同对象。预调用 identity 断言正确地拒绝了这个 fixture。

v170 没有提供候选数值结论。容器已退出并移除，物理12恢复为空。该 run 计入 1 个组件进程。修复只让 fixture 直接在 `npu:0` 上构造共享的 Parameter 和 persistent Buffer；Offloader 候选 source 和全部数值/资源门槛均未改变。

## 5. v171：tied 语义和数值通过，allocator 精确回收失败

v171 再次通过 fresh 2×60 秒准入。最低 conservative SMT idle 为 `0.9967491184`；NUMA2/3/4/7 最低 MemFree 分别约为 `230.907/230.181/177.025/224.708 GiB`。训练容器完整 ID 与 cgroup 匹配，没有未知物理12进程。

真实 Ascend 组件得到：

| 检查 | v171 结果 |
|---|---:|
| 调用次数 | 7 |
| 输出 hash | 7/7 与 reference exact |
| 最大 relative L2 | 0.0 |
| Parameter tied identity | 7/7 保持 |
| Buffer tied identity | 7/7 保持 |
| CPU 参数字节 | 8,392,704 |
| Offloader 计数字节 | 8,392,704 |
| steady wall median（排除首样本） | 1.44833 ms |
| allocator baseline | 0 bytes |
| cleanup 后 allocated | 4,608 bytes |
| cleanup 后 reserved | 2,097,152 bytes |

`result.json` 因最后一项被原样标为 `FAIL_ROUND12_P4_OFFLOADER_ALLOCATOR_CLEANUP`，controller 也原样 FAIL；没有放宽容差或改写状态。

源码审计显示，探针的 `state = module.state_dict(keep_vars=True)` 在 `finally` 中没有删除，因此仍引用 2,048 个 BF16 元素、即 4,096 bytes 的 NPU `scale` Buffer。它与 4,608 bytes allocator 残留高度吻合，差额可能来自 allocator 粒度。这是有依据的根因推断，但尚未由新设备 run 证明，不能据此把 v171 改成 PASS。

进程级清理是完整的：guard 记录终止，精确 owned container 已移除；最终只读检查没有 Round12a 容器或进程，物理12为空，训练容器继续运行且未被修改。

## 6. 为什么没有启动 non-KT 请求和 ABBA

冻结合同要求组件同时通过 tied 数值、identity、allocator baseline、guard ACK、容器移除和物理设备清理。它还规定一个新组件 revision，失败后模型阶段关闭。实际消耗为：

- v169：预设备 guard 失败，0 个组件预算；
- v170：fixture 构造失败，1 个组件预算；
- v171：数值/identity PASS、allocator baseline FAIL，1 个组件预算；
- 自动原样重试：0。

因此账本为组件 `2/2`，模型启动 `0/8`，generation `0/44`，abort `0/1`。P2 首个 non-KT 请求和 P3 公平 ABBA 都是 `NOT_RUN_COMPONENT_GATE_CLOSED`。

两臂使用同一晋级 QSA 的公平设计已经冻结，但没有变成运行证据。Round12 中已有的 KT 请求或 QSA ABBA 不能与一个未完成的 non-KT 请求拼成新比较。

## 7. “KT-style heterogeneous execution 是否值得”的严格回答

**本轮未能判定。** 当前可以确认的是：

- tied-aware Offloader 的核心调用语义在 Ascend 组件上算对了；
- non-KT full-model integration 仍缺一个严格通过的资源生命周期 gate；
- KT arm 已有历史可运行证据，但本轮没有同 workload、同晋级 QSA、四独立进程的 non-KT 完成样本；
- 因而没有合法的 E2E、TTFT、Host memory 或 HBM 配对比值。

这不是 `NOT_WORTH_IT`，也不是 `WORTH_IT`。唯一合规分类是 **`UNANSWERED_INTEGRATION_INCOMPLETE`**。

若后续获得新预算，最小下一步是让组件在 cleanup 前释放 `keep_vars` state 引用，以新身份重新验证 allocator 精确基线；只有通过后，才运行一个 64-input/4-decode non-KT smoke，再按已冻结队列执行 KT/non-KT/non-KT/KT。无需改变 QSA、模型、精度、阈值、bank、map 或资源合同。

## 8. 资源、缓存和证据

本轮两个实际组件只挂载 source overlay 并生成合成 tensor，没有打开登记的 bank `.gguf` 或 checkpoint `.safetensors`。因此缓存退休 hook 的状态是 `NOT_APPLICABLE_COMPONENT_ONLY_NO_REGISTERED_WEIGHT_FILES`；没有执行全局 drop cache，也没有无依据地对模型文件调用 DONTNEED。

最终状态：物理12为空；Round12a 自有容器 0；Round12a 自有进程 0；匹配活动 timer 0；训练容器完整 ID 保持运行。Round12a 没有 push、merge 或 tag。

主要证据入口：

- `ROUND12A_TIED_SEMANTICS.json`：离线 tied 语义证明
- `ROUND12A_COMPONENT_DISPOSITION.json`：v169/v170/v171 分项结论
- `remote-evidence/v171-component/result.json`：7 次真实 Ascend 数值与 allocator 结果
- `ROUND12A_EXECUTION_BUDGET.json`：最终预算
- `ROUND12A_RESOURCE_RETIREMENT_RECEIPT.json`：缓存 no-op 理由与终态
- `ROUND12A_FINAL_REMOTE_STATE.json`：物理12、容器、进程、timer 和训练身份
- `ROUND12A_EVIDENCE_HASHES.json`：最终证据 hash 清单
