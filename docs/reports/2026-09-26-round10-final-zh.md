# KT-ASCEND Round10 最终报告：Hybrid 资格、Kunpeng、Dynamic 与 Qwen 状态

日期：2026-09-26  
分支：`codex/round10-hybrid-kunpeng-dynamic`  
基线 HEAD：`b41e901c059222b8c1e4367abe193d7f58cc1d83`  
最终状态：`ROUND10_AB_NARROW_QUALIFIED`

## 1. 最终固定配置

本轮最终配置没有晋级新的性能候选。HFINAL 等于 H0：阈值 129 按 EXTEND 请求总 token 判断，实际输入小于 129 走 Hybrid；Decode 走 Hybrid。CPU Expert 保持 Kunpeng C128（4 个 NUMA pool × 32 worker）与只读 MXFP4 bank，NPU 保持 Static Q32、BF16/NZ29，scheduler chunk32，PLE ASYNC，Dynamic OFF。R2、R3、transpose-view、shadow320 均关闭，原 Layerwise/NZ update 不变。

资格范围是 TP1、串行单请求、模板处理后的输入不超过 128 token。长上下文、多请求并发、HiCache/KV offload、FP8 KV、Graph、MTP、TP2、部署与广泛产品质量不在本轮结论内；129/130 的新 Layerwise 实际边界也未取得完整请求资格。

## 2. T0–T2：身份、Hybrid 质量与成本

T0 从 v90 原始快照恢复了模型 revision、tokenizer/template、C128/MXFP4 bank、Q32 map、torch/torch_npu/CANN、PLE 与选路条件。冻结快照逐文件清单通过，未把 v71 或合成 map 当作当前身份。

T1 使用两个模型进程完成 23 次 generation 与 2 次 abort。五类确定性 canary、63/64/65/127/128 输入、32-step 与 49-step Decode、两次取消、cleanup 和同进程 reuse 通过。当前 Hybrid 的“阈值 129”是有限资格，不是全长度最佳阈值。

T2 只使用无同步的 host observer。32-token 样本的 enclosing apply 中位数约 3.430 ms，CPU result wait 约 1.311 ms；CPU submit 与 resident NPU apply 的 host-return 中位数约 0.302/0.394 ms。这些是嵌套时钟，未相加成伪造的设备时间。真实 hidden/route fixture 已冻结供重放。

## 3. T3：Kunpeng Expert 核心

T3 用完 6 个有界组件进程。v96–v99 保留了 package/import/pinned-provider/NUMA 能力等启动合同失败；v100/v101 到达当前真实路径：`KTMoEWrapper -> CPUInfer -> llamafile MOE`，4×32 worker，四个 160-wide intermediate 分区。

两个真实 layer0 group 与独立 MXFP4 decode reference 的 relative L2 分别约 `3.27e-7` 与 `2.00e-7`，重复 hash 完全一致。MXFP4 与捕获的 BF16 contribution 相差约 0.129/0.128，因此只记录跨精度边界，不声称 BF16 等价。T2 没有给出足够依据选择单变量 CPU 性能优化，现有 backend 保留。

## 4. T4：Dynamic 当前事务与净收益

v102 在真实 Hybrid 生命周期中完成 5 次 generation 与 1 次 abort。四次 prefill-end commit 都保留 live KV、GDN、QSA 与 PLE 状态；取消请求没有发布 placement，lease 归还，随后同进程请求得到 `391`。

事务正确，但没有净收益。H0 v95 的 64-token 未 profile 中位数为 5.188200 秒；D1 v102 为 23.648729 秒，即 4.558176× H0。第一次迁移 48 层、1431 slots，耗时 68.381127 秒；重复输入仍复制 118/86 slots，耗时 24.130848/23.166609 秒。Dynamic 保持默认 OFF，没有重新打开 NZ 更新优化。

## 5. T5：GDN、QSA、PLE 与状态池

GDN v103 使用真实权重、层 0/24/46、FP32/BF16、1/3/17 token 与独立 FP64 方程。最大误差 FP32 `1.087496e-6`、BF16 `0.005065961`，重复 hash、Decode/reset state 与 allocator 回收通过。

QSA v103 使用真实权重、层 3/23/47、长度 1/3/4/9/17。最大误差 FP32 `1.079355e-6`、BF16 `0.005116598`；compressed-key、index、signed multi-head ranking、Top-K stress、重复性和 allocator 回收通过。

PLE v105 在物理 12 上通过原始行、重复行、跨分片、consumer stream 顺序、slot reuse、request switch、取消晚完成和 close 后零 credit 八项检查；三个 case 均逐字节一致，额外峰值 493,056 bytes。

v103、v104、v105 的失败状态全部保留：v103 是 gate 将物理 15 写死；v104 是恢复控制器把 `{rev}` 当字面量；v105 的 PLE 已 PASS，随后旧聚合器误用 `canonical_output_relative_l2`，而真实 GDN 字段是 `relative_l2`。这些问题均测试先行修复。最终以固定 SHA 离线组合 v103 GDN/QSA 与 v105 PLE，未重跑成功设备用例。

v102 的四次真实 commit 每次都保留 124 个 live tensor 记录：72 GDN、49 QSA、1 PLE 和 2 个请求元数据。T2 没有识别 B 模块热点，所以不强写 kernel，现有 provider 保持。

## 6. T6：最终组合回归

v107 fresh admission 的 12 个窗口全部通过，最低 conservative SMT-min 平均 idle 99.6236%；NUMA、系统内存、训练完整容器 ID/cgroup 和物理 12 未知进程门禁通过。

唯一模型进程完成 8/8 generation、1/1 abort、0 retry：quote/transform canary、64-token ACK/EOS、32-step Decode、128-token 中途取消、lease 回收和同进程 `391` reuse 全部通过。可比较的输出 token IDs 与 v95 对应前八项完全一致。

v107 的两个未 profile H64 测量为 5.409330 与 5.832713 秒，中位 5.621022 秒，相对 v95 H0 的 5.188200 秒为 1.083424×。两者启用配置相同，因此这里只记录重复波动，不作提速或候选回归晋级结论。

v107 supervisor 为 PASS，driver exit 0，guard exit ACK 与容器删除通过。外层 host controller 随后读取旧文件名 `round10-t1-supervisor.json`，自身留下 FAIL；实际 T6 receipt 名为 `round10-t6-supervisor.json`。原始 FAIL 没有改写，生成器已补回归测试修复，并且没有用第二次模型启动掩盖问题。

资源峰值：board HBM 28,778 MiB、owned HBM 25,762 MiB、host cgroup 338.977 GiB。最终逐文件审计 179 个 model/bank 文件无活动 mmap/fd 后，只执行 `POSIX_FADV_DONTNEED`；文件身份不变，没有全局 `drop_caches`，未修改外部任务。物理 12 最终无设备进程，自动任务保持暂停。

## 7. 预算与结论

Round10 总计使用 4 个模型启动、36 次 generation、4 次 abort、0 自动重试。T3 使用 6 个组件进程，T5 使用 3 个组件进程。未 push、merge 或 tag。

最终结论是 `ROUND10_AB_NARROW_QUALIFIED`。它表示固定的 Hybrid129 静态配置、当前 Kunpeng backend、Dynamic 默认关闭和现有 GDN/QSA/PLE provider 在上述窄范围内可复现并通过质量、取消、复用与资源回收。它不表示获得新的性能收益；`PERFORMANCE_OBJECTIVE` 明确为 `NO_NEW_PROMOTED_PERFORMANCE_GAIN`。

核心机器可读入口：

- `artifacts/round10/ROUND10_FINAL_STATUS.json`
- `artifacts/round10/T6_FINAL_REGRESSION.json`
- `artifacts/round10/T5_MODEL_STATE_RESULT.json`
- `artifacts/round10/T4_DYNAMIC_RESULT.json`
- `artifacts/round10/BUDGET_LEDGER.json`
- `artifacts/round10/RESOURCE_RETIREMENT_RECEIPT.json`
- `artifacts/round10/EVIDENCE_SHA256.json`
