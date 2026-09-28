# Round15 最终报告：成熟 NPU 算子与 KT 分段图执行

## 1. 最终结论

Round15 在用户批准的简化请求评测口径下完成，最终状态为 `ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION`。

性能目标没有达成。A 的短上下文 QSA 快路径在组件中很快，但没有转化为 Decode-heavy W2 请求收益；B 的 resident-MoE Decode 分段图真实命中、数值正确，并带来小幅请求改善，但改善明显低于原晋级门槛，同时 W2 TTFT 退化并增加约 984 MiB HBM。因此最终仍使用 Round14a 的 S0 配置，两个新候选都保持默认关闭。

请求实验按用户要求从四进程 ABBA 缩减为每条线各一个 fresh 基线进程和一个 fresh 候选进程，每个 W1/W2 各两次 measurement。以下数字只说明方向和大致幅度，不能解释为四进程稳定性、统计资格或生产晋级。

## 2. 固定身份

本轮父配置为 Round14a S0：Qwen3.8 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`，Kunpeng C128（NUMA 2/3/4/7），只读 MXFP4 CPU bank，Uniform Q32 BF16/NZ29 resident，chunk32，PLE ASYNC，QSA Vectorized Prefill＋DIRECT_BF16_ROW Decode，Round14a promoted GDN，H129，TP1、并发1。Dynamic、deferred、R2/R3、transpose-view、shadow320 和 Graph 均关闭。

A3 的 S0 镜像为 `kt-round64-migration:20260916`，实际环境是 aarch64 Ubuntu 22.04、Python 3.11.15、torch 2.9.0、torch_npu 2.9.0.post2 和 CANN 9.0.0。新 vLLM-Ascend TP8 报告只用于寻找成熟实现入口，没有被当作本轮单卡收益证据，也没有重跑 TP8。

## 3. P0：真实热点

两个 fresh S0 profile 块均通过身份、输出、路径和 provider 对照。两块都显示 QSA 是采样到的最大 NPU 成本：W2 Decode 的 QSA attention 每个采样调用约 6.06 ms，indexer 约 2.8–3.0 ms；GDN 约 1.04–1.11 ms，routed-expert parent 约 1.95–2.02 ms，resident NPU 部分约 0.40–0.44 ms。父子 span 没有相加。

这个证据决定先筛 QSA，再独立研究减少 Host 下发的 resident-MoE 图段。

## 4. A：成熟 QSA 路径

### 4.1 组件

上游 `npu_lightning_indexer` 在 520 groups、Top512 的形状上可表达，但这不属于当前 <=256-token 资格范围。当前合格短上下文中 causal selected set 已等于全部可用 set，因此选取的候选是 `FULL_SET_CAUSAL_SHORT_CONTEXT`：保持 causal、GQA、KV/index 和 state 身份，在完整集合场景避免无效 ranking 工作。

v339/v340 两个独立组件进程均通过 FP32/downstream 数值、真实权重、20 轮复用、allocator、source 和清理合同。完整 parent 的候选/基线耗时比约为 0.170–0.198，组件门成立。

### 4.2 请求方向性结果

v341 基线和 v342 候选各完成 14/14 请求，包含五类 canary、多步 Decode、取消和同进程复用。14 组输出 token 完全相同，logprob 有限；候选在全部 12 个 QSA 层真实命中。

| Workload | 指标 | 基线 | 候选 | 候选变化 |
|---|---:|---:|---:|---:|
| W1 128/16 | TTFT | 2.8650 s | 2.7607 s | 改善 3.64% |
| W1 128/16 | E2E | 7.0964 s | 6.8340 s | 改善 3.70% |
| W1 128/16 | 串行吞吐 | 2.2547 tok/s | 2.3412 tok/s | 改善 3.84% |
| W2 128/128 | TTFT | 2.7720 s | 2.8621 s | 退化 3.25% |
| W2 128/128 | TPOT | 0.26651 s | 0.26762 s | 退化 0.41% |
| W2 128/128 | E2E | 36.6190 s | 36.8494 s | 退化 0.63% |
| W2 128/128 | 串行吞吐 | 3.4954 tok/s | 3.4736 tok/s | 退化 0.63% |

W2 没有请求收益，A 结论为 `ROUND15_A_DIRECTIONAL_NO_REQUEST_GAIN_KEEP_DEFAULT_OFF`。该结果也说明一个明显的组件局部优化可能被完整请求的其他成本与波动完全吸收。

## 5. B：KT resident-MoE Decode 分段图

### 5.1 安全边界

捕获范围只包含 `UnquantizedFusedMoEMethod.apply` 内 Decode batch1 的 resident NPU MoE 子调用。CPU Expert 提交、执行、等待和合并，PLE Host gather/H2D，QSA、GDN、KV/请求状态、scheduler 和取消均保留 eager。Prefill 在执行前回退 eager。实现没有把 CPU Expert 塞进图、没有跨层延迟结果、没有删除同步，也没有启用 SGLang 整模型 Graph。

### 5.2 失败与修复

v356 完成了请求，但图命中为零。根因是探索组件把 `topk_ids` 冻结为 int64，而生产 Decode 实际使用 int32。v356 只证明 eager fallback 正确，其性能不能代表图候选。

v357 按生产 int32 重建组件。20 轮的 relative L2、max abs 都为 0，重复 hash 完全一致，20 个变化输入产生 20 个不同输出。host 中位耗时从 0.483 ms 降至 0.347 ms，device 从 0.364 ms 降至 0.241 ms；reset 后 allocated 增长为 4096 bytes。

v358 在首个 Decode 的 layer0 capture 自检失败，relative L2 为 1.0，但 fail-stop 在消费前触发，没有发布错误输出。定位发现组件总是在 capture 后调用一次 `graph.replay()`，而全模型路径直接读取 capture 阶段的 `graph.output`。新增回归测试后，最小修复只是在首次 capture 后先 replay，再验证和消费结果。

v360 完成 14/14 请求、取消/复用和清理。48 层均 capture，记录 48 个 capture event 和 192 个 replay marker；首轮自检最大 relative L2 为 0；13 组可对齐输出 token 与基线完全相同。

### 5.3 请求方向性结果

| Workload | 指标 | v354 eager | v360 graph | 候选变化 |
|---|---:|---:|---:|---:|
| W1 128/16 | TTFT | 2.9001 s | 2.7849 s | 改善 3.97% |
| W1 128/16 | E2E | 6.9774 s | 6.7507 s | 改善 3.25% |
| W1 128/16 | 串行吞吐 | 2.2931 tok/s | 2.3701 tok/s | 改善 3.36% |
| W2 128/128 | TTFT | 2.7342 s | 2.8789 s | 退化 5.29% |
| W2 128/128 | TPOT | 0.26878 s | 0.26555 s | 改善 1.20% |
| W2 128/128 | E2E | 36.8694 s | 36.6035 s | 改善 0.72% |
| W2 128/128 | 串行吞吐 | 3.4717 tok/s | 3.4969 tok/s | 改善 0.73% |

候选的 owned 和 whole-card HBM 峰值均比基线增加 984 MiB。Host cgroup 峰值因本次 monitor 未记录而保持 `UNKNOWN`，没有用系统 MemAvailable 代替它。

B 的小幅方向性收益远低于原 W2 TPOT/吞吐均改善 10% 的请求门，同时 W2 TTFT 退化超过 5%。因此结论为 `ROUND15_B_DIRECTIONAL_SMALL_GAIN_BELOW_PROMOTION_GATE`，默认保持关闭。

## 6. 最终配置、资源和预算

没有候选晋级，最终配置不贴新的性能标签，保持 S0：Uniform Q32、C128、MXFP4、BF16/NZ29、chunk32、PLE ASYNC、Vectorized Prefill、DIRECT_BF16_ROW Decode、promoted GDN、H129、TP1、并发1，Graph OFF。

Round15 实际消耗 13 个模型进程、83 次 generation、5 次取消控制、4 个组件进程和 0.8372649977 NPU device-hour，均低于上限。所有自有进程在每块后退出，guard ACK 与容器移除通过；归档和 SHA 完成后才对登记的只读 bank/checkpoint 文件执行定向 DONTNEED，没有全局 drop_caches，也没有停止或修改训练与外部任务。

最终只读检查时，A3 新出现 16 个外部 `qwen38-vllm-*` 容器。Round15 没有把它们视为自有对象，也没有继续提交设备工作。

## 7. 证据边界

- A、B 请求结果均只有一个 fresh 进程/臂和两个 measurement/workload。
- 本轮证明候选能否接通、是否命中、质量和取消/复用是否保持，以及大致性能方向。
- 本轮不证明四进程 ABBA 稳定性、统计显著性、生产可用性或更长上下文资格。
- B 的组件局部收益没有转化为同量级请求收益。下一轮若继续图执行，应先扩大真正占关键路径且能安全捕获的 NPU 区域，而不是继续为当前约 0.4 ms resident 子调用增加图实例。

完整原始指标见 `A_REQUEST_DIRECTIONAL_RESULT.json`、`B_REQUEST_DIRECTIONAL_RESULT.json` 和 `BENCHMARK.csv`；失败证据 v356/v358 原样保留。
