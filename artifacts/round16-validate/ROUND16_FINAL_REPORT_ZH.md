# Round16 最终报告（外部资源阻断终态）

生成时间：2026-09-28T19:23:00Z。

本轮终态为 `ROUND16_PARTIAL_WITH_GAPS_EXTERNAL_RESOURCE`。这不是性能负结果，也不是 A/B 已完成的声明。Round15 屏障、离线合同和源码接线已经完成；A 的 fresh profile、B 的真实设备 MoE、完整层、模型和请求比较均未启动。默认实现继续保留 S0，即 SGLang＋KT 的 Round14a 晋级配置。

## 起点和预算

Round15 已以 `ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION` 终止，cleanup 与资源 lease 释放证据通过。S15 没有新晋级变化，因此本轮父配置仍为 Round14a promoted GDN、QSA Vectorized Prefill／DIRECT_BF16_ROW Decode、PLE ASYNC、Kunpeng C128／MXFP4、Uniform Q32 BF16/NZ29、chunk32、H129、TP1、并发1。

本轮没有创建 NPU 组件、模型或请求进程。14 个模型组、128 次 generation、6 次 abort、16 个设备组件进程和 32 NPU device-hours 的消耗均为 0。准入失败不计入这些设备预算，但受独立失败周期上限约束。

## A 线：当前关键路径仍需设备证据

A1 已有严格的双 fresh-block observer、控制器和分析器。它把 CPU Expert、NPU 计算、Host dispatch、同步依赖和 UNKNOWN 分开，并禁止将嵌套 parent/child span、Host/device 时钟域或旧轮热点占比直接相加。

A2 只有在两个 fresh S0 块都证明 CPU 结果暴露等待是首要可操作阻塞、占直接 Decode wall 至少 10%，且同层存在尚未使用的独立 NPU 窗口时才可进入。A3 只有在两个块都证明 Host launch／metadata 碎片达到相同比例且区域明显超过 Round15 被拒绝的 resident-only graph 时才可进入。

由于没有任何安全设备通过准入，A1 的两个模型块均为 `NOT_RUN`。所以当前约 3.x tok/s 的真实剩余限制仍为 `UNKNOWN_WITH_FRESH_EVIDENCE_PENDING`；不能把历史 GDN、CPU Expert 或 launch 结果写成当前答案。A2/A3 都是“未选择，缺 fresh 证据”，并非“证明没有窗口”。A 没有请求收益数字。

## B 线：已从 scaffold 推进到可审计的真实接线合同

源码审计确认，冻结 vLLM-Ascend donor 的 `MoECommMethod.fused_experts` 可以在同一个 `prepare → select once → fused_experts → finalize` 包络中消费预先冻结的 Top‑K。它返回的是已乘 route weights、按 token 聚合、尚未乘 routed scaling 的 `[T,H]` partial；旧 scaffold 的逐 occurrence raw mapping 与这个真实入口不兼容。

集成实现现在具备以下离线证据：

- `qwen3_next.FusedMoE` 在目标模型构造期间被有界、可回滚地替换；Q32 owner 与自定义 runner 作为一对注入。
- 48 层全部确认消费同一个 immutable placement plan，物理 `w13/w2` 首维为32，逻辑专家数保持512。
- 真实形状的 fake Qwen block 通过 `self.experts.forward` 进入桥；prepared hidden/router 进入唯一一次 upstream selector，无效行在 CPU/NPU payload 前变成安全ID和零权重。
- 混合路由顺序固定为 CPU submit、native weighted aggregate、CPU sync/drain、单次 routed merge。
- routed scaling、shared expert/gate、output transform、reduce/zero-expert 仍由 inherited upstream `MoERunner.forward` 唯一处理；residual 仍由调用 Qwen layer 处理。
- all-CPU、all-NPU、混合路由、comm object 漂移、构造异常、旧 ABI 误用和未 finalize aggregate 都有 fail-closed 测试。

首个可审计设备候选被严格限制为 TP1／EP1、eager、ALLGATHER、BF16／`QuantType.NONE`、physical Q32、`expert_map=None`、`log2phy=None`。MC2、FUSED_MC2、ALLTOALL、Graph、TP/EP 多 rank 和异步请求资格均没有被外推。

集成树通过 220/220 项测试、compileall、JSON 解析、overlay byte identity 和 `git diff --check`。但 `live_eligible=false`：真实 donor image 内的48层接线、physical32 device adapter、一层数值、S15 weighted aggregate parity、真实 stream drain、TP1 eager 模型和请求都还没有运行。因此 B 当前达到的是“源码与真实形状接线合同”，不是 `REAL_MOE_COMPONENT_PASS`，也没有保留图覆盖率或请求吞吐结论。

## 外部资源阻断

本轮进行了6个彼此独立的只读准入周期。第6次在 2026-09-28T19:21:44Z 再次看到16个外部 `qwen38-vllm-0…15` 容器分别显式声明物理0–15号卡。`npu-smi` 显示奇数卡服务已高占用，多个偶数卡容器刚重启并正在装载；低瞬时 HBM 不能覆盖容器的显式所有权声明。Round16 自有远端进程为空。

所有候选 `[12,10,8,6,4,2,0,14]` 因 `EXTERNAL_CONTAINER_CLAIM` 在第一道所有权门被拒绝。因此 CPU/NUMA 两轮采样被安全短路，没有创建容器、初始化 NPU、触碰训练或外部服务，也没有降低任何资源门。规定的6次失败周期已用完，不执行第7次无变化重试。

## 四个最终问题

1. **现在约3.x tok/s的关键限制是什么，A2/A3为何选或不选？** fresh A1 没有获得设备执行机会，所以关键限制仍未知。A2/A3 都因缺两个 fresh 块而未选择，不能用历史占比代替。
2. **A是否减少了真实请求成本？** 未测。没有 A 候选、模型 smoke 或 ABBA，因此没有收益或无收益结论。
3. **B实际接到了哪一级，原生融合／图保留多少？** 已接到 exact-source、构造期 selective Q32、成对 owner/runner 和真实形状 `self.experts.forward` 的 CPU-only合同层；真实 NPU MoE、完整 decoder、模型与请求未跑。图模式未启用，保留比例未知。
4. **同资源下A/B表现如何，是否值得更换默认承载？** 未测，不能比较。默认继续使用 S0 SGLang＋KT；没有依据切换到 vLLM-KT。

## 恢复点

只有外部状态发生变化后才恢复：至少一个明确候选卡不再有外部容器声明，然后按冻结合同完成两轮各60秒的 CPU/SMT、NUMA、HBM、进程与训练 cgroup 准入。通过后顺序不变：先跑 A1 两个 fresh profile 块；B 从 TP1/EP1 eager ALLGATHER BF16 physical-Q32 单层 probe 开始；任何模型或性能预算仍从零计数继续，不重做已经通过的离线DEV。
