# Round16 最终报告

生成时间：2026-09-29 11:31 UTC。

## 结论

本轮终态是 `ROUND16_PARTIAL_WITH_GAPS`。B 线已完成真实 Q32 构造、选择性
加载、混合专家、48 层容量、完整模型、取消/复用和四进程请求对照，但请求
性能没有晋级；A 线完成了两个 fresh profile 块，却缺少能够安全隔离的 CPU
暴露等待和 Host launch/metadata 关键区间，因此没有进入 A2 或 A3。默认承载
继续使用 S0（SGLang + KT）。

这个终态中的 `PARTIAL` 只对应 A 的关键路径归因缺口。B 不是集成未完成，
也不是样本内不确定；B 是 `B16_MEASURED_NO_PROMOTION`。

## A：3.x tok/s 的剩余限制

两个 fresh S0 块都完成了 W1/W2 的直接设备事件记录。QSA indexer 是已采样
NPU 区域中最大的单项：A/B 分别为 2.868/2.986 ms；GDN 为
1.057/1.088 ms，resident expert 为 0.416/0.434 ms。

这些数值不能相加成请求关键路径，也不能据此写“QSA 是 3.x tok/s 的最终
瓶颈”。现有 native CPU worker 的 `submit_forward/sync_forward` 不提供 worker
起止 callback。虽然 Host submit/sync 调用边界可见，但证据中没有直接隔离的
`HOST_SYNC_EXPOSED_WAIT` 或 `HOST_LAUNCH_METADATA_CRITICAL` 区间。用 parent
减 children、跨 Host/NPU 时钟相加或把全部 sync 调用求和后冒充连续关键区间
都会违反冻结比较合同。

因此 A2 精确同层 overlap 和 A3 更大 NPU 分段图均为
`NOT_SELECTED_EVIDENCE_GAP`。A 没有候选、没有模型 smoke，也没有请求收益或
无收益结论。

## B：从 scaffold 到真实运行

B 已把 512 个逻辑专家与每层 32 个 NPU 物理槽位分开：

- V1 在真实镜像构造一层 Q32，payload 为 314,572,800 bytes，未创建 512 行
  NPU 专家 payload。
- V2 从原 512 专家 checkpoint 在 Host 侧选择 resident 行，覆盖 32×3 个
  gate/up/down 投影；加载和 Ascend post-load 峰值保持在受控 Q32 范围。
- V3 用真实 C128/MXFP4 与 BF16/NZ29 跑通 all-CPU、all-NPU 和 mixed Top-10，
  保持一次 Top-K、一次 route weight、一次同层 merge；20 轮生命周期、取消
  drain 和 repeat hash 通过。
- V4 同时保留 48 个 owner、96 份 storage，allocator peak 为
  15,099,666,432 bytes，随后回到组件基线。
- V5 完整 TP1 eager 模型通过五类 exact canary、32-token Decode、取消后无
  unfinished request 和同进程 exact reuse；所有选中 logprob 有限。

B 由 vLLM 持有 request、KV、QSA/GDN/PLE state、batch、cancel 和 token 提交；
KT bridge 只持有专家计算资源、输入输出 buffer 与短期 completion lease。执行
模式为 eager，`CUDAGraphMode.NONE`。它保留了桥外的原生 vLLM-Ascend 主干算子，
没有保留图 replay，也不能继承 FULL_DECODE_ONLY 的性能标签。

## KV：不能由单页 unsqueeze 宣布解决

`unsqueeze(0)` 只修复单页 allocator 丢失 page 轴时的局部 view 形状。它不是
KV 正确性资格。

独立真实 NPU 探针使用 `[3, 3072, 2, 256]` cache、非恒等 block table
`[[2, 0, 1]]` 和跨三页的 12 个逻辑位置；两 KV heads 均精确写入，24 query
heads / 2 KV heads 的 GQA sparse attention 对独立 reference 的 relative L2 为
`0.00184652`，max abs 为 `0.015625`。这证明当前组件的跨页地址、multi-head
store 与 sparse GQA 消费者能够共同工作。

完整模型只验证了本轮范围：输入与输出各不超过 128、累计 context 不超过
256。没有运行长上下文完整模型跨多页生成。因此结论是“多页组件通过＋本轮
模型范围通过”，不是“所有 KV 场景完全解决”。

## 四进程请求对照

运行顺序是 `S0-A -> B-A -> B-B -> S0-B`。每个进程 profiler OFF，W1、W2、
W4 各 1 warmup＋2 measurement，再加 1 个 exact arithmetic canary；所有性能
请求都生成完整预算 token，没有通过早停获得速度。

| Block | Workload | E2E s | TTFT s | TPOT s | 串行 output tok/s |
|---|---|---:|---:|---:|---:|
| S0-A | W1 | 6.2865 | 2.5326 | 0.25025 | 2.5452 |
| B-A | W1 | 6.8409 | 2.4779 | 0.29087 | 2.3389 |
| B-B | W1 | 6.6691 | 2.4406 | 0.28190 | 2.3991 |
| S0-B | W1 | 6.4293 | 2.5732 | 0.25706 | 2.4886 |
| S0-A | W2 | 34.6947 | 2.5394 | 0.25319 | 3.6893 |
| B-A | W2 | 41.9237 | 2.5329 | 0.31016 | 3.0532 |
| B-B | W2 | 42.3549 | 2.4759 | 0.31401 | 3.0221 |
| S0-B | W2 | 36.0147 | 2.6268 | 0.26290 | 3.5541 |
| S0-A | W4 | 9.2313 | 1.2581 | 0.25720 | 3.4665 |
| B-A | W4 | 10.5995 | 1.3658 | 0.29786 | 3.0190 |
| B-B | W4 | 10.9684 | 1.4035 | 0.30854 | 2.9175 |
| S0-B | W4 | 9.2265 | 1.2883 | 0.25607 | 3.4683 |

W2 的两个配对块中，B 的 TPOT 分别增加 22.50% 和 19.44%，串行 throughput
分别下降 17.24% 和 14.97%，E2E 分别增加 20.84% 和 17.60%。它既没有达到
“TPOT 至少下降 10%＋吞吐至少提高 10%”，W4 的保护门也在两块中都失败。

这是系统部署方案比较：两边框架与图模式不同，不能写成纯 runtime 消融。
它足以回答默认承载是否应切换：当前不值得，保留 S0。

## 资源与回收

四个正式请求进程均通过 2×10 秒 fresh admission。最低平均 CPU idle 仍高于
99.6%，四个 NUMA 节点最低可用内存的下界约为 120.29 GiB。B 日志记录模型
权重 24.2668 GB、peak activation 2.85 GiB、KV cache 23.04 GiB／3730 tokens，
warmup 后 torch reserved/allocated 为 47.34/47.30 GiB。B wrapper 没有记录
Host cgroup 峰值，该字段保持 UNKNOWN。

每个实验块退出并完成证据归档/SHA 后，对 131 个模型文件和 48 个 bank 文件
执行有引用检查与定向 `POSIX_FADV_DONTNEED`。四次释放 296.55 GB、348.99 GB、
209.02 GB 和 293.70 GB；`fincore` 复核登记集合 resident bytes 为 0。没有调用
全局 `drop_caches`，没有修改外部任务。

## 必答问题

1. **现在 3.x tok/s 的关键限制是什么，A2/A3 为什么选或不选？** 当前证据
   不能安全给出唯一关键限制。QSA 是最大已采样 NPU span，但 CPU 暴露等待与
   Host launch 仍未直接隔离。A2/A3 因证据边界未选择。
2. **A 是否减少真实请求成本？** 没有形成安全候选，因此未测，不能写 NO_GAIN。
3. **B 接到哪一级，融合/图保留多少？** 接到完整 TP1 eager 模型和新鲜四进程
   请求对照；桥外原生主干仍在，图 replay 为 NONE。
4. **同资源 A/B 表现如何，默认承载是否值得更换？** A16=S0。B 在两块 W2
   中都显著慢于 S0，且 W4 退化，因此不更换默认承载。

本轮未 push、merge、tag、reset 或 clean，也未恢复历史 timer。
