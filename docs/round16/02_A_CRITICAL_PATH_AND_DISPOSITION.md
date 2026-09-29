# A 线：关键路径归因与候选去留

## 1. Fresh profile 设计

A1 使用两个独立 S0 模型进程。每个块固定六个请求：

1. W1 warmup；
2. W2 warmup；
3. W1 measurement；
4. W2 measurement；
5. short drain；
6. fixed canary。

observer 默认关闭，启用后不增加 `torch.npu.synchronize()`。Host 记录使用 `HOST_MONOTONIC_NS`，NPU provider 使用已有 event elapsed time。两种时钟域不合并。

observer 覆盖：

- 48 层 routed-expert parent；
- `KTEPWrapperMethod._submit_cpu_forward`；
- `KTEPWrapperMethod._sync_cpu_forward`；
- 既有 stream wait 后的 Host merge；
- 早/中/晚层 QSA、GDN 和 resident expert 的直接 device event；
- serial scheduler 的 Decode bounding interval。

## 2. 直接诊断结果

| Fresh block | QSA indexer | GDN | NPU resident expert |
|---|---:|---:|---:|
| A | 2.868250 ms | 1.057100 ms | 0.416410 ms |
| B | 2.986330 ms | 1.087900 ms | 0.434150 ms |

这些数值是直接 provider event 的中位数，只用于描述已观察区域。它们不是 candidate score，不能简单乘层数，也不能与 Host wall 相加。

## 3. CPU Expert 为什么仍是 UNKNOWN

现有 `submit_forward`/`sync_forward` ABI 只提供提交与取回边界，没有：

- native worker 的开始时间；
- native worker 的结束时间；
- work ID 对应的 worker-clock event；
- 跨 pool 的统一完成 callback。

所以 CPU Expert 被记录为：

```text
UNKNOWN_NATIVE_ABI_NO_CALLBACK
```

Host submit 调用时间不是 CPU compute；Host sync 调用时间也不全等于暴露等待。异步 CPU 可能在 NPU 工作期间已经执行，sync 中还可能包含结果包装或其他固定开销。

## 4. A2 为什么没有进入

A2 要求两个块同时证明：

1. 直接 `HOST_SYNC_EXPOSED_WAIT` 至少占 Decode wall 的 10%；
2. 它是首要可操作阻塞；
3. 同层存在尚未使用、且不依赖 CPU 结果的 NPU 工作窗口；
4. CPU 和 NPU 均完成后才进行本层 merge；
5. 不跨层推迟专家贡献。

当前数据只有 Host sync 调用 span，没有冻结定义要求的 exposed-wait 连续区间，也没有一个可与其成对证明的 unused same-layer NPU window。因此 A2 为 `NOT_SELECTED_EVIDENCE_GAP`。

## 5. A3 为什么没有进入

A3 要求两个块直接证明 Host launch/metadata 或由其造成的 NPU 空洞达到 Decode wall 的 10% 量级，并且候选区域必须明显大于 Round15 已拒绝的 resident-only graph。

现有 trace 记录 Host dispatch 调用，但没有隔离出一个直接 `HOST_LAUNCH_METADATA_CRITICAL` 连续区间。将所有层的 dispatch span相加会重复计算串行与嵌套关系，也不能证明 NPU 当时空闲。因此 A3 同样为 `NOT_SELECTED_EVIDENCE_GAP`。

## 6. 当前可写和不可写的结论

可写：

- 两个 fresh profile 块完成；
- QSA 是最大的已采样 NPU provider span；
- CPU worker 的内部时间和真实 overlap 未被 ABI 暴露；
- A2/A3 没有满足证据门；
- A16 与 S0 相同。

不可写：

- “CPU Expert 不是瓶颈”；
- “QSA 占请求的某个精确百分比”；
- “A2/A3 已证明无收益”；
- 把 Host sync 全部求和当作可消除等待；
- 把 parent 减去已知 children 得到 UNKNOWN。

## 7. 权威证据

- [A1_CRITICAL_PATH.json](../../artifacts/round16-a/A1_CRITICAL_PATH.json)
- [A1_CRITICAL_PATH.md](../../artifacts/round16-a/A1_CRITICAL_PATH.md)
- [A_CANDIDATE_DECISION.json](../../artifacts/round16-a/A_CANDIDATE_DECISION.json)
- [A_REGION_DEPENDENCY_DAG.json](../../artifacts/round16-a/A_REGION_DEPENDENCY_DAG.json)
- [A 当前状态](../../artifacts/round16-a/CURRENT_EXECUTION.md)

原始大体积 JSONL trace 保留在本地工作树和集群 evidence 目录；GitHub 提交保留归一化分析、receipt、绑定身份与 SHA，而不提交四个约 58–61 MB 的可再生 trace 文件。
