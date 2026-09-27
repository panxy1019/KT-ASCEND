# Round12a 最终报告：tied-aware OffloaderV1、KT/non-KT 公平 ABBA 与 Native TP8 补充基线

最终状态：`ROUND12A_COMPLETE_KT_STYLE_VALUE_CONFIRMED`。

## 结论

在本轮严格限定的 Qwen3.8、promoted QSA、串行 W1（64 输入 token、2 输出 token）、Ascend 物理12＋Kunpeng C128 配置上，**KT-style heterogeneous execution 值得保留**。公平的四独立进程 ABBA 中，KT 相对 SGLang OffloaderV1 的 E2E 改善在两个块中分别为 `99.752454%` 和 `99.749092%`，输出 token 跨模式完全一致。

这个结论针对当前支持的 OffloaderV1 全 BF16 层 staging 路径。它不表示任意 pure-NPU、任意上下文长度或任意并发下 KT 都更快。

## 1. tied-parameter-aware OffloaderV1

Qwen3.8 的 tied 参数会让默认 `functional_call(..., tie_weights=True)` 在同一张量以多个 state key 出现时拒绝调用。候选保留 `tie_weights=True`，按精确 tensor object 去重 state，并在调用后保持 alias 身份。v172 通过 7 次数值/身份迭代：最大 relative L2 为 0，repeat hash exact，参数、buffer 和 tied alias 身份均保留，allocator 从 0 回到 0。没有放宽误差或 cleanup 门。

## 2. 第一个同 revision non-KT 请求

v173 在 promoted QSA 下跑通第一个真实 non-KT 请求：48 个完整 BF16 decoder layer 由支持的 OffloaderV1 驻 Host 并逐层 staging，VECTOR Prefill 和 ROW Decode 的 12 个 QSA 层均命中，PLE ASYNC 参与，输出和有限 logprob 正确，cleanup 完整。4-token smoke E2E 为 1984.047039 秒，超过 1800 秒请求门，因此只作为正确性和成本证据；正式 ABBA 缩为仍覆盖一次真实 Decode 的 64×2 W1。

## 3. promoted-QSA 公平 KT/non-KT ABBA

固定模型、promoted QSA、请求、采样、物理12、准入和 profiler OFF；顺序为 v174 KT / v175 NON_KT / v176 NON_KT / v177 KT。每个进程 warmup 1 次、measurement 3 次。

| Block | KT E2E 中位数 | non-KT E2E 中位数 | KT 改善 | KT TTFT | non-KT TTFT |
|---|---:|---:|---:|---:|---:|
| A | 2.982179 s | 1204.699036 s | 99.752454% | 2.638159 s | 798.590287 s |
| B | 3.386336 s | 1349.630513 s | 99.749092% | 3.006188 s | 868.332767 s |

两块都远超预先冻结的 5% E2E 价值门。12 个正式请求在各自 arm 内稳定，KT/non-KT 均输出 `[3897, 248046]`。差异来自系统 backend：KT 使用 Kunpeng C128/MXFP4 routed experts 加 Static Q32 BF16/NZ29；non-KT 使用完整 BF16 decoder layer 的 OffloaderV1 staging。它是系统级公平比较，精度/执行位置的差异正是被比较机制的一部分。

资源上，KT 两个进程的单卡峰值均为 28,773 MiB，Host cgroup 峰值为 340.094/352.644 GiB；non-KT 峰值为 22,311/22,312 MiB，Host cgroup 峰值为 482.420/482.446 GiB。OffloaderV1 节省部分 HBM，却把请求暴露时间和 Host 占用推得很高。

## 4. Native SGLang TP8 / old-QSA 补充基线

用户新增的 Native TP8 基线在物理0–7上完成 v180/v181/v182/v183 四个独立进程。它采用 BF16、old QSA、无 CPU offload、无 KT weight。四个 E2E 中位数为 3.428953、3.371347、3.489021、3.325310 秒，最大/最小比 1.049232；12 次正式测量全部精确输出 `[3897, 248046]`。跨运行逐卡 HBM 峰值离散 0–2 MiB，Host 峰值比 1.000053。

Native 八卡同时峰值使用量中位数为 292857.5 MiB，Host 峰值约 293.84 GiB。主/Expert 权重与计算在 NPU0–7，但 PLE row table 仍依赖 Host。

KT 两个 promoted-QSA 进程的 E2E 中位数之中位数为 3.184257 秒；Native old-QSA 四进程为 3.400150 秒，表面上 KT 低 6.350%。这个数字只作描述：QSA 不同且设备数量不同，不能称为公平加速。它仍提供一个有价值的系统事实：当前 KT 单 NPU＋C128 的短请求时延已与八卡 Native TP8 同量级，而公平的 same-QSA OffloaderV1 路径慢约三个数量级。

## 5. 资源、回收与预算

主线使用 3/3 component、5/8 model starts、17/44 generations；Native 补充使用独立的 4/4 model starts、16/16 generations。两者自动重试均为 0。

v173–v177 与 v180–v183 共 9 个模型块都在完整归档和 SHA 校验之后执行定向 POSIX_FADV_DONTNEED；每次先审计全机 fd/mmap，只处理登记的只读 bank/checkpoint 文件。累计退休前 resident pages 为 578,112,868，退休后为 0，未使用全局 `drop_caches`。

最终只读核验：物理0–7与物理12均无进程；无 Round12 容器、控制器、guard 或定时任务；登记训练容器完整 ID 与 cgroup 未变。

## 6. 结论边界与下一步

本轮证明的是串行 64×2 窄 workload。未资格项包括更长 Prefill/Decode、多请求并发、功耗、吞吐、产品部署以及全面质量。Native TP8 使用 old QSA，若要得出 KT 相对 Native TP8 的因果速度差，下一轮必须先建立 same promoted-QSA Native TP8 arm，并配置新的设备/质量预算。

当前无需继续消耗 Round12a 设备额度。保留 KT HFINAL，保留 tied-aware OffloaderV1 作为受支持参考路径，Native TP8 old-QSA 作为可复现上下文基线。

## 主要证据

- `ROUND12A_KT_NONKT_ABBA_RESULT.json`
- `ROUND12A_CROSS_BACKEND_SYNTHESIS.json`
- `native_tp8/NATIVE_TP8_ABBA_RESULT.json`
- `native_tp8/NATIVE_TP8_FINAL_STATUS.json`
- `ROUND12A_EXECUTION_BUDGET.json`
- `ROUND12A_RESOURCE_RETIREMENT_RECEIPT.json`
- `ROUND12A_FINAL_REMOTE_STATE.json`


## 7. 最终复核

2026-09-27 06:22:31 UTC 的新只读远端核验再次确认物理0–7与物理12均无设备进程，Round12 容器、进程和定时器均为空，登记训练容器完整 ID 与 cgroup 保持一致。初版本地过滤器曾把三个四天前已退出的 Round9 容器误纳入 Round12 作用域；失败解析已保留，修正后的核验只按 Round12 所有权判定，没有删除或修改这些历史容器。

最终本地复核重新运行 148 项单元/合同测试与 34 个 Python 文件的编译检查；主 KT/non-KT 比较器与冻结 JSON 规范化后完全一致，Native TP8 比较器逐字段一致。主线 388 项和 Native 156 项归档 SHA 均匹配，所有原始证据目录已设为只读。
