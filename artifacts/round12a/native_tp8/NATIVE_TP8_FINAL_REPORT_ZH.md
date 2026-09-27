# Round12a Native SGLang TP8 / old-QSA 最终报告

状态：`PASS_ROUND12A_NATIVE_TP8_BASELINE_ABBA`。

## 执行范围

- 模型 revision：`de4b8e4d43b917e7706784d8bb445c9af86a3540`。
- Native SGLang，TP8，BF16，物理 NPU 0–7，`cpu_offload_gb=0`，无 KT weight。
- QSA 为旧实现，SHA-256 `76e9f34066f0b42f42545427967c0b789d5c32031dc5c085e230efaf95e3c54c`。
- 固定请求 W1：64 个输入 token、2 个生成 token、temperature 0、seed 20260920。
- v180/v181/v182/v183 按 A/B/B/A 顺序运行；A/B 只是相同实现的独立重复标签。
- “pure-NPU”仅指主模型和 Expert 权重及计算；PLE checkpoint row table 仍驻 Host。

## 质量与真实性

四个独立模型进程各执行 1 次 warmup 和 3 次 measurement，共 16 次 generation。12 次正式测量全部返回精确 token IDs `[3897, 248046]`，logprob 有限。每个进程都验证了八个 rank 的 48 个 BF16/NPU Expert 层、36 个 TP-local GDN 层、12 个 old-QSA 层以及 PLE ASYNC drain。

## 性能

| 进程 | E2E 中位数（秒） | TTFT 中位数（秒） | TPOT 中位数（秒） |
|---|---:|---:|---:|
| v180 A | 3.428953 | 3.172701 | 0.256151 |
| v181 B | 3.371347 | 3.102270 | 0.269113 |
| v182 B | 3.489021 | 3.210625 | 0.278328 |
| v183 A | 3.325310 | 3.063772 | 0.262339 |

四个进程中位数的最大/最小比为 `1.049231636`，通过冻结的 5% 稳定门。四进程中位数的中位数为 `3.400150` 秒。

## 资源

逐卡峰值约 36.47–36.79 GiB；跨进程同一卡峰值离散仅 0–2 MiB。八卡同时峰值使用量中位数为 `292857.5` MiB（包含每卡约 2.87–3.13 GiB 的系统基线）。Host cgroup 峰值为 293.8376–293.8530 GiB，最大/最小比 `1.000052514`。所有 guard 均确认退出，容器均删除。

每个进程归档和全 SHA 校验后执行一次定向缓存退休。四次分别释放 61,833,268、61,837,371、61,835,095、61,837,525 页，最终均为零；没有调用全局 `drop_caches`。

## 无设备预检的处置

三次无设备同镜像预检依次暴露 `is_npu()`、Qwen native extension 和 `torch.npu.mem_get_info()` 对真实设备的依赖。两次修复额度用尽后没有增加第三个假设备 shim。四个正式进程在权重加载前执行真实设备 ServerArgs 和 source SHA gate，均通过。旧失败证据保留在 `remote-evidence/preflight/`。

## 结论边界

Native TP8/old-QSA 在本次 W1 串行短请求上质量、性能和资源可重复。由于它使用 old QSA，而主 KT/OffloaderV1 ABBA 使用 promoted QSA，Native 结果只能作为系统上下文，不能直接写成 KT 相对 Native 的因果加速比。


## 最终复核

最终重新运行 Native TP8 的 18 项合同测试，比较器与冻结结果逐字段完全一致；四个 run 及预检的 156 项归档 SHA 全部匹配。2026-09-27 06:22:31 UTC 的新远端只读核验确认物理0–7为空，相关容器、控制器和定时器均不存在。
