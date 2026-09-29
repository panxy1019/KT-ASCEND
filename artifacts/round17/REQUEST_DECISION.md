# Round17 请求级决定

## Fresh S0

本轮重新运行两个独立 profiler-OFF 模型进程。每块固定执行 W1/W2/W4 各 1 次 warmup＋2 次 measurement，再执行 1 条 canary，共 10 次 generation。所有性能请求都生成完整固定 token 数；canary 按冻结 stop token 正常结束。

| Block | Workload | TTFT (s) | TPOT (ms) | E2E (s) | serial output tok/s |
|---|---|---:|---:|---:|---:|
| v519 S0-A | W1 128/16 | 2.5518 | 270.235 | 6.6055 | 2.4222 |
| v519 S0-A | W2 128/128 | 2.5983 | 259.926 | 35.6092 | 3.5946 |
| v519 S0-A | W4 64/32 | 1.3005 | 259.290 | 9.3386 | 3.4266 |
| v520 S0-B | W1 128/16 | 2.6858 | 273.251 | 6.7847 | 2.3583 |
| v520 S0-B | W2 128/128 | 2.6217 | 265.632 | 36.3572 | 3.5206 |
| v520 S0-B | W4 64/32 | 1.3057 | 263.024 | 9.4595 | 3.3828 |

W2 两块中位的中位为 262.779 ms/token、3.5576 output tok/s。该数字只描述本轮 fresh S0；正式候选不存在，所以没有将其与跨轮旧数拼成 ABBA。

## 为什么没有候选 ABBA

A 的成熟原语覆盖部分即使理想归零，也只能改善完整 QSA parent 约 12.4%–13.4%，低于 20% 组件门。B 没有找到能把必要设备计算与可减少 launch/glue 分开的 bounded segment，也没有完整层 15% 的可测预期。根据冻结合同，未过组件选择门的候选不得消耗请求进程或写成请求负收益。

因此本轮请求结论是：

- `candidate request ABBA = NOT_RUN_NO_COMPONENT_CANDIDATE`；
- `performance promotion = NONE`；
- S0 保持默认；
- 不能写成“A/B 实现跑过但无收益”，因为本轮没有构建这两个高风险实现。

## 资源

两块 whole-card HBM 峰值均为 28,933 MiB；相对监控到的 board baseline，估算 owned 峰值 25,651 MiB。Host cgroup `memory.max_usage_in_bytes` 分别折合 341.85 GiB 和 341.24 GiB，主要包含可退休的只读 file cache。结束后容器为 0、物理 12 进程为 0；完成 SHA 后仅对已登记且无人引用的 bank/model 执行了定向 DONTNEED。
