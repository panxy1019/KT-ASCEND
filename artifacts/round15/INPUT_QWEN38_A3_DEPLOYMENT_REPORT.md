# Qwen3.8-Flash-Next A3 部署性能初步报告（MTP 关闭）

## 1. 结论摘要

本轮在 A3 机器 `110.123.0.3` 上完成了 BF16 原始权重与 W8A8 权重的
TP=8 部署验证。根据用户最后指令，MTP 实验已停止，所有本次创建的容器
均已删除，16 张卡已释放。

当前可确认的结论如下：

1. **单并发 decode 延迟：BF16 略优于 W8A8。** 排除首次运行效应后，
   BF16 为 **19.76 output tok/s、41.82 ms/输出 token**；W8A8 为
   **19.06 output tok/s、43.86 ms/输出 token**。BF16 的输出吞吐高
   **3.66%**，或等价地，W8A8 比 BF16 低 **3.53%**。
2. **W8A8 的主要收益是容量，不是单请求速度。** 每 rank 模型权重从
   BF16 的 42.94 GB 降到 28.84 GB，减少 **32.84%**；可用 KV cache
   从 9.48 GiB 提升到 23.58 GiB，提升到 **2.49×**；8K 请求的理论
   最大并发从 10.15× 提升到 25.30×。
3. **W8A8 并发 8 的稳定输出吞吐为 62.38 tok/s。** 相对其单并发
   19.06 tok/s，批量吞吐约为 **3.27×**。对应平均 TTFT 为 10.43 秒，
   因此这是偏吞吐的工作点，不是低延迟工作点。
4. **BF16 并发 8 暂无可用正式结果。** 该轮与另一套 8 卡模型的加载/
   编译同时进行，两个服务都开启了 CPU 绑核，出现明显主机侧资源竞争；
   测试被按“尽快释放机器”的要求终止，未生成成功完成的 JSON。该无效
   样本不参与 BF16/W8A8 高并发结论。
5. **W8A8 decode 时间轴显示通信不是唯一瓶颈，但不可忽略。** 4.193 秒
   rank 0 稳态窗口内推断出 86 个 decode cycle；导出的设备侧 HCCL 时间
   平均约 **3.86 ms/cycle**，占该窗口墙钟时间约 **7.92%**。通信时间中
   AllReduce 占 71.22%，AllGather 占 28.78%。

因此，若业务目标是低并发、单请求 decode 延迟，当前 BF16 基线更好；若
目标是显存容量、长上下文或更多并发，W8A8 的容量优势非常明显。由于缺少
无干扰的 BF16 并发 8 数据，当前不能严谨宣称 W8A8 在高并发下比 BF16
快多少。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| 机器 | A3，`110.123.0.3` |
| 加速卡 | 16 个物理 device，单个变体使用 8 卡 |
| 镜像 | `quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler` |
| vLLM | 0.26.0 |
| BF16 权重 | `/home/admin/models/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| W8A8 权重 | `/home/admin/models/Qwen3.8-Flash-Next-w8a8-mtp`，本报告关闭 MTP |
| 并行方式 | TP=8、DP=1、EP 开启 |
| 模式 | text-only / language-model-only |
| 最大上下文 | 8192 tokens |
| 最大序列数 | 8 |
| 最大 batch tokens | 4096 |
| 显存利用率 | 0.90 |
| decode 图 | `FULL_DECODE_ONLY`，capture sizes `[1,2,4,8]` |
| 权重加载 | `--safetensors-load-strategy eager` |

W8A8 比 BF16 只额外启用了：

```text
--quantization ascend
--additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false}}'
```

BF16 使用：

```text
--additional-config '{"enable_cpu_binding":true}'
```

说明：限制容器只挂载 8 个物理 device 后，CANN 会将容器内可见卡重新编号为
逻辑 `0..7`。因此物理卡 8–15 的容器内部也必须使用
`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。

## 3. 测试方案

### 3.1 吞吐 workload

所有有效吞吐测试使用同一 workload：

- 数据集：vLLM `random`；
- 固定输入：1024 tokens；
- 固定输出：256 tokens；
- `--ignore-eos`，保证每个请求实际生成 256 tokens；
- 并发 1：每轮 8 个请求；
- 并发 8：每轮 32 个请求；
- seed：0、1、2；
- 记录 TTFT、TPOT、ITL、request throughput、output token throughput 和
  total token throughput。

每组 seed=0 同时也是服务启动后的首个完整实测轮次，容易包含首次图路径、
调度器和缓存建立开销。因此报告同时保留：

- **全样本结果**：seed 0、1、2；
- **稳态结果**：seed 1、2；这是主要横向比较口径。

### 3.2 decode profiler

profiler 客户端先发起流式长输出请求，收到第一个 token 后才调用
`/start_profile`，4 秒后调用 `/stop_profile`。因此窗口以稳态 decode 为主，
不混入初始 prefill。该次 W8A8 请求生成 2048 tokens，客户端墙钟吞吐为
22.65 tok/s。

## 4. 吞吐与时延结果

### 4.1 单并发：BF16 与 W8A8 可直接比较

| 权重 | 全 3 轮输出吞吐，tok/s | 稳态输出吞吐，tok/s | 稳态 TPOT，ms | 稳态平均 TTFT，ms | 稳态 total tok/s |
|---|---:|---:|---:|---:|---:|
| BF16 | 19.23 ± 0.92 | **19.7565 ± 0.0027** | **41.8208** | 2293.18 | 98.78 |
| W8A8 | 18.52 ± 0.94 | **19.0595 ± 0.0020** | **43.8636** | 2246.16 | 95.30 |

解释：

- BF16 稳态输出吞吐比 W8A8 高 **3.66%**；
- W8A8 的 TTFT 略低约 47 ms，但相对约 2.25 秒 TTFT 仅约 2%，意义较小；
- W8A8 的 decode TPOT 高约 2.04 ms/token，说明当前 Ascend W8A8 路径在
  batch=1 时没有表现出权重带宽节省带来的净加速，量化算子、Cast 和调度
  开销抵消了收益。

### 4.2 W8A8 并发 8

| 口径 | 输出吞吐，tok/s | total tok/s | 平均 TTFT，ms | 平均 TPOT，ms |
|---|---:|---:|---:|---:|
| 全 3 轮 | 56.99 ± 9.33 | 284.97 | 12274.95 | 95.04 |
| 稳态 seed 1–2 | **62.3774 ± 0.1114** | **311.89** | **10429.25** | **87.77** |

seed 0 为首次并发 8 运行，输出吞吐 46.23 tok/s；seed 1、2 分别为
62.46 和 62.30 tok/s。首次批量路径的额外开销很明显，因此生产压测应在
服务启动后先做与目标 batch shape 一致的 warm-up。

### 4.3 BF16 并发 8：无效样本说明

BF16 并发 8 开始时，后八卡正并行加载/编译 W8A8+MTP；两套服务都使用
`enable_cpu_binding=true`。运行日志中一度只有 6 个请求运行、2 个等待，
生成速率约 3.3–4.4 tok/s，明显违背 BF16 单并发约 24 raw decode tok/s 的
正常尺度。该轮未完成、未生成 benchmark JSON，已按无效样本处理。

若后续只补一项实验，优先在整机无其他模型加载/编译的情况下重新跑一次
BF16 c8；否则无法回答“W8A8 的纯量化高并发收益是多少”。

## 5. 显存与容量

| 指标（每 rank） | BF16 | W8A8 | W8A8 相对变化 |
|---|---:|---:|---:|
| 模型权重 | 42.9447 GB | 28.8425 GB | **-32.84%** |
| 可用 KV cache | 9.48 GiB | 23.58 GiB | **2.49×** |
| KV token 容量 | 83,133 | 207,227 | **2.49×** |
| 8K 理论最大并发 | 10.15× | 25.30× | **2.49×** |
| torch.compile | 74.38 s | 89.85 s | W8A8 多 15.47 s |
| decode 图捕获 | 约 57 s | 约 58 s | 基本相同 |

W8A8 权重不是严格减半，原因是并非模型中所有张量都按 INT8 存储，且还有
scale、元数据以及保持高精度的层。但节省出的约 14.10 GB/rank 绝大部分
被转化为 KV cache 容量，这解释了 8K 理论并发约 2.5 倍的提升。

## 6. W8A8 decode rank 时间轴初步分析

### 6.1 采集完整性与边界

- 8 个 rank 均生成了约 448 MB 的 Ascend profiler 原始目录；
- 因用户要求尽快停止并释放机器，只完成了 rank 0 的 MindStudio 导出；
- 因此以下是 **rank 0 的代表性稳态分析**，不能据此量化 8-rank skew；
- 原始 8-rank 数据仍保留在远端 run 目录，可在不占用 NPU 的情况下继续
  离线导出。

### 6.2 每个 decode cycle 的结构

profile 窗口从客户端时间戳看为 4.1933 秒。算子计数存在完全一致的公因子
86，可还原出 86 个 decode cycle：

| 事件 | 窗口总次数 | 每 cycle 次数 | 解释 |
|---|---:|---:|---|
| `hcom_allReduce_` | 8,428 | 98 | 主要 TP/EP 同步 |
| `hcom_allGather_` | 4,214 | 49 | 每层/输出聚合路径 |
| MoE dispatch | 4,128 | 48 | 48 个 MoE block |
| sparse QSA attention | 1,032 | 12 | 12 个 attention block |
| recurrent GDN | 3,096 | 36 | 36 个线性/recurrent block |

这与 12 个 attention block + 36 个 GDN block = 48 个混合 block 的执行结构
相吻合。按 86 个 cycle 计算，profiler 窗口内平均约 **48.76 ms/cycle**。
非 profiler 长输出请求的客户端平均约 44.15 ms/token，因此 profiler 引入了
约 10% 的可见开销；时间轴绝对值应谨慎使用，相对构成更可靠。

### 6.3 通信构成

| 通信 | 总设备时间，µs | 每 cycle，ms | 通信内部占比 |
|---|---:|---:|---:|
| AllReduce | 236,399.277 | 2.749 | 71.218% |
| AllGather | 95,537.152 | 1.111 | 28.782% |
| 合计 | 331,936.429 | **3.860** | 100% |

设备侧导出的 HCCL 总时间约占 4.1933 秒窗口的 **7.92%**。另一方面，
MindStudio 的算子累计时间中 `allreduceAicpuKernel` 占 27.66%，
`allgatherAicpuKernel` 占 12.61%，两者合计 40.27%。后者是异步任务累计值，
会与设备计算重叠，**不能当作墙钟占比**；但它说明 AICPU/通信调度是值得
继续优化的热点。

rank 0 的其他主要累计热点包括 MatMulV2 7.45%、Mul 5.20%、Cast 3.70%、
MoE dispatch 3.64%、QSA attention 3.51%、GroupedMatmulSwigluQuant 3.04%。
W8A8 中 Cast、量化 GroupedMatmul 与 MoE 通信共同存在，符合“显存容量收益
明显，但 batch=1 速度没有提升”的测量结果。

## 7. 启动与部署观察

### 7.1 eager loader 是本镜像/权重组合的必要规避项

默认 mmap safetensors loader 在 W8A8 的 10/61 分片后停滞超过 16 分钟：
每个 rank 单线程高 CPU、磁盘读取计数不增长、无 HCCL/NPU 活动且无报错。
改为 `--safetensors-load-strategy eager` 后，61 个分片在约 322 秒内完成。

该变化只影响启动 I/O/主机内存行为；BF16 与 W8A8 的运行时测试均使用
同一个 eager 策略，因此不破坏推理性能对比。

### 7.2 启动耗时不是本报告的主要比较指标

两个模型曾并行从同一磁盘加载，因此其后续启动耗时受到共享 I/O 影响。
报告只使用模型权重加载完成后的吞吐和 profiler 数据，不将并行加载时间
解释为 BF16/W8A8 的固有差异。

## 8. 机器释放状态

停止操作后已复核：

- `qwen38-fourway-front`：已停止并删除；
- `qwen38-fourway-active`：已停止并删除；
- NPU 0–7（对应 16 个物理 chip 0–15）：全部显示
  `No running processes found`；
- HBM 回到约 2.9–3.1 GB 的驱动/系统基线；
- 未停止或删除用户的 `mzy-train` 非 NPU 容器。

## 9. 数据与复现材料

本地材料：

- `results/summary/summary.csv`：全样本与稳态聚合结果；
- `results/summary/summary.json`：含每轮原始值的结构化汇总；
- `results/raw/bf16-off-benchmarks/`：BF16 有效 c1 JSON 与日志；
- `results/raw/w8a8-off-benchmarks/`：W8A8 c1/c8 JSON 与日志；
- `results/profile-summary/decode_client.json`：稳态 decode 客户端记录；
- `results/profile-summary/rank0-communication_statistic_*.csv`：rank 0 通信统计；
- `results/profile-summary/rank0-op_statistic_*.csv`：rank 0 算子统计；
- `results/configs/*.run.env`：实际部署命令与 device 映射；
- `run_variant.sh`、`benchmark_variant.sh`、`capture_decode.py`：复现脚本。

远端原始 run 根目录：

```text
/home/admin/qwen38-fourway/20260928-1525
```

其中 W8A8-off 的 8-rank profiler 原始数据仍在：

```text
/home/admin/qwen38-fourway/20260928-1525/w8a8-off/profiles
```

## 10. 后续最小补测建议

如需把“初步结果”升级为完整结论，最小补测集为：

1. 整机没有其他模型加载/编译时，单独启动 BF16-off；
2. 只跑一次预热 c8，再跑两次正式 c8；
3. 保持 1024 输入、256 输出、32 请求、并发 8 不变；
4. 离线导出其余 7 个 W8A8 rank，并补一轮 BF16 8-rank profiler；
5. 比较 rank 首尾偏斜、每 cycle HCCL 时间及最慢 rank。

在完成该补测前，推荐的保守决策是：低并发优先 BF16；容量/长上下文/高并发
上限优先 W8A8，但不要把 62.38 tok/s 与尚未获得的 BF16 c8 直接比较。
