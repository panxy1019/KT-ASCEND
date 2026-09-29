# Round15 基线、范围与实验方法

## 1. 起点身份

Round15 从 Round14a 提交 `e3a443ee7bb3e075b6ca3bfe19709b2a183c95fb` 和真实运行快照恢复 S0。Git HEAD 只用于确定代码谱系，真正的运行父身份是 `artifacts/round14a/a-gdn-model-snapshots/v295`，其 manifest SHA-256 为 `5639b0a21ac4f94c0c2ca6f5ca66248a0dddcd3b5a83a2eb1f25b4899a7c28cd`。

固定配置如下：

| 项目 | S0 冻结值 |
|---|---|
| 模型 | Qwen3.8 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| CPU | Kunpeng C128，4 个 NUMA pool × 32 worker，NUMA 2/3/4/7 |
| CPU Expert | 只读 MXFP4 bank |
| NPU resident | Static Uniform Q32，BF16/NZ29 |
| Scheduler | chunk32 |
| PLE | ASYNC |
| QSA | VECTORIZED Prefill＋DIRECT_BF16_ROW Decode |
| GDN | Round14a promoted NPU conv/recurrence＋本地 sigmoid gate/norm/out projection |
| Hybrid | H129；当前窄资格输入、输出均不超过 128 tokens |
| 并行 | TP1、并发1、单请求串行 |
| 关闭项 | Dynamic、deferred、R2、R3、transpose-view、shadow320、Graph |

运行设备是 A3 主机物理 NPU 12，对应 `npu-smi card 6 / chip 0`。S0 镜像为 `kt-round64-migration:20260916`，实际运行环境为 aarch64 Ubuntu 22.04、Python 3.11.15、torch 2.9.0、torch_npu 2.9.0.post2、CANN 9.0.0。

用于机制调查的 donor 镜像是 `quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler`，其 torch/torch_npu/CANN 分别是 2.10.0、2.10.0.post4、9.1.0。donor 只提供算子和图机制入口，不能把其 TP8、不同精度或不同负载性能外推到本轮单卡 KT。

权威身份见 [ROUND15_BASELINE_MANIFEST.json](../../../artifacts/round15/ROUND15_BASELINE_MANIFEST.json)。

## 2. 研究问题

### A：成熟 NPU 算子

先从真实 S0 profile 选择一个热点算子族，再要求：旧 provider 可回退、候选默认关闭、独立 reference、真实权重、多形状、状态复用和资源回收。候选组件过门后才进入模型请求。

### B：KT 分段图

研究 Decode batch1 中一个稳定、纯 NPU 且不跨 CPU 依赖的区域。CPU Expert、PLE Host 工作、scheduler、取消和模型状态不得被塞进图；不允许借 Graph 引入 deferred、删同步或冻结 capture 时的 token/route。

## 3. 工作负载和指标

本轮请求对照聚焦：

| ID | 实际输入/输出 | 用途 |
|---|---:|---|
| W1 | 128 / 16 | Prefill、TTFT 和短生成 |
| W2 | 128 / 128 | Decode、TPOT 和串行吞吐主观察点 |

请求进程另外包含五类确定性 canary、多步 Decode、取消、同进程复用和 warmup。候选必须真实命中；只看到环境变量打开或 eager fallback 不能算候选接入成功。

指标定义：

- `TTFT`：请求提交至首个真实输出 token。
- `TPOT`：从首 token 到末 token的间隔除以 `completion_tokens - 1`。
- `E2E`：请求提交至请求完成。
- 串行 output tokens/s：合格测量窗口内的实际输出 token 总数除以连续墙钟时间。
- requests/s：同一连续窗口内成功请求数除以窗口时间。

`1 / TPOT` 不被当作完整服务吞吐；不同层次的 parent/child span 也不相加。

## 4. 执行中缩减的请求证据

原计划要求四进程 ABBA。用户随后明确要求“大幅简化，只需要测出来大体性能”，因此请求阶段改为：

```text
A: v341 baseline → v342 candidate
B: v354 baseline → v360 candidate
```

每个进程包含 14 次 generation attempt、1 次取消控制；每个 W1/W2 是 1 次 warmup＋2 次 measurement。缩减只改变请求性能证据强度，没有改变组件数值、状态、取消、清理或资源门。

所以本轮使用三个不同结论层次：

1. **组件通过**：说明某个冻结输入合同下实现正确并可更快。
2. **方向性请求结果**：说明一个基线进程和一个候选进程中的大致变化。
3. **默认晋级**：仍要求收益足够大、保护指标不退化；A/B 均未满足。

详细缩减合同见 [ROUND15_SCOPE_AMENDMENT.json](../../../artifacts/round15/ROUND15_SCOPE_AMENDMENT.json)。

## 5. 不可变项和排除项

本轮没有改变 CPU bank、resident map、精度、chunk、PLE、Hybrid 阈值、TP、并发度和采样合同；没有重新打开 NZ update、R2/R3、shadow320、Dynamic、deferred、HiCache、KV offload、MTP 或多卡 KT。B 的图候选也没有迁移 A 的 QSA 候选，因为 A 未晋级；B 使用不变的 S0 作为父配置。

## 6. 终态判定

最终状态是 `ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION`。这表示：在用户缩减后的请求证据口径下，P0、A、B 都有明确技术去留，安全与资源合同闭环，但没有候选取得足够请求收益。`PERFORMANCE_OBJECTIVE` 因此是 `NOT_ACHIEVED`，最终配置不变。
