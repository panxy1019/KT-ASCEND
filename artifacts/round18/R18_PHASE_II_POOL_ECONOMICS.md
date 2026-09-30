# Round18 Phase II Native expert pool：fresh route 经济性与容量裁决

- 日期：2026-09-30 UTC
- 范围：本地只读分析；无远端写入、无设备启动
- 状态：`PASS_ROUND18_PHASE_II_POOL_ECONOMICS_DISPOSITION`
- 裁决：`NO_DEPLOYABLE_NATIVE_POOL_POLICY_UNDER_CURRENT_FULL_GRAPH`
- 是否进入完整 pool 实现：**否**

这份裁决不把“理论局部性尚可”误写成“当前实现可运行”。fresh W2 Decode
证明 Q32 pool 在容量意义上能容纳单次 Top-10，但冻结 FULL replay 不能把当前
token 的 route 交给 Host planner；唯一无需当前 token readback 的 W1 静态
频率表又存在大量 miss。LRU 和 Belady 的流量更低，但前者仍需要当前 route，
后者还需要未来 route，二者都不是当前 Native FULL 可运行策略。

## 1. 数据身份与质量边界

主数据来自 `native-route-trace-v016`：48 层、512 logical experts、Top-10，
W1/W2/W4 共 8,784 个 Native route event。原 trace、result 和 working-set
分析的 SHA-256 已写入机器可读结果。

分析前核对出一个会影响“holdout”措辞的事实：

- W1 与 W2 使用相同 128-token prompt，其 Prefill route 完全相同；
- W1 每层 15 个 Decode row，且是 W2 每层 127 个 Decode row 的严格前缀；
- W4 是不同的 64-token prompt/shape 控制。

因此 W2 是 W1 的较长 continuation，不是完全独立的 route holdout。W2 仍可
评估长 Decode 的延续行为，但不能把它与 W1 的共享前缀包装成独立泛化证据。
W4 的不同输入结果是更重要的交叉检查。

每个 miss 只按 9,830,400 bytes（9.375 MiB）BF16 gate/up/down payload
计量。所有数字都未包括 NZ padding、格式转换、metadata、publish event、同步
和无法重叠的等待，因此是 transfer 下界，不是 miss 的真实 exposed latency。

## 2. 三种口径不能混用

| 口径 | 它回答什么 | 当前 FULL 下能否直接运行 |
|---|---|---|
| Invocation working-set fit | 一个 Native MoE 调用的全部 unique experts 是否能同时放进 Q slots | 否；只是一项容量必要条件，不负责找出或加载当前 experts |
| W1 static-frequency | W1 全 trace 选出的 per-layer Top-Q 固定表，在 W2/W4 上命中多少 | 固定 map 本身可预装，但所有 workload/phase 都有 miss；没有 fill path 时结果不完整 |
| Sequential LRU | 已知当前 token-row route 后，在线替换的顺序模拟 | 否；当前 FULL 不向 Host planner 暴露当前 route |
| Sequential Belady | 使用未来 route 的离线最优替换上界 | 否；只可作为乐观下界 |

另外，W1 自身的 static/LRU/Belady 数字都复用了由完整 W1 建出的初始频率表，
包含开发数据泄漏，只用于描述，不是 qualification。

## 3. 容量裁决

冻结 whole-card cap 为 56,832 MiB（55.5 GiB），已包含至少 8,704 MiB
reserve。相同 vLLM+KT Q32 runtime 的历史同类回执为 warmup 后 torch
allocated/reserved 47.30/47.34 GiB。这个回执不是 fresh Round18 pool 容量
PASS，但比 SGLang 的 28.9 GiB board peak 更适合作为 vLLM pool 算术参考。

| Pool | 48层 routed payload | 基于 47.34 GiB Q32 回执的估算 reserved | 相对 55.5 GiB cap | 裁决 |
|---|---:|---:|---:|---|
| Q32 | 14.0625 GiB | 47.3400 GiB | +8.1600 GiB | 唯一已经实际构造过的容量；仍缺 graph-safe fill |
| Q64 | 28.1250 GiB | 61.4025 GiB | −5.9025 GiB | cap 排除 |
| Q96 | 42.1875 GiB | 75.4650 GiB | −19.9650 GiB | cap 排除 |
| Q128 | 56.2500 GiB | 89.5275 GiB | −34.0275 GiB | routed payload 单独已超过 cap |

每层每增加一个 slot，48 层合计增加 450 MiB。纯算术上 Q50 约为
55.2502 GiB，只余 255.8 MiB；它没有给 graph pool、slot staging、format
workspace 或运行波动留出合格余量，因此不是安全候选。Q48/Q49/Q50 也没有
本轮 route 分析和真实加载证据，不能从 Q32/Q64 插值出资格。

旧 SGLang `maximum_owned_hbm_mib=37376` 与 vLLM 47.30 GiB allocation 的
口径不一致。本报告保留该差异，不用旧 owned cap 冒充 vLLM 资格；硬裁决采用
共同的 whole-card cap。

## 4. Native invocation working set

| Workload / phase | unique experts min / median / max | Q32 fit | Q64 fit | Q96 fit | Q128 fit |
|---|---:|---:|---:|---:|---:|
| W1 Prefill | 10 / 83 / 160 | 13.54% | 33.85% | 63.54% | 97.40% |
| W1 Decode | 10 / 10 / 10 | 100% | 100% | 100% | 100% |
| W2 Prefill | 10 / 83 / 160 | 13.54% | 33.85% | 63.54% | 97.40% |
| W2 Decode | 10 / 10 / 10 | 100% | 100% | 100% | 100% |
| W4 Prefill | 72 / 111 / 182 | 0% | 0% | 16.67% | 82.29% |
| W4 Decode | 10 / 10 / 10 | 100% | 100% | 100% | 100% |

结论分两部分：

1. Decode-only pool 在结构上有希望，因为每次调用只有 Top-10，Q32 就能容纳；
2. 同一 Q32/Q64 pool 不能直接执行大多数完整 Prefill invocation。若未来只做
   FULL_DECODE pool，Prefill 必须保留单独的已资格 fallback，或另行证明
   waves/staging；本报告不把 sequential token-row oracle 当成 Prefill 可运行证据。

## 5. Miss payload：W1 静态频率表

下表是当前唯一不需要读取“当前 token route”即可预装的 policy。值为
`GiB / generated output token`，只计 BF16 payload。

| Workload / phase | Q32 | Q64 | Q96 | Q128 |
|---|---:|---:|---:|---:|
| W1 Prefill（DEV self） | 5.6002 | 3.5780 | 2.1509 | 1.1501 |
| W1 Decode（DEV self） | 1.3521 | 0.6048 | 0.3296 | 0.1596 |
| W2 Prefill（与 W1 完全相同） | 0.7000 | 0.4472 | 0.2689 | 0.1438 |
| W2 Decode | 1.6593 | 0.9650 | 0.6755 | 0.5382 |
| W4 Prefill | 2.5483 | 2.0393 | 1.6065 | 1.2597 |
| W4 Decode | 1.3204 | 0.6372 | 0.3920 | 0.2592 |

Prefill 的 per-output 分母会随请求输出预算变化，不能用来比较 Prefill 本身的
每输入 token 成本；机器可读文件同时保存了 total GiB 和 miss count。关键事实
不是某个分母，而是固定表始终存在 miss。没有动态 fill 时，static policy 不能
计算完整模型结果。

Q32 static 在 W2/W4 Decode 的总 miss 分别是 23,199/4,615 个专家，即
212.393/42.252 GiB；按输出 token 是 1.659/1.320 GiB。即使传输足够快，
冻结 FULL 也没有办法从当前 route 安全触发这些 load。

## 6. Sequential LRU / Belady oracle

下面仍是 `GiB / generated output token`。`LRU / Belady` 都假设处理当前
token-row 前已经知道其 Top-10；Belady 还知道未来 route。

### Decode

| Workload | Q32 LRU / Belady | Q64 LRU / Belady | Q96 LRU / Belady | Q128 LRU / Belady |
|---|---:|---:|---:|---:|
| W1 DEV self | 1.0380 / 0.7702 | 0.6157 / 0.4761 | 0.3656 / 0.3021 | 0.1837 / 0.1585 |
| W2 | 0.6353 / 0.3874 | 0.2621 / 0.1638 | 0.1507 / 0.1112 | 0.1001 / 0.0825 |
| W4 | 0.6958 / 0.4901 | 0.4186 / 0.3124 | 0.2704 / 0.2234 | 0.1785 / 0.1545 |

W2 Q32 Belady 的 0.3874 GiB/output token 说明 route locality 本身没有把
Decode-only pool 判死。在理想 32/64 GiB/s payload 带宽下，它分别对应约
12.1/6.1 ms/token 的纯传输下界，尚未计格式和 publish。问题是该数字不可由
当前 Host planner 在 FULL replay 中实现，不能拿它当预期真实性能。

### Prefill

| Workload | Q32 LRU / Belady | Q64 LRU / Belady | Q96 LRU / Belady | Q128 LRU / Belady |
|---|---:|---:|---:|---:|
| W1 DEV self | 10.4771 / 6.9660 | 6.1741 / 3.6100 | 4.1382 / 2.1000 | 2.5217 / 1.1581 |
| W2（与 W1 prompt 相同） | 1.3096 / 0.8708 | 0.7718 / 0.4513 | 0.5173 / 0.2625 | 0.3152 / 0.1448 |
| W4 | 4.2186 / 2.8244 | 2.7057 / 1.8093 | 2.0622 / 1.2866 | 1.6345 / 0.9988 |

这些 Prefill oracle 把一个 Native invocation 的多 token route 顺序化，不能让
超过 Q 的 invocation 在原 fused call 中自动变得可运行。

## 7. 当前 FULL graph 下的策略可运行性

| Policy / 证据 | 是否可运行 | 原因 |
|---|---|---|
| Working-set fit | 否 | 只证明容量足够，不提供 current experts 或 load/publish |
| W1 static-frequency | 否（完整模型语义） | 可预装，但所有阶段仍有 miss；忽略 miss 会丢专家贡献 |
| LRU | 否 | 必须先看到当前 token route；冻结 FULL 不回到 Python/Host planner |
| Belady | 否 | 还需要未来 route，只是离线下界 |
| Captured exact-trace schedule | 否（资格） | 只能重放已知请求，不能服务任意输入 |
| Device-side planner | 尚不存在 | 需要 current-token miss detection、backing transfer、原子 weight→map publish 和 graph 依赖 |

因此“容量可放下 Top-10”和“Native FULL 可正确运行 pool”之间仍缺核心机制。
当前 source review 找到 graph-external mutation seam，只说明有受控组件 probe 的
价值，不等于已经有 current-token planner。

## 8. 与固定 slot 格式门的交叉证据

独立的 v018 真实设备组件探针发现 Physical-Q32 owner 的 `w13/w2` 都是 NPU
format 29（NZ29）。现有受限 writer 只资格化 format 2（ND），因此在 mutation
开始前 fail-closed：

- 没有修改 parent；
- 没有 publish generation；
- 没有用整 parent rebuild 绕过；
- allocator 回到相同 baseline。

这不是 route 经济性结论，但它进一步说明当前缺少完整实现所需的 format-aware
fixed-slot writer。即使未来补出 device planner，也必须先解决这个独立门。

## 9. 最终裁决

当前不存在值得进入完整 Native expert pool runtime 的“容量＋规划”组合：

- Q64/Q96/Q128 被 frozen whole-card cap 排除；
- Q32 是唯一容量候选，Decode invocation 全部可容纳，但 static map 不完整；
- Q32 LRU/Belady 显示理论流量可能有经济性，却无法在当前 FULL graph 下执行；
- Prefill 对 Q32/Q64 的完整 invocation fit 很差，必须保留独立 fallback；
- 当前还没有 qualified device-side planner，也没有 NZ29 subslot writer。

所以这里的准确结论是：**working-set 本身没有否定 Decode-only expert pool，
但冻结容量和 FULL graph 合同下没有可部署 policy，完整实现不应启动。**

只有一个新的、有界机制同时证明以下两项，才有理由重新打开：

1. device-side current-token miss planning，不做每 token Host route readback；
2. format-aware fixed-slot 写入与 weight-then-map 原子 publish，不重建 parent。

这会是新的运行时机制，不是对当前 static/LRU/Belady 分析的简单验证。
