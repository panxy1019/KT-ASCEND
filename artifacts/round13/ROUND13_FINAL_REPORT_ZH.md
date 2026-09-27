# Round13 最终报告：静态驻留、QSA Decode 与 Deferred 去留

## 1. 最终状态

Round13 的终态为 **`ROUND13_PARTIAL_WITH_GAPS`**。本轮取得了可晋级的真实请求收益：B 线 QSA Decode 候选通过组件、smoke、请求 ABBA 和最终 F 镜像，形成 `HFINAL_ROUND13_LIMITED`。总状态仍保留缺口，因为 A 线冻结的八进程镜像没有完成第二半；不能用“所有候选首块已失败”改写必需覆盖合同。

可单独成立的子结论是 **`ROUND13_REQUEST_GAIN_QUALIFIED`**：相对本轮 fresh S0，FINAL 在两个独立块的 W2 TPOT 分别改善 `32.23%`、`32.10%`，W2 E2E 改善 `29.81%`、`29.30%`，串行 output tokens/s 提高 `41.65%`、`41.75%`。

## 2. A 线：四种同预算静态放置

四种策略都固定 1,536 个 routed experts、名义 BF16 14.0625 GiB；Frequency 使用独立 DEV 全局 Top1536，Front-loading 为前三层各 512，Random 使用 seed42。q=0、q=512 和不均匀层容量、CPU/NPU 空分支、四 NUMA、真实 MXFP4/BF16 reference、取消和回收组件合同均通过。

| 策略 | fresh 首块 W2 TPOT | 相对 Uniform | W2 门 | 其他观察 |
|---|---:|---:|---|---|
| Uniform Q32 | 0.424992 s | 基线 | 保留 | 所有层 Q32 |
| Frequency global | 0.412192 s | 改善 3.01% | 未达到 5% | W0 E2E +19.35%，W3 E2E +12.56% |
| Front-loading | 0.421696 s | 改善 0.78% | 未达到 5% | W0 E2E +22.90%，W4 TTFT +9.67% |
| Random seed42 | 0.411771 s | 改善 3.11% | 未达到 5% | W0 E2E +23.72%，W1 E2E +8.17% |

因此当前更合适的是 **Uniform Q32**。这里的“更合适”来自候选未过必要首块门和保护指标，不是完整八进程统计胜出。A 的四策略各有一个同源成功块，第二镜像缺失，状态保持 `A_NO_PROMOTION_WITH_MIRROR_GAPS`。

## 3. B 线：新 QSA 后仍值得优化的模块

P0 的两个 fresh S0 块显示，QSA Decode 可归因占比峰值为 66.02% / 64.47%，GDN Decode 为 15.91% / 16.00%；PLE 暴露很小。源码复核发现原 ROW Decode 对已经驻留在 device 的 BF16 K/V row 仍执行 CPU round-trip staging。

唯一候选 `DIRECT_BF16_ROW` 只删除该冗余 staging：保留 vectorized Prefill、sparse-GQA 方程、causal selection、BF16 输入与 FP32 attention math，以及不合格 shape 的原 staged fallback。真实权重组件 v233/v234 的最差完整调用 candidate/baseline 比值为 0.197938 / 0.188911；数值、repeat hash、causal/GQA、fallback、取消、late completion、lease 与 allocator baseline 全部通过。

请求 ABBA v219–v222 的 W2 TPOT 比值为 0.663446 / 0.683139，没有两块一致的保护门退化。最终 F v227–v230 再次确认：

| Workload | E2E 改善 A / B | TTFT 变化 A / B | TPOT 改善 A / B | 串行吞吐提高 A / B |
|---|---:|---:|---:|---:|
| W0 64/2 | -8.70% / 2.34% | -10.23% / 0.36% | 3.61% / 17.57% | -8.73% / -2.29% |
| W1 128/16 | 4.67% / 15.38% | -12.19% / -0.46% | 19.59% / 28.37% | 10.40% / 18.37% |
| W2 128/128 | 29.81% / 29.30% | 4.60% / -0.15% | 32.23% / 32.10% | 41.65% / 41.75% |
| W3 512/16（H513 实验） | 21.52% / 18.01% | 3.56% / -1.23% | 58.18% / 56.95% | 27.46% / 21.55% |
| W4 64/32 | 14.80% / 15.59% | 2.86% / 1.21% | 17.21% / 19.11% | 17.54% / 18.63% |

负号表示该单块变慢。W0 在 A 块回退、W1 TTFT 在 A 块回退，但对应 B 块没有同方向超过 5%，所以不构成冻结的“两块一致”保护门失败。FINAL 的主要收益集中在有实际 Decode 长度的 W1/W2/W3/W4。

## 4. C 线：Deferred

C 没有进入设备研究。两个 S0 块中 CPU Expert exposed wait / phase wall time 的上限为 6.7043% 和 6.4842%，均低于冻结的 10% 入口。没有构建或运行 d=1，没有删除同步，没有新增 buffer/event/epoch/cancel-drain 合同，也没有改变后续层输入。因此本轮结论是 `C_DEFERRED_WITH_REASON`，不是“deferred 永久无效”，也没有 deferred 的质量或速度数据。

## 5. 质量与资源

F 的四个独立模型进程共完成 96 次 generation attempt 和 2 次 cancel/reuse control；五类 workload 的实际输出 token 预算、有限 logprob、QSA Prefill/Decode 命中、PLE ASYNC 与状态连续性均通过。FINAL 的 owned HBM 在四臂中均为 25,794 MiB。Host cgroup 峰值为 S0-A 334.073 GiB、FINAL-A 337.239 GiB、FINAL-B 333.800 GiB、S0-B 334.583 GiB；没有一致的 paired Host 回退。whole-card HBM 的 28,811 / 28,811 / 29,694 / 28,811 MiB 包含非 owned 波动，owned 峰值保持不变。

v230 完成后，guard ACK、容器移除和物理12无进程均核验通过。缓存 hook 在全机 fd/mmap 无引用、归档和完整 SHA 后，仅对 179 个登记只读文件执行 DONTNEED；resident pages 从 69,340,089 降为 0，没有使用全局 drop_caches。

## 6. 相对 Round12a 的 W0 背景

Round12a 的 64×2 样本在本轮命名为 W0。历史 KT v174/v177 E2E 中位数为 2.982179 / 3.386336 秒；OffloaderV1 为 1204.699036 / 1349.630513 秒；Native TP8 old-QSA 四块为 3.428953 / 3.371347 / 3.489021 / 3.325310 秒。本轮 fresh S0 v227/v230 为 2.994295 / 2.899652 秒，FINAL v228/v229 为 3.254704 / 2.831800 秒。

这些跨轮数只作描述。OffloaderV1、Native TP8 与本轮 FINAL 在设备数、QSA 或 backend 上不同；Round13 的新加速声明只使用 fresh v227–v230 配对数据。

## 7. 预算与边界

本轮最终记账为 21/28 个模型进程、328/600 次 generation attempt、9/11 个 abort control、9/20 个组件进程，自动无修改重试为 0。没有 push、merge、tag、reset、clean，没有恢复历史定时任务，也没有修改外部训练任务。

`HFINAL_ROUND13_LIMITED` 保留 Uniform Q32、H129、C128/MXFP4、Static BF16/NZ29、chunk32、PLE ASYNC、Dynamic OFF，并把 QSA Decode 从 ROW staged path 改为 `DIRECT_BF16_ROW`。W3 仍是 H513 实验 profile。当前结论不扩展到并发、p99、功耗、部署、TP、HiCache、FP8 KV、Graph 或 MTP。

## 8. 证据入口

- `A_RESULT.json`：静态放置测量与覆盖缺口。
- `PROMOTED_QSA_DECODE.patch`、`PROMOTED_QSA_DECODE_PATCH_MANIFEST.json`：晋级行为的精确源码差异和双端 hash。
- `B_COMPONENT_GATE.json`、`B_REQUEST_RESULT.json`：组件与请求 ABBA。
- `F_RESULT.json`、`F_BENCHMARK.csv`：最终四进程镜像。
- `BENCHMARK.csv`：A/B/C/FINAL 和 Round12a W0 描述性背景的统一表。
- `FINAL_CONFIG.json`、`FINAL_STATUS.json`：最终配置与终态。
- `ROUND13_RESOURCE_RETIREMENT_RECEIPT.json`：最终资源、缓存和 SHA 回执。
