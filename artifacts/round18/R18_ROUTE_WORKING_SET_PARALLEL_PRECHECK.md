# Round18 Phase II：历史 Route 工作集与 Native Expert Pool 经济性预判

状态：`HISTORICAL_PRECHECK_ONLY_NOT_ROUND18_QUALIFICATION`

本预检没有启动 NPU、模型或远端任务。仓库中没有找到 fresh Round18 Native W1/W2/W4 route trace；唯一可完整还原 Top-10 的证据是 Round9 v80 的 SGLang Layerwise host-only 诊断带。因此本文只能回答“这条历史 64-token Prefill route 对 expert pool 有什么敏感性”，不能回答 Round18 Native Decode 的正式经济性，也不能据此晋级 Phase II。

## 1. 数据身份和质量

来源：`/home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round9-fixed-shadow32-update-20260925/v80/real-prefill.routes.jsonl`

SHA-256：`201a9b65b24c2fd9774c8c2dbfc4f795078d12dc8f20ce4b2a82efee12d857c0`

核验结果：

- 单一请求 `r9-plan15-p4a-on-medium`，输入 64 token；
- 两个 32-token Prefill chunk，48 层完整覆盖；
- 724 个历史 Layerwise group，sequence `0..723`、allocation epoch `1..724` 连续；
- 3,072 个 token-layer event，30,720 个 Top-10 occurrence；
- 每个 token-layer 恰有 10 个互异专家，ID 均在 `0..511`；
- header 明确写有 `performance_evidence=false`、`diagnostic_host_observer=true`；
- 文件没有单独 terminal footer，但层、token、sequence 和 epoch 覆盖内部完整。

旧的 `R18_ROUTE_WORKING_SET_HISTORICAL_PRECHECK.json` 使用 32 作为 bytes/token 分母。实际请求包含两个 chunk、共 64 token；本文已改用 64。旧文件的 hit/miss 计数本身不受影响，但 bytes/token 高估了 2 倍。

## 2. 模拟口径

每层维护独立 pool，每个专家的 BF16 gate/up/down 名义 payload 为 9,830,400 bytes（9.375 MiB）。对同一个 token-layer 的 Top-10，模拟器在处理该 event 时禁止把当前请求集合中的专家作为 victim。

- Static-frequency：使用同一条 trace 的 per-layer Top-Q，是带未来信息的 oracle。miss 计为需要加载的专家，但固定集合不变，因此隐含额外 scratch 或恢复成本，只能作命中率预测。
- LRU：给出 cold 和“同一 trace frequency warm-start”两种结果；后者同样有信息泄漏。
- Belady：离线 farthest-next-use 上界，也给出 cold 和同一 trace warm-start。warm-start 是最乐观的传输下界，不能当作可部署策略。

传输量只计算 `miss × 9.375 MiB`，不含 padding、NZ/物理格式、metadata、publish event、同步和 Graph break。

## 3. 聚合结果

下表中的 miss/token 是 48 层合计，因此理论最大 occurrence 为 `48 × 10 = 480/token`。

| Pool | 48层名义payload | Static-frequency hit / miss·token⁻¹ / GiB·token⁻¹ | LRU cold hit / miss·token⁻¹ / GiB·token⁻¹ | Belady cold hit / miss·token⁻¹ / GiB·token⁻¹ | Belady warm上界 hit / miss·token⁻¹ / GiB·token⁻¹ |
|---|---:|---:|---:|---:|---:|
| Q32 | 14.0625 GiB | 55.32% / 214.48 / 1.964 | 50.40% / 238.09 / 2.180 | 65.08% / 167.59 / 1.534 | 68.40% / 151.67 / 1.389 |
| Q64 | 28.125 GiB | 75.69% / 116.70 / 1.068 | 66.02% / 163.11 / 1.493 | 73.16% / 128.83 / 1.179 | 81.48% / 88.88 / 0.814 |
| Q96 | 42.1875 GiB | 87.43% / 60.34 / 0.552 | 71.00% / 139.22 / 1.275 | 74.19% / 123.89 / 1.134 | 88.00% / 57.61 / 0.527 |
| Q128 | 56.25 GiB | 94.07% / 28.48 / 0.261 | 72.78% / 130.66 / 1.196 | 74.19% / 123.89 / 1.134 | 93.61% / 30.69 / 0.281 |

Static-frequency 在 Q128 略优于动态 Belady warm，不表示 Static 比 Belady 更聪明。Static 模型保持固定 Q 项并把 miss 当作一次临时加载，没有把临时专家放回同一个 Q-slot pool；若没有额外 scratch，真实实现需要覆盖、恢复或改变固定集合，这些成本没有被计入。

若只看最乐观的 Belady warm，并假设有效 Host→NPU 带宽为 16/32/64 GiB/s，纯 payload 传输下界分别约为：

| Pool | 16 GiB/s | 32 GiB/s | 64 GiB/s |
|---|---:|---:|---:|
| Q32 | 86.8 ms/token | 43.4 ms/token | 21.7 ms/token |
| Q64 | 50.9 ms/token | 25.4 ms/token | 12.7 ms/token |
| Q96 | 33.0 ms/token | 16.5 ms/token | 8.2 ms/token |
| Q128 | 17.6 ms/token | 8.8 ms/token | 4.4 ms/token |

这些只是 transfer-only 下界；实际 miss 还要计格式、publish、同步和图兼容成本。

## 4. 层间偏斜

64-token 历史请求中，每层出现的不同专家数为 119–223，中位 161。层 22、0、38 在所有容量下持续属于低命中层；层 15、16、19 通常最集中。

Static-frequency 的 per-layer hit 范围：

| Pool | min | median | max |
|---|---:|---:|---:|
| Q32 | 33.28% | 55.39% | 72.50% |
| Q64 | 55.00% | 76.72% | 90.78% |
| Q96 | 71.41% | 88.52% | 96.41% |
| Q128 | 82.50% | 94.84% | 100.00% |

这说明统一 Q 对不同层并不等价；但本轮计划冻结 per-layer fixed capacity，本文不据此开启新的 placement 搜索。

## 5. Native Prefill 的关键限制

逐 token 的缓存模拟对 Decode 有直觉意义，但源 trace 实际来自 32-token Prefill chunk。按 Native 单次 layer invocation 计算每个 chunk 的专家并集：

- 96 个 layer×chunk invocation 的 unique expert 数为 77–183，中位 111；
- Q32 和 Q64：`0/96` invocation 可以一次容纳完整 working set；
- Q96：`15/96`；
- Q128：`76/96`。

所以，Q32/Q64 无法在不 staging/splitting 的情况下直接保持这条历史 Prefill 的完整 Native routed-expert invocation。把 32-token batch 顺序化为 32 个单 token 会改变 launch/Graph 组织，不能用本文结果声称 Native Prefill 已兼容。

## 6. HBM 预判

冻结 whole-card cap 为 56,832 MiB（55.5 GiB，已包含 8,704 MiB reserve）。仅 expert pool 的名义 payload：Q32 14.0625 GiB、Q64 28.125 GiB、Q96 42.1875 GiB、Q128 56.25 GiB。

- Q128 的 expert payload 单独就超过 whole-card cap，直接不合格；
- Q96 只给 dense/shared weights、QSA/GDN/PLE state、KV、workspace 和 graph pool 留 13.3125 GiB，没有 fresh 资源证据支持；
- Q64 比 Q32 多 14.0625 GiB，只有 fresh HBM/KV/graph-pool sizing 后才值得做容量 probe；
- 历史 vLLM-KT Q32 在 `artifacts/round16/B_RUNTIME_RESULT.json` 中记录 warmup 后 torch allocated/reserved 约 47.3/47.34 GiB，但其 KV 配置和 Round18 pool runtime 不同，不能直接拿来判 Q64 OOM 或 PASS。

## 7. 结论

本历史预检不支持直接开发完整 Native expert pool：

1. Q32/Q64 即使在带未来信息的 oracle 下仍有 0.81–1.39 GiB/token 的 BF16 transfer 下界；
2. Q32/Q64 对历史 32-token Prefill 的完整层 working set 均为 `0/96` 可容纳；
3. Q96/Q128 的命中率更好，但尚未取得容量资格，且 Q128 已被 HBM cap 排除；
4. 历史 Prefill 路由不能代表 W2 Decode 的时间局部性。

Phase II 继续之前必须取得：fresh Round18 Native W1/W2/W4 路由（Prefill/Decode 分开）、独立 DEV/HOLDOUT 的 frequency plan、fresh HBM headroom、固定 slot 写入成本，以及不依赖每 token Host route readback 的 Graph-safe residency probe。

## 8. 复现

分析器和单元测试已经存在：

```bash
cd /home/admin/Documents/ChatGPT/KT/kt-round18-native-preserving
python3 -m unittest tests.round18.test_expert_pool_analysis
```

最小聚合复现：

```bash
cd /home/admin/Documents/ChatGPT/KT/kt-round18-native-preserving
PYTHONPATH=. python3 -c 'from round18.expert_pool_analysis import load_round9_route_tape,frequency_resident_plan,simulate_pool; p="/home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round9-fixed-shadow32-update-20260925/v80/real-prefill.routes.jsonl"; e=load_round9_route_tape(p); print([(q, simulate_pool(e, capacity=q, policy="lru").miss_loads, simulate_pool(e, capacity=q, policy="belady", initial_resident=frequency_resident_plan(e,q)).miss_loads) for q in (32,64,96,128)])'
```

机器可读明细见 `R18_ROUTE_WORKING_SET_PARALLEL_PRECHECK.json`。
