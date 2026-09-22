# A/B 主线证据覆盖矩阵

日期：2026-09-22  
性质：只读证据审计；未启动设备、SSH、测试作业或自动化。本文是文档 09 的路线账本，不是全功能实现或新增资格。

## 判定口径

| 标记 | 含义 |
|---|---|
| `CURRENT_PARTIAL` | `round9-kt-mainline-20260922` 在选定配置上实际执行，但只覆盖条目的一部分。 |
| `CURRENT_SOURCE` | 当前冻结源码/绑定已定位；没有足以完成该条验收的执行证据。 |
| `HISTORICAL_EVIDENCE_NOT_REVALIDATED` | 旧 snapshot 有执行证据，但相关 core source/profile 等价性没有在当前 snapshot 上完整证明；只记录历史范围，不能继承为当前资格或改写成 v63 重跑。 |
| `NOT_EXERCISED` | 本轮未执行，且现成证据不能完成所需验收。 |
| `UNAVAILABLE` | 所需事实在本地交付中没有提供。 |

本轮优先配置是：模型 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`；C128；CPU MXFP4；NPU 原始 BF16/NZ29；每层 Q32；Static Uniform（实际逻辑 ID `0..31`）；Layerwise Prefill；serial streaming32；PLE 显式 `ASYNC`。`ASYNC` 来自 profile `migration_bridge/round9/runtime/profiles/framework_streaming_q32_v1.json`（SHA256 `d8b4661fc7b5b41969a7815dc61f1f764184faf44ace98603854649434315251`），不改变全局默认 OFF。streaming profile `streaming_bounded32_v1.json` 的 SHA256 为 `38cc8b20c2359f4b05c7c23b1a1a8ddff8adc5a78cefa55451b9ec8157dd09b9`。

当前 Round9 主线基线 commit 是 `57de93dfbb5775f1265378c1eae737edb8d5cc3e`；冻结 SGLang commit 是 `f02f304a25364d1e8fe9c52515a22f20f5089a52`。`8f68fb1157353c0e95c2e52ad93c8a2136d1d0a7dce8867324a3d17f0d6f2099` 是继承的 v61 baseline `source_runtime_tools_tests` tree digest；v63 observation build 的 tree digest 是 `12cf0c95651ea8b5b75d599b1d440cf62e74d3fa6b6df572d775003ce2aa2094`。v20 replay 使用后续、不同的 replay snapshot；其逐文件 manifest 是 `artifacts/round9-kt-mainline/evidence/replay-source-sha256.json`，该 manifest 文件 SHA256 为 `756b883be945a2e3a63595c9c3f1c9387822ff43b48c39058a196638eee836b0`。三者不得写成同一个 source tree。native extension SHA256 是 `614ae60f70ff70d1e47d7fd32e857f887f874b6248eb332172c8a637cb471a6d`。模型身份采用可信全量 receipt 加当前 144 文件大小/元数据检查：manifest `62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e`，receipt `7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239`；本轮没有重新 hash 完整 checkpoint。config、tokenizer config、checkpoint index 的本地 SHA256 分别是 `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b`、`b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27`、`99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`。

## A 主线

| ID | 状态 | 已有证据与精确边界 | 仍缺的验收 |
|---|---|---|---|
| A01 CPU Expert 真执行 | `CURRENT_PARTIAL` | `artifacts/round9-kt-mainline/evidence/v20-real-expert-replay.json`，SHA256 `58269cf28abb999b705d14fdae1b8643cf821b406d69708041ec32481ea98f5c`。真实 layer0 decode row，C128/128 workers/NUMA 2,3,6,7，MXFP4 bank `e9d6b27f352d501a9ed3737ec5d3ab140b9d1f5bda22d2e36ef9ef0bcc45b08b`，CPU partial 相对 L2 `1.17252872e-4`，重复三次输出 hash 一致。当前时间线还观察到 `_submit_cpu_forward`/`_sync_cpu_forward`。 | native queue、通用 activation packing/权重量化解码、GEMM、合并的互斥分解未导出；没有证据把本条 MXFP4 运行的内部 activation packing 特指为 Q8。Python submit 不是 worker compute。 |
| A02 混合路由正确性 | `CURRENT_PARTIAL` | 同一 v20 fixture：route `[20,150,14,111,203,414,211,358,33,230]`；Q32 map 下 20/14 走 BF16 NPU、其余 8 项走 MXFP4 CPU；mixed oracle merge 相对 L2 `3.05562251e-4`，与完整模型捕获输出 exact。 | 仅一层、一 token、一条路由；空 CPU/NPU 分支及多路由形状未覆盖。 |
| A03 Prefill microbatch | `CURRENT_PARTIAL` | v63 在 Layerwise/streaming32 下实际完成 5-token 和 64-token prefill，并记录 shadow prepare；`artifacts/round9-kt-mainline/evidence/v63-normal-summary.json` SHA256 `9dfcc8dcd51cc4ffbc123f8592e0e2c5ae41653b7f99eab9ccaa561842a8dce2`。历史 v43/v44 覆盖 `(0,32)`、`(32,5)` chunk。 | 没有一个以 prefill microbatch 算术/reference 为目标的当前独立回放；scheduler chunk32、CPU workspace ≤32 和 shadow lifetime 不得混成一个粒度。 |
| A04 Static identity | `CURRENT_PARTIAL` | `artifacts/round9-kt-mainline/evidence/v63-q32-physical.json`，SHA256 `fdc73639b5d3b0f97db2e10c689b4b3a802240d72e50d3bc00aeee402d9f1872`：48 层均为 Q32、逻辑 ID `0..31`、每层 314,572,800 bytes，总 15,099,494,400 bytes。v20 replay 使用同一实际 map；不是旧 interval16 fixture。 | 当前 Uniform 规则可复现，但没有为每层 map 单独列 digest；也不是 Frequency 价值证明。 |
| A05 Static 价值 | `NOT_EXERCISED` | 历史 Frequency smoke 在 `migration_bridge/round9/evidence/physical15/20260920-full-model-v27/framework-summary.json`，SHA256 `feef9df78d9fb62e683de505cc6fc7b1344f0f0dd0ee05690dae15eb2a5c96c6`，仅能证明该旧配置执行。 | 缺同一质量、预算、精度、workload 下的 Static Uniform/Frequency 公平成本对照；不能声称 Frequency 更快或全局最优。 |
| A06 Dynamic 正确性 | `HISTORICAL_EVIDENCE_NOT_REVALIDATED` | 旧 implementation commit `53902048d35ad4cc6e7fda2ef41706552a03bb88`、模型 revision 相同。v34 `dynamic-events.jsonl` SHA256 `b749df27329aebf486785867afab8ddcfa8b76d58eeabf24ef531131c383d9d4` 记录 Prefill-End commit/final drain；v30 `framework-summary.json` SHA256 `5605b9863532ccc79c4f94cbe2a5f62168b946127c7b8520530142074728fe9a` 记录 Q32 request-boundary smoke。 | 当前 v63 是 placement epoch 0 的 Static Uniform，没有迁移；旧 source tree 与当前观测 snapshot 不同，相关 core source/profile 等价性未证明，不能继承为当前资格。failure/cancel 的完整事务组合仍只在旧限定证据内。 |
| A07 Dynamic 收益 | `NOT_EXERCISED` | 没有本地证据给出迁移成本与后续 decode 节省的同请求净值。 | 需要固定请求、算术/质量与资源，分别记录迁移、同步、后续节省和质量；不得借官方数字。 |
| A08 Hybrid Prefill | `HISTORICAL_EVIDENCE_NOT_REVALIDATED` | 旧 v27/v30/v35/v36/v39 是 Hybrid smoke；矩阵来源 `migration_bridge/round9/60_FEATURE_INTERACTION_MATRIX.md` SHA256 `a1a068b3aa0414219a9cbc2411b0855c79e0d8848315f345d3b1bc72c73cf4ce`。当前 decode 是混合 CPU/NPU，但当前 prefill 选定 Layerwise。 | 当前主配置没有 Hybrid Prefill 同语义 reference；旧 snapshot 没有经当前 core source/profile 等价性复核；混合 decode 不能代替。 |
| A09 Layerwise Prefill | `CURRENT_PARTIAL` | v63 真实 provider：5-token 请求 65 次 prepare、Host-observed 累计 60.2026s、15,532,032,000 accounted bytes；64-token 请求 274 次、264.2664s、75,959,500,800 bytes。source identity、logical expert list、BF16、NZ29、allocation epoch 均写入 timeline。 | prepare 仍把 canonical BF16 materialization、ND/NZ parent、copy/wait/retire 合在一起；跨层/同层复用关系未证明。下一步只允许做已选 shadow preparation decomposition。 |
| A10 Hybrid/Layerwise 选择 | `NOT_EXERCISED` | 旧 bounded interaction matrix 证明多个组合可运行；没有公平策略比较。 | 固定 workload、精度、Q32 map、bank 和 batch 后比较已支持的两种模式；不能同时 sweep map、bank、batch。 |

## B 主线

| ID | 状态 | 已有证据与精确边界 | 仍缺的验收 |
|---|---|---|---|
| B01 模型身份 | `CURRENT_PARTIAL` | revision/receipt/config/tokenizer/index 身份见上；`Qwen4ExpForConditionalGeneration`，48 层，36 GDN + 12 QSA，512 experts/top10，PLE config `[2]` 对应 runtime layer1。实际入口和 provider 见 `COMPONENT_AND_KERNEL_MAP.md` SHA256 `b2e34e6b4c1a449431540727233537e292b023907ef2a0cda046fcb1ee936605`。 | tokenizer 完整文件集合未在本轮逐文件重 hash；完整 checkpoint 未重 hash，只能保留 receipt+live metadata 资格措辞。 |
| B02 GDN Prefill | `CURRENT_PARTIAL` | 当前 provider `Qwen4ReferenceGatedDeltaNet.forward -> round9.gdn.eager_gdn/gated_delta_recurrence`；v63 sampled layer0 prefill 有真实调用。历史 canonical 合同在 `migration_bridge/round9/22_GDN_CANONICAL_AUDIT.md`，但其来源审计与本轮 snapshot 不同。 | 当前运行没有独立同语义 prefill/chunk reference；36 层没有全量 provider timing，合法 chunk continuity 只可继承旧限定合同。 |
| B03 GDN Decode | `CURRENT_PARTIAL` | v63 decode capture 在 sampled layer0 观察到连续 GDN 调用，state 由请求本地 pool 持有；取消后同进程复用成功。 | 没有当前逐步 state tensor 对 oracle、epoch/reuse exact 检查；复用成功不能单独证明无串状态。 |
| B04 QSA/indexer | `CURRENT_PARTIAL` | 当前 layer3 provider `Qwen4ReferenceIndexer.forward` 与 `Qwen4ExpAttentionDecoderLayer.self_attention` 在 prefill/decode 均被观察；indexer 是 attention parent 的 child，不能相加。历史 v41 真实 4K pool/reference consumer SHA256 `8db1a55a0df2d56678716331ff0f76ea981fa8cade4dc6b8abfbd7fb61d6ff3c`。 | 当前短输入未触发 >512 compressed-block pressure；该 sparse branch 为 `NOT_EXERCISED`。当前没有独立 selection/KV/attention exact reference。 |
| B05 PLE | `CURRENT_PARTIAL` | 当前 v63 profile 显式 ASYNC；时间线观察 gather、H2D dependency、consumer 和 retire，2-slot ring；`ple_layer_ids=[2]` 映射 runtime layer1。v35/v36 旧 ON/OFF smokes 的 SHA256 分别 `934c2b2621a73f107681a2fe18f4f0e42bf7de6b46dfdfc3bf693e5805cf8122`、`97fe3129ea8eebf385974d08c84f32d55a8c15857abc8ffa181763b40f4f2d99`。 | 当前没有 exact row/history 内容导出；PLE wait 极小但不能据此证明 overlap 或调度收益。全局默认仍 OFF。 |
| B06 GR/Router/Shared Expert | `CURRENT_PARTIAL` | 当前 sampled layer0/3 观察 GR `mix/combine`；真实 decode capture 给出 router top10 和权重。历史 shared expert/router/final mixer component 证据在 `migration_bridge/round9/30_SHARED_EXPERT_FINAL_MIXER_GATE.md`。 | 当前没有 shared expert 独立 span/reference，也没有 router/GR 全层成本；不得把未单列部分吞成“Other 已通过”。 |
| B07 Ascend correctness | `CURRENT_PARTIAL` | v62 observer OFF 与 v63 observer ON 的 base/chunk token IDs 一致；v20 保存真实 mixed fixture，并对 CPU、NPU、merge 使用不同 reference。 | OFF/ON 是回归一致性，不是已知答案质量；一条真实 Expert row 不是 GDN/QSA/PLE 模块全覆盖。产品质量仍 `PENDING`。 |
| B08 成本归因 | `CURRENT_PARTIAL` | v63 有 242 对 NPU event durations、190 sampled decode spans、无 drop/error；父子关系、clock domain、request/phase 均记录。prepare 是当前最强线索。 | 只采样 layer0/3；CPU worker、DRAM、绝对跨流校准未知，首次 decode parent 834.198ms 未解释。状态保持 `PARTIAL_ATTRIBUTION`。 |
| B09 kernel 必要性 | `CURRENT_SOURCE` | GDN/QSA/GR 是 eager reference；NPU Expert source dispatch 为 routing_v2 → grouped_matmul → swiglu → grouped_matmul → finalize。source map SHA256 `1a5d88da05a3cb79ccac75d541dab64809fca8f3601fbc5c524708b9f952ca30`。 | 没有精确热点 shape/dtype/layout 的 mutually-exclusive device ranking、fallback 原因或可消除成本；eager 身份不足以批准新 kernel。 |

## 继承规则与路线结论

历史 Round9 framework snapshot 的模型 revision 与当前相同，但 implementation commit、source tree、资源模式和具体 feature profile 不同。模型 revision 相同本身不足以证明 core source 等价。历史证据只能作为“该精确旧组合曾执行”的账本；除非当前 source/hash、profile、map、bank、phase 和 reference 均已证明相符，否则状态保持 `HISTORICAL_EVIDENCE_NOT_REVALIDATED`，不能升级为 current-profile PASS。当前 profile 中 Static、Layerwise、streaming32 和显式 PLE ASYNC 可以同时陈述；Dynamic 与 Hybrid Prefill 只能列为历史范围或未执行。

最近可执行路线仍是 `EXPERT_SHADOW_PREPARATION_COST_DECOMPOSITION`：在同一 layer0 fixture、Q32 map、MXFP4/BF16 source、NZ29 layout 和 32-slot reservation 下拆分 materialization、parent/layout preparation、copy/completion 和 retire。任何 reuse、changed-slot、额外 buffer 或 kernel 改动都需要分解结果后的独立立项。A05、A07、A10 和 B09 仍不足以批准性能结论或行为变更。

质量 canary 另见 `CANARY_CONTRACT.md`。现有 `max_new_tokens=1` 返回 token 271 / raw text `"\n\n"` 只证明 OFF/ON 路径一致，不能标成质量 PASS。
