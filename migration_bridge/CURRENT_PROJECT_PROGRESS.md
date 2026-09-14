# KTransformers × Kunpeng × Ascend 当前整体进展

> 文档状态：Round 4A 最终冻结后的当前工程总览
> 更新时间：2026-09-12
> 当前分支：`feature/kt-round4a5-p3`
> 最终标签：`round4a-tp1-multiplacement-a3-verified`

## 1. 执行摘要

DeepSeek-V2-Lite 已在 Kunpeng 920 + 单张 Ascend NPU、TP=1、静态
CPU/NPU expert placement 下完成从 P1 到 P3 的完整正确性闭环：

```text
Round 2A CPU expert plane             = A3_VERIFIED_READY
Round 2B Ascend runtime plane         = A3_VERIFIED_READY
Round 2C synthetic Hybrid MoE         = A3_VERIFIED_READY
Round 3 real-model TP1                = A3_VERIFIED_READY
Round 4A pairwise contract            = HELDOUT_VERIFIED
Round 4A P1 (1 layer / 4 experts)     = A3_VERIFIED_READY
Round 4A P2 (4 layers / 16 experts)   = A3_VERIFIED_READY
Round 4A P3 (8 layers / 32 experts)   = A3_VERIFIED_READY
DEEPSEEK_V2_LITE_TP1_MULTI_PLACEMENT = A3_VERIFIED_READY
```

这表示架构 bring-up 和当前 TP1 static placement scale validation 均已完成。
它不代表 TP2、Graph、动态 placement、量化或性能优化已经完成。

## 2. 当前冻结配置

| 项目 | 冻结值 |
|---|---|
| 模型 | DeepSeek-V2-Lite revision `604d5664` |
| Serving | `kvcache-ai/sglang` 的 `sglang-kt` 路径 |
| SGLang child | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |
| CPU backend | LLAMAFILE / CPUInfer |
| NPU backend | Ascend BF16/unquantized |
| 硬件 | Kunpeng 920 + NPU0 only |
| 并行 | TP=1、batch=1、greedy |
| Graph / Deferred / Dynamic | OFF / OFF / OFF |
| MTP / Speculative | OFF / OFF |
| numerical contract | `B_pair=2.1875`, K=32 |
| numerical contract SHA | `223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717` |
| quality manifest SHA | `c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce` |

P2 immutable tag 是 `round4a5-p2-a3-verified`，target 为
`d4830525c6806d5d587b3096634614aad7fef181`。Round 4A 最终 tag 不允许移动。

## 3. 已冻结的系统架构与修复

Hybrid 职责链保持：

```text
SGLang router (global logical expert IDs)
       ├── NPU-owned routes -> Ascend grouped MoE
       └── CPU-owned routes -> KT wrapper -> CPUInfer -> LLAMAFILE
CPU partial + NPU partial -> FP32 merge -> one final BF16 cast
outer DeepSeek layer -> shared expert exactly once
```

Round 4A.1 保留两个 production numerical fix：BF16-visible boundaries 和
FP32 Hybrid partial accumulation。Round 4A.5 进一步定位 P2 nondeterminism：

```text
raw acl.rt.memcpy pointer transfer
-> bypass framework stream/lifetime semantics
-> multi-layer nondeterministic visibility
```

生产路径已在 commit `00c7acfe...` 改为 framework-managed Tensor `copy_`。
P3 全程保留此路径，没有重新引入 raw pointer transfer。

## 4. P1 与 numerical contract

P1 使用 Layer 17 和 CPU experts `{6,8,25,36}`。Round 4A.4 通过全新 Q2、
H2 held-out、free generation 和 downstream quality，冻结 pairwise-order
contract：stable ordering 必须 100% exact；ambiguous token 必须属于 baseline
ambiguity set；ambiguity cardinality 仅作诊断。

P1 requalification、C0-C4、same-path determinism、clean restart、CPU-not-hit、
1,000-forward lifecycle 和质量全部通过。

## 5. P2 关闭结果

P2 固定 Layers `{1,9,17,26}`、16 个 CPU experts。framework transfer 修复后：

- 9 prompts × 10 repeats × 64 tokens 及 8/16/32 prefix exact；
- clean restart 9/9 exact；
- Q2/H2 subset 共 512 positions，stable/membership 100%，overflow 0；
- 4/4 layers、16/16 CPU experts、74,807 CPU route hits；
- CPU-not-hit 32/32 exact；
- C-Eval 27/128、GSM8K MC 31/128，与同环境 All-NPU 完全一致；
- retained regression 与 lifecycle 全部关闭。

因此 `MULTI_LAYER_P2 = A3_VERIFIED_READY`。

## 6. P3 最终结果

P3 固定 Layers `{1,5,8,12,17,19,22,26}`、32 CPU experts，placement SHA：

`77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d`。

P3 GGUF SHA：

`992d32e0de59e135464f77fc4869e810f3d71b3ec01f6ed0996cbeaa58e49df4`。

| Gate | 结果 |
|---|---|
| same-process production overlap | 117 requests，9/9 trajectories/prefixes exact |
| clean restart | 9/9 exact |
| pairwise | Q2/H2 512 positions，0 overflow，stable/membership 100% |
| coverage | 8/8 layers，32/32 experts，161,925 CPU hits |
| CPU-not-hit | 32/32 exact |
| free generation | 9 次 divergence 全为 ambiguous/in-set，stable divergence 0 |
| quality | C-Eval 27/128，GSM8K 31/128；paired delta/CI 均为 0 |
| stability | 两个独立 campaign，共 2,304 tokens，无 crash/deadlock/non-finite/unbounded growth |
| retained regression | R2A/R2B/R2C/R3/SGLang/P1/P2 全通过 |

因此 `MULTI_LAYER_P3 = A3_VERIFIED_READY`。

## 7. 当前能力边界

已完成的是：TP1、单 NPU、静态 placement、LLAMAFILE CPU experts、BF16
unquantized NPU path 的 correctness closure。

以下仍是“未资格验证”，不是本轮失败：

- TP2/TP4/TP8、HCCL、多 NPU、多节点；
- Graph、Deferred Experts、Dynamic Placement；
- MTP、Speculative Decoding；
- W8A8、MXFP4、KML production integration；
- NUMA scaling、并发 serving 和 production throughput；
- 与最新 upstream Ascend 实现的架构合并。

## 8. 下一阶段

下一阶段建议正式命名为 `Round 5 — Upstream Ascend Alignment`。必须从
Round 4A immutable tag 新开分支，比较 local framework-managed transfer 与
官方 callback-worker/synchronous-submit 设计，并建立全新的 P1/P2/P3 evidence
chain。禁止改写当前 Round 4A 历史。

详细证据入口：

- `migration_bridge/round4a5/p3/P3_FINAL_SUMMARY.md`
- `migration_bridge/round4a5/ROUND4A_FINAL_SUMMARY.md`
- `migration_bridge/round4a5/ROUND4A_FINAL_GATE_MATRIX.md`
- `migration_bridge/round4a5/ROUND4A_FINAL_PROVENANCE.md`
- `migration_bridge/round4a5/p3/evidence/SHA256SUMS`
