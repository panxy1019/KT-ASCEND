# Round 6 — DeepSeek-V2-Lite Deferred Expert Ascend

## Codex 正式执行方案（审阅稿）

> 项目：KTransformers × Kunpeng × Ascend Migration  
> 仓库：`panxy1019/KT-ASCEND`  
> 目标：在 DeepSeek-V2-Lite 上完成 `defer=1` Ascend 资格验证  
> 硬件：Kunpeng 920 + NPU0，TP=1，batch=1  
> CPU：LLAMAFILE / CPUInfer；NPU：Ascend BF16 / unquantized  
> 状态：`APPROVED_EXECUTION_IN_PROGRESS`  
> Owner 已审阅批准；实施证据保存在独立 Round 6 feature branch  

---

## 0. Executive mandate

Round 6 只打开一个新的执行语义风险面：

```text
DEFERRED EXPERT EXECUTION ON ASCEND
```

目标链：

```text
Round 5 immutable static Hybrid
        ↓
defer=0 exact reproduction
        ↓
defer=1 mechanism correctness
        ↓
Ascend ordering / pending / buffer lifetime
        ↓
same-path determinism + clean restart
        ↓
matched-history numerical derivation
        ↓
independent held-out validation
        ↓
downstream quality + stability + performance + regression
        ↓
ASCEND_DEFERRED_EXPERT = A3_VERIFIED_READY
```

本轮只正式资格验证：

```text
max_deferred_experts_per_token = 1
```

`defer>=2`、Dynamic Placement、Qwen3.8、TP2、Graph、MTP、Speculative、量化迁移
均不进入本轮。

---

## 1. 关键设计判断

### 1.1 当前代码已有 Deferred 主体

Round 5 aligned parent 已有：

```text
select_deferred_experts
KExpertsCPUBuffer.buffer_depth
_layer_has_pending_deferred
allow_n_pending
incremental merge
NPU device wait
framework-managed tensor copy_
```

SGLang child 已有 `--kt-max-deferred-experts-per-token` 和 per-layer wrapper 构造。
所以策略是先审计和验证，只修证据证明的缺口，不重写 scheduler。

### 1.2 Deferred 不是普通浮点 backend 差异

普通 Hybrid 在 Layer L 内闭合 routed contribution。Deferred 将低 route-score 的
CPU contribution 作为 pending task，并在下一 MoE layer 的 CPU output 中增量合入。
这是有意的跨层近似，不能直接套用 Round 4A “相同模型语义、不同 reduction”结论。

Round 4A pairwise contract 继续保护 `defer=0`；`defer=1` 需要独立的、由 QD 推导并
由 HD held-out 验证的 `DEFERRED_NUMERICAL_ACCEPTANCE_CONTRACT`。

### 1.3 原 P1/P2/P3 placement 不能直接打开 Deferred

当前 SGLang 只给存在 CPU-owned expert 的层创建 KT wrapper，而 Deferred 只检查
紧邻 `layer_idx-1` 的 pending state。原 placement：

```text
P1 = {17}
P2 = {1,9,17,26}
P3 = {1,5,8,12,17,19,22,26}
```

大多不是连续层。孤立 CPU layer 若产生 pending contribution，下一层没有 wrapper
消费，存在 silent drop 风险。

因此：

1. 原 P1/P2/P3 只以 `defer=0` 运行，作为 retained regression；
2. Round 6 新建连续层 D1/D2/D3 qualification placements；
3. 每个连续 CPU-enabled block 的末层强制 `defer=0`；
4. 只有下一层也有 KT wrapper 时，当前层才允许 defer；
5. 非法 topology 必须 fail-fast 或显式 per-layer 降为 0，禁止 silent pending loss。

---

## 2. Immutable source and artifacts

### 2.1 Git

```text
GitHub main merge:
5feb57f91aa9c786d2bc0002585a8958a456eb0f

Round 5 tag:
round5-upstream-aligned-tp1-a3-verified

tag object:
94baf6272dd86eae7e465a04f4ead630032400d4

tag target / aligned runtime:
be5b48621db32fb894402e3e886060dea263cdc9

SGLang child:
412219b4b05a888808f0b88d2963a8b6164a0fda
```

Round 6 从 `origin/main` 创建分支以保留完整 evidence，但必须验证 production tree
和 child 与 immutable runtime tag 一致。

### 2.2 Model/contracts

```text
model = deepseek-ai/DeepSeek-V2-Lite
revision = 604d5664dddd88a0433dbae533b7fe9472482de0
tokenizer SHA256 = 41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669
P3 GGUF SHA256 = 992d32e0de59e135464f77fc4869e810f3d71b3ec01f6ed0996cbeaa58e49df4
P3 placement SHA256 = 77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d
pairwise contract SHA256 = 223e738436659d389d913656952a368b6d32ccaceeef195abc2e6589c651d717
quality manifest SHA256 = c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce
B_pair = 2.1875
candidate K = 32
```

---

## 3. Git/worktree discipline

当前目录属于 dirty Qwen research branch。实施时不得在该目录 reset/clean/switch。

审阅批准后：

```bash
git fetch origin --prune --tags
git worktree add \
  /home/admin/Documents/ChatGPT/KT/kt-round6-deferred \
  -b codex/round6-deferred-expert-ascend \
  origin/main
git -C /home/admin/Documents/ChatGPT/KT/kt-round6-deferred \
  submodule update --init --recursive
```

如 SGLang 需修改，child 分支：

```text
codex/kt-ep-round6-deferred-expert-ascend
```

建议 commits：

```text
docs(round6): freeze Deferred scope and provenance
test(round6): add Deferred selection and lifecycle regressions
fix(round6): enforce valid Deferred topology                 # only if proven
fix(round6): close Ascend Deferred ordering defect           # only if proven
test(round6): add A3 qualification harness
docs(round6): record numerical and quality qualification
docs(round6): freeze Deferred v1 result
```

禁止合入 Qwen、Dynamic、moving upstream 或无关重构。最终只推 feature branch；是否
合并 `main` 由 owner 另行授权。

---

## 4. Scope

### In scope

```text
DeepSeek-V2-Lite；TP1；batch=1；greedy；NPU0
LLAMAFILE/CPUInfer；Ascend BF16 experts
static contiguous placements
production overlap
defer=0 and defer=1
selection/pending/incremental/slot/transfer semantics
determinism、held-out numerical、quality、stability、performance telemetry
```

### Out of scope

```text
Dynamic Placement / online migration
defer>=2
Qwen3.8
TP2/HCCL/multi-NPU
batch>1/concurrent-serving qualification
Graph/MTP/Speculative
W8A8/MXFP4/KML
NUMA or threadpool redesign
performance architecture rewrite
```

---

## 5. Frozen runtime

除 placement 与 defer 参数外沿用 Round 5：

```text
device=npu; NPU0 only; TP=1; max-running-requests=1
dtype=BF16; greedy; seed=0
Graph OFF; Dynamic OFF; MTP OFF; Speculative OFF
attention=ascend; sampling=pytorch
LLAMAFILE; CPUInfer=16; threadpool=1; NUMA node=0
weight-loader mmap OFF
framework-managed D2H/H2D
production overlap ON
sequential control = diagnostic only
```

---

## 6. Evidence states

```text
CODE_INSPECTED
LOCAL_VERIFIED
A3_VERIFIED
A3_VERIFIED_READY
ANALYTIC_DERIVED
HELDOUT_VERIFIED
QUALITY_VERIFIED
BLOCKED_<SPECIFIC_GATE>
```

禁止把 local、单次 smoke、代码存在或 defer=0 evidence 写成 defer=1 A3 ready。

---

## 7. Deferred semantic contract

### D0 Routed partition exact

```text
CPU_ROUTED = IMMEDIATE_CPU ∪ DEFERRED_CPU
IMMEDIATE_CPU ∩ DEFERRED_CPU = ∅
```

logical IDs、route weights、CPU/NPU ownership 不变；NPU-owned route 不能因 defer
变为 CPU work；sentinel ID 不能进入 weight lookup。

### D1 Selection deterministic

必须冻结并验证 equal-score tie、prefill 多 token 是 per-row 还是 batch-union protection、
CPU-owned intersection 后的实际 deferred count。CLI `1` 不等于每 token 必然 defer
一个 CPU expert。

### D2 Topology exact

连续 block `[L0...Ln]`：

```text
L0...L(n-1): defer<=1
Ln: defer=0
```

### D3 Exactly-once merge

每个 deferred contribution 必须 produced、consumed、merged exactly once；不能 drop、
duplicate 或跨 request/token 合入。

### D4 Queue/lifecycle exact

```text
pending high-water <=1
terminal block drains
request completion leaves no unowned task
exception and shutdown drain safely
clean restart retains no state
```

### D5 Buffer ownership exact

slot trace 至少包含 request/generation、token/qlen、producer、consumer、write/read 时段。
consumer 完成前禁止复用。

### D6 Transfer exact

保持 framework `copy_`、device wait before CPU consumption、CPU drain before H2D。
严禁为性能恢复 raw ACL pointer copy。

### D7 Same-path exact

相同 source/artifact/placement/prompt/seed/process 必须得到 exact token trajectory。

### D8 Approximation held-out verified

defer=1 对 defer=0 的 matched-history 差异只由 QD 推导、HD 验证。

### D9 Quality

必须 `NO_STATISTICALLY_MEANINGFUL_REGRESSION`，且无 NaN/Inf/invalid collapse。

---

## 8. Qualification profiles

### B0 retained baseline

| Profile | CPU layers | Experts | Deferred |
|---|---|---:|---:|
| P1-B0 | `{17}` | 4 | 0 |
| P2-B0 | `{1,9,17,26}` | 16 | 0 |
| P3-B0 | `{1,5,8,12,17,19,22,26}` | 32 | 0 |

### D1 mechanism

```text
2 consecutive MoE layers ×4 CPU experts
first layer defer=1; terminal layer defer=0
```

### D2 scale

```text
4 consecutive layers ×4 CPU experts
first 3 defer=1; terminal defer=0
```

### D3 full Round 6 profile

```text
8 consecutive layers ×4 CPU experts
first 7 defer=1; terminal defer=0
```

D1⊂D2⊂D3，使用 nested block。

### Placement anti-overfit

新建独立 selection corpus S6，只在 defer=0 下记录 route statistics。预先固定评分：

```text
CPU route coverage
lowest-score CPU route coverage
4 experts/layer stress
```

从 S6 一次性选择 contiguous block 和 experts，随后冻结 PT/JSON/SHA256。S6 与
QD/HD/free/quality 不重叠，禁止根据 QD/HD 结果重新选 placement。

---

## 9. Corpora

| Corpus | 建议规模 | 目的 | 是否可拟合 |
|---|---:|---|---|
| S6 | 12×64 positions | placement selection / route stress | 只选 placement |
| QD | 16×64，>=1024 | Deferred numerical derivation | YES |
| HD | 16×64，>=1024 | independent held-out | NO |
| FD | >=8×64 free greedy | first-divergence qualification | NO |
| Quality | C-Eval128 + GSM8K MC128 | downstream quality | NO |

冻结 prompt text、token IDs、baseline history、category、protocol、tokenizer revision 和
canonical JSON SHA256。若 HD 被用来修改 bound/rule，必须废止其 held-out 身份并另建
HD2。

---

## 10. Numerical acceptance

### N0 Defer=0 non-regression

旧 contract 不变：B_pair=2.1875、K=32、stable exact 100%、ambiguous membership
100%、overflow=0、expert rel_l2<=1e-2。

### N1 Same-history

```text
z_t^0 = f_defer0(x_<=t)
z_t^1 = f_defer1(x_<=t)
```

禁止用分叉后的 free-running history 推导 threshold。

### N2 Per-position capture

记录 defer0/defer1 top1/top2/top32、full-logit max/mean/rel_l2/cosine、pairwise
distortion、baseline margin、actual deferred IDs/count/weights、producer/consumer、
slot/generation、CPU hits 和 finite。

### N3 Stagewise D1 capture

```text
producer input/router
immediate CPU partial
deferred CPU partial
NPU partial
producer visible output
consumer input
consumer incremental input/output
consumer NPU partial/final output
final logits
```

### N4 QD derivation

对 layer/logit error、top32 candidate error 和 pairwise distortion 统计
p50/p90/p95/p99/p99.5/p99.9/max；按 actual defer count、route-weight bucket、layer、
CPU hit、prompt family 和 prefill/decode 分层。

### N5 Candidate contract

候选 universe=`defer0 top32 ∪ defer1 top32`。对 defer0 top1 `a`：

```text
m_i^0 = z_a^0-z_i^0
m_i^1 = z_a^1-z_i^1
e_i = |m_i^1-m_i^0|
```

在 QD 内比较 p99/p99.5/p99.9/max 和预先声明 safety factor；选择需保持 stable
region 判别力、stable exact=100%、membership=100%、finite、无系统性逐层放大。
进入 HD 前冻结 candidate JSON 与 SHA256。

### N6 HD hard gate

```text
pairwise overflow=0
stable top1 exact=100%
ambiguous membership=100%
all finite
no unseen layer amplification
```

任意 stable flip、escape、overflow：REJECT，停止后续 promotion。

ambiguity size/mass/entropy 仅做 diagnostic；若 ratio 过大，ADR 必须判断 contract 是否
失去区分力。

---

## 11. Quality and performance

质量比较 All-NPU、对应 placement defer=0、D1/D2/D3 defer=1。沿用冻结 evaluator：

```text
C-Eval 128
GSM8K MC 128
20,000 paired bootstrap
```

记录 raw correctness vector、score、invalid、discordant pairs、delta 和 95% CI。
性能必须在同一 D profile 上 defer0/defer1 交替 A/B，warm-up 后至少 5 cycles，记录
TTFT、prefill、decode tok/s、request p50/p95、CPU queue/work、overlap、RSS/HBM、
CPU/NPU utilization。

状态分开：

```text
correctness/quality PASS + benefit beyond noise:
  DEFERRED_ASCEND=A3_VERIFIED_READY; recommended=1

correctness/quality PASS + no measurable benefit:
  DEFERRED_CORRECTNESS=A3_VERIFIED_READY; recommended=0

correctness or quality FAIL:
  DEFERRED_ASCEND=BLOCKED
```

practical performance floor 在只看到 defer0 variance、未看 defer1 结果时冻结；改善的
95% bootstrap CI 不得跨 0。性能不能覆盖 correctness/quality failure。

---

## 12. A3 discipline

新 disposable container：`kt-r6-deferred-ascend`。只用 NPU0 和 Round 5 相同 CPU
affinity。允许在容器内构建 wheel/extension、安装普通依赖、生成 artifacts；禁止修改
host library、业务容器、NPU1–7、宿主 KML 或覆盖 Round 5 artifacts。

保存 container ID、OS/kernel、CANN、torch、torch_npu、Python、compiler、affinity、
NPU visibility、parent/child/wheel/model/tokenizer/GGUF/placement hashes、command/env。

---

## 13. Strict execution order and gates

```text
WP0 provenance freeze                         → G0 COMPLETE
WP1 callgraph/topology/lifetime audit         → G1 SEMANTICS_KNOWN
WP2 unit + behavioral tests/instrumentation   → G2 LOCAL_TESTS_PASS
WP3 S6/QD/HD/FD freeze                        → G3 CORPORA_FROZEN
WP4 D1/D2/D3 placement + GGUF freeze          → G4 ARTIFACTS_FROZEN
WP5 minimal implementation if proven needed   → G5 REVIEWED
WP6 local build/diff/regression               → G6 LOCAL_VERIFIED
WP7 A3 clean build + defer=0 reproduction     → G7 A3_READY
WP8 D1 two-layer mechanism                    → G8 A3_READY
WP9 D1 full determinism/numerical/quality     → G9 A3_READY
WP10 D2 four-layer scale                      → G10 A3_READY
WP11 D3 eight-layer qualification             → G11 A3_READY
WP12 stability + performance A/B              → G12 A3_READY
WP13 retained regression                      → G13 PASS
WP14 ADR/final freeze/feature-branch push
```

前置 hard gate fail 时停止昂贵后续阶段并保存 first-failure evidence。

---

## 14. WP details

### WP0 — Provenance

记录 status、branch、HEAD、logs、recursive submodules、Round5 tag object/target、
main-vs-tag production diff。任何 SHA drift 停止。

### WP1 — Static audit questions

必须回答：

1. 非连续 wrapper 当前如何处理 pending；
2. block terminal 如何 disable defer；
3. 无 valid CPU-deferred ID 是否仍建虚假 task/state；
4. prefill protection 是 per-row 还是 batch-union；
5. class-level pending dict 是否跨 request/model 污染；
6. exception/shutdown 是否 drain；
7. qlen/buffer key/slot 是否隔离 prefill/decode；
8. BF16 boundaries 和 FP32 Hybrid merge 是否保留；
9. NPU immediate/deferred submit 是否 host-observable；
10. callback/sync 是否可能早于 enqueue。

### WP2 — Tests

新增 focused tests：selection partition、tie determinism、defer0 equivalence、no-op
suppression、topology terminal、pending<=1、incremental exactly once、request reset、
exception/shutdown、slot reuse、multi-layer repeated output。grep “no raw memcpy”仅作补充。

### WP3/WP4 — Corpus/artifacts

冻结四套 corpus；用 S6 生成 nested contiguous placements；导出 GGUF；记录 source
tensor、BF16-value、F32 materialized hashes；不得用 QD/HD 重选。

### WP5 — Expected minimal fixes（仅在审计证明后）

1. topology-aware per-layer defer：next layer 有 CPU wrapper 才允许；
2. no valid deferred CPU route 时不提交 task、不置 pending；
3. 缺失时增加最小 request drain/reset；
4. debug-only task/slot/generation trace，默认关闭。

不改变 routing、ownership、mapping、shared expert 或 routed scaling。

### WP6 — Local verification

运行 targeted Python/kt-kernel/SGLang KT EP tests、可用 build、`git diff --check` 和
focused review。确认 default defer=0、无 Dynamic/Qwen/raw ACL/unrelated churn。

### WP7 — A3 defer=0

先跑 Round 5 P1/P2/P3 compact、old pairwise、coverage、CPU-not-hit、transfer regression。
不能重现则 `STOP: BASELINE_REGRESSION`。

### WP8 — D1 mechanism

forced route 覆盖 top-k 每个位置、0/1/multi CPU hit、equal scores、prefill/decode、
two-layer terminal drain；构造 `Σimmediate CPU + ΣDeferred CPU + ΣNPU` 显式结构 reference；
1000 forwards 一个 hash，final pending=0。

### WP9 — D1 full model

```text
9 prompts ×10 repeats ×64 + 8/16/32 prefixes
>=2 clean processes ×9 prompts
QD derive → freeze → HD validate
2/2 layer coverage；8/8 expert target；actual deferred events>0
C-Eval/GSM8K full quality
```

### WP10 — D2

同一 frozen Deferred contract，不 refit。4/4 layers、16/16 experts target、每个非末层
deferred path exercised、terminal drain、determinism/held-out/quality smoke。overflow 即
`D2_DEFERRED_SCALE_OVERFLOW`。

### WP11 — D3

117-request matrix、clean restart、8/8 layers、>=24/32 experts（目标32/32）、七个
deferred boundaries exercised、no overflow、stable=100%、membership=100%、FD 和 full
quality。

### WP12 — Stability/performance

```text
6 prompts×64×3×2 = 2304 tokens
1000 contiguous Deferred forwards
100 sequential requests for state leakage
graceful shutdown with pending work
injected task exception
```

记录 hashes、pending high-water、slot generations、RSS/HBM、threads/FD、crash/deadlock。

### WP13 — Retained regression

重跑 R2A/R2B/R2C、R3 real anchor、SGLang KT EP、R4A numerical boundaries、R4A.5
transfer determinism、Round5 P1/P2/P3 compact 和 Round6 fixtures。unexplained failure=0。

---

## 15. Failure classification/debug order

分类：

```text
BASELINE_REGRESSION / DEFER_SELECTION / DEFER_TOPOLOGY / DEFER_EMPTY_TASK
PENDING_STATE / PENDING_DRAIN / INCREMENTAL_MERGE / DOUBLE_MERGE
DROPPED_CONTRIBUTION / BUFFER_SLOT_REUSE / CROSS_REQUEST_LEAK
CALLBACK_ORDERING / TRANSFER_ORDERING / CPUINFER_QUEUE
SAME_PATH_NONDETERMINISM / EXPERT_NUMERICAL / LAYER_NUMERICAL
DEFERRED_ENVELOPE_OVERFLOW / STABLE_FLIP / AMBIGUITY_ESCAPE
QUALITY_REGRESSION / COVERAGE_FAILURE / HOST_MEMORY / NPU_MEMORY
PERFORMANCE_VALUE_NOT_DEMONSTRATED / RETAINED_REGRESSION / OTHER
```

Debug 顺序：source/artifact → partition → ownership → eligible defer → producer inputs →
enqueue → pending → slot generation → CPU output → incremental exactly once → NPU partial →
FP32 merge → layer → matched-history logits → token/quality。

---

## 16. Hard STOP rules

遇到 provenance/artifact drift、defer0 regression、silent sparse pending、drop/duplicate、
pending>1、request残留、nondeterminism、restart mismatch、slot mismatch、raw ACL 回退、
held-out stable flip/overflow/escape、quality regression、crash/deadlock/non-finite、unbounded
memory 或 unexplained regression，立即停止。

禁止通过降低 Deferred coverage、换低频 expert、删除 prompt、HD fitting、扩大 bound、
减少 token、用 sequential 代替 production overlap、改 worker 后直接宣称通过来规避。

---

## 17. Deliverables

```text
migration_bridge/round6/
├── 00_BASE_AND_SCOPE.md
├── 01_DEFERRED_CALLGRAPH.md
├── 02_DEFERRED_TOPOLOGY_AND_LIFETIME.md
├── 03_DEFERRED_RISK_REGISTER.md
├── 04_CORPUS_FREEZE.md
├── 05_PLACEMENT_AND_WEIGHT_FREEZE.md
├── 06_LOCAL_TESTS_AND_BUILD.md
├── 07_A3_DEFER0_BASELINE.md
├── 08_D1_MECHANISM_QUALIFICATION.md
├── 09_D1_FULL_MODEL.md
├── 10_D2_FOUR_LAYER.md
├── 11_D3_EIGHT_LAYER.md
├── 12_DEFERRED_NUMERICAL_DERIVATION.md
├── 13_HELDOUT_VALIDATION.md
├── 14_FREE_GENERATION.md
├── 15_DOWNSTREAM_QUALITY.md
├── 16_STABILITY_AND_MEMORY.md
├── 17_PERFORMANCE_AB.md
├── 18_RETAINED_REGRESSION.md
├── 19_GIT_COMMITS.md
├── ADR_R6_DEFERRED_EXPERT_ASCEND.md
├── DEFERRED_NUMERICAL_ACCEPTANCE_CONTRACT.json
├── ROUND6_FINAL_SUMMARY.md
├── tools/
├── corpora/
├── placements/
├── manifests/
└── evidence/
```

工具覆盖 callgraph、partition、lifecycle trace、route profile、contiguous placement、GGUF
export/identity、mechanism matrix、same-path、teacher-forced、contract derive/validate、free
generation、quality、stability、performance 和 summary。全部 deterministic、seed frozen、
JSON+human report、SHA256、A3 运行无隐藏网络依赖。

---

## 18. ADR outcomes

### A — APPROVE defer=1

所有 topology、exactly-once、determinism、restart、QD/HD、stable/membership、quality、
stability、regression gates PASS：

```text
ROUND6_DEFERRED_EXPERT_ASCEND = A3_VERIFIED_READY
DEFERRED_NUMERICAL_ACCEPTANCE = HELDOUT_VERIFIED
DSV2_LITE_DEFERRED_TP1 = A3_VERIFIED_READY
MAX_DEFERRED_EXPERTS_QUALIFIED = 1
```

只有性能收益也成立，`DEFERRED_RECOMMENDED_PROFILE=1`。

### B — CORRECT BUT NO PERFORMANCE VALUE

```text
DEFERRED_CORRECTNESS = A3_VERIFIED_READY
DEFERRED_RECOMMENDED_PROFILE = 0
PERFORMANCE_VALUE = NOT_DEMONSTRATED
```

### C — REJECT

```text
ROUND6_DEFERRED_EXPERT_ASCEND = BLOCKED_<GATE>
MAX_DEFERRED_EXPERTS_QUALIFIED = 0
```

Round 5 static Hybrid 保持 verified，不降级。

---

## 19. Final gate table

| Gate | D1 | D2 | D3 | Requirement |
|---|---:|---:|---:|---|
| topology | PASS | PASS | PASS | exact |
| partition | PASS | PASS | PASS | exact |
| pending/slot | PASS | PASS | PASS | exact |
| same-process | PASS | PASS | PASS | exact |
| clean restart | PASS | PASS | PASS | exact |
| numerical contract | PASS | PASS | PASS | frozen/no refit |
| layer coverage | 2/2 | 4/4 | 8/8 | exact |
| expert coverage | target 8/8 | target 16/16 | >=24/32 | frozen workload |
| deferred boundaries | >0 | all 3 | all 7 | required |
| CPU-not-hit | PASS | PASS | PASS | exact |
| free generation | PASS | smoke | full | stable hard gate |
| quality | full | smoke | full | no regression |
| stability | mechanism | compact | full | PASS |
| regression | compact | compact | final full | 0 unexplained |

建议成功 tag：

```text
round6-dsv2lite-deferred1-ascend-a3-verified
```

tag 指向 final clean commit，不得移动。

---

## 20. Owner 审阅的四项决策

1. **v1 只资格验证 defer=1？** 建议 YES；`>=2` 另开 Round 6.x。
2. **允许新建 contiguous D1/D2/D3 placement？** 建议 YES；这是紧邻增量语义的必要
   条件，原 P1/P2/P3 保持 defer=0 historical identity。
3. **建立独立 Deferred held-out numerical contract？** 建议 YES；不能偷用普通异构
   reduction contract。
4. **correctness PASS 但无可测收益时仍默认 defer=0？** 建议 YES。

---

## 21. One-line rule

```text
从 Round 5 immutable static Hybrid 建立独立 worktree，先证明 defer=0 无回退，
再在连续 CPU-enabled layer block 上只打开 defer=1；以 exactly-once pending/merge、
same-path determinism、独立 held-out numerical contract、quality 和 lifecycle 为硬门禁，
不触碰 Dynamic Placement 或 Qwen3.8。
```

