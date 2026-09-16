# Round6.5A 总结

## 结论

```
ROUND6_5A_ROOT_CAUSE_INVESTIGATION = COMPLETE
ROOT_CAUSE = INHERENT_CROSS_BACKEND_NUMERICAL_AMPLIFICATION
PRODUCTION_FIX = NONE_NO_SEMANTIC_INVARIANT_DEFECT_DEMONSTRATED
```

本轮完成了冻结失败轨迹的因果分流，没有通过修改阈值、换样本或改
placement 来规避失败。结论限定于本轮冻结轨迹及明确列出的局部测试，
不是“所有输入都绝无实现缺陷”的证明。

## 最关键的新发现

1. **仅 Layer1 做 R32 已能复现放大。** C1/C2/C3/C4 的最大 pairwise
   distortion 分别为 8.953125 / 11.84375 / 9.59375 / 9.59375，首次越界
   都在原 token54。原单层通过项是 Layer17，不能外推为任意单层安全。
2. **真实 grouped-MoE 的 R64/R32 布局对照 exact。** 四层共32个
   resident experts，每种布局10次重复，无数值布局异常。
3. **canonical mixed 重建 exact。** 四层失败 decode 输入及四层真实
   prefill 输入，最终 routed output 均与生产捕获 exact。没有发现应修复
   的多余 cast、错误 merge 或明显局部 kernel 失效。
4. **同一 router 算子重现 expert-set flip。** E29 在 baseline 是第8名，
   不是第7名；E62−E29 从 +0.0421795 变为 −0.0622580。
5. **NPU-for-CPU 诊断恢复完整 logits exact。** 保留四层R32 ownership、
   CPU任务完成及原merge，只替换CPU数值结果后，64/64个完整logits向量与
   原All-NPU oracle一致，maxabs=0、pairwise overflow=0。

## 门禁与证据

| 项目 | 结果 | 证据文档 |
|---|---|---|
| 历史失败保护 | 原825份封存artifact校验通过；未覆盖 | 00_PROVENANCE.md |
| 实际部署身份 | 前后91个manifest文件及repo状态一致 | 02_R32_NPU_SUBSET_EQUIVALENCE.md |
| R64/R32 NPU布局 | 32/32 exact，布局内重复exact | 02_R32_NPU_SUBSET_EQUIVALENCE.md |
| Canonical mixed | 8/8最终边界exact | 03_CANONICAL_MIXED_ORACLE.md |
| 新局部expert安全性 | 最大rel-L2约0.000517213 <0.01 | 03_CANONICAL_MIXED_ORACLE.md |
| 累积层定位 | 最小已测失败前缀为C1 | 04_CUMULATIVE_LAYER_BISECTION.md |
| Router重放 | Layer12/17 logits与Top-K均exact复现 | 05_ROUTER_BOUNDARY_ANALYSIS.md |
| 固定baseline routes | Layer17/full-logit L2降低，但max pairwise更坏 | 06_COUNTERFACTUAL_ROUTE_REPLAY.md |
| NPU-for-CPU | 260次干预，64/64完整logits exact | 06_COUNTERFACTUAL_ROUTE_REPLAY.md |
| 线程/算术审计 | 32workers、1subpool；未改生产边界 | 07_ARITHMETIC_AND_REDUCTION_AUDIT.md |
| 最终决策 | 固有跨backend扰动放大；不晋级 | 08_ROOT_CAUSE_ADR.md |

固定baseline routes不是修复：最终logit rel-L2由约0.569降至0.135，
但max pairwise由9.59375升至16.65625。报告保留这项反向结果，不能只引用
改善的指标。NPU-for-CPU也仅为因果诊断，不能作为真实CPU offload资格结果。

## 生产与历史状态

```
ROUND6_5_HEAVY_OFFLOAD_ALIGNMENT = BLOCKED_HEAVY_OFFLOAD_RUNTIME
BLOCKER_SUBCLASS = STATIC_NUMERICAL_SCALE_BLOCKER
FIRST_BLOCKED_PROFILE = four-R32
B_pair = 2.1875
K = 32
RESIDENT_RATIO_PARETO = NOT_COMPLETE
ROUND7_BASELINE = NOT_FROZEN
ROUND7 = NOT_STARTED
DEFAULT_DEFERRED = OFF
DEFERRED_HEAVY_PROFILE_PROBE = NOT_RUN_PREDECESSOR_FAILURE
```

没有生产代码修改。没有运行R16/R8晋级、all-MoE、quality晋级或Dynamic。
没有增加CPU核数、改变worker/NUMA配置或操作他人任务。

## 验证与可复现性

- 本地Python语法检查、shell语法检查和git diff whitespace检查通过。
- 目标机CPU-only frozen metric保护测试4/4通过，包括“top1相同不能覆盖
  pairwise overflow”和非有限值拒绝。
- 九个model/operator guard均结束并回收自己的子进程。
- 两次诊断工具问题保留：forced baseline被原资格评估器拒绝；离线汇总
  多余依赖SGLang。分别通过独立诊断比较器、CPU-only metrics helper修正，
  没有修改原冻结评估器，也没有覆盖原失败记录。
- 机器可读决策和分层证据在`evidence/`；大型逐forward原始张量保留目标机，
  其SHA记录在artifact manifest，不伪称已全部放入Git。

## Git与运行环境

Base: `b2e9d8721911402e94cf3b103f3e0d3800fd81be`。
Branch: `codex/round6-5a-static-numerical-rootcause`。
实际部署parent/child及overlays保持不变，以00文档和runtime manifest为准。
目标：110.129.0.22，`kt-r65-static-offload`，仅NPU0，32workers。
已有`third_party/llama.cpp`未提交修改原样保留，不属于本轮改动。
本轮只在本地提交，未自动推送GitHub或合并main。

## 下一步边界

需要新建独立的Round6.6/6.5R heavy-static numerical qualification，预注册
新的QD/HD及paired quality协议，继续严格保留系统语义、局部算子安全、
同路径确定性、clean restart和CPU-not-hit门禁。不得拿本轮观察到的最大值
加margin做新阈值，也不得自动继承Deferred近似执行的质量容忍度。
该新研究不包含在本轮已完成的根因诊断结论中。

## 最终封存与资源释放

紧凑证据包SHA256：
`62570988b5f7052beace0df7bb8ba1a4cb3023dab07106206a7f004df0e8c5d2`。
完整manifest登记115,946个诊断文件；本地仅保存紧凑证据，完整张量仍在
目标机持久目录，未删除。可运行`tools/validate_freeze.py`复核本地证据。

57,579,411,648-byte GGUF新算SHA仍为
`20114dba7790a9b6be40a17ffb3881c516b81ea86b8f2fcfd3ea61d052e96fe2`。

所有测试进程已回收；专用容器`kt-r65-static-offload`已恢复为exited，
最后一次NPU0查询为`No process in device`。只停止本项目专用容器，未删除
容器/镜像/证据，未操作业务任务或NPU1–7。
