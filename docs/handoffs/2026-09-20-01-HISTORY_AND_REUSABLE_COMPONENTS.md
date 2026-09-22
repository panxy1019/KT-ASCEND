# KT-ASCEND 历史开发与可复用组件交接

日期：2026-09-20。用途：新 Codex 窗口接手，不依赖旧对话记忆。
这是证据索引与工程地图，不为历史未通过门禁补发 PASS。

## 1. 当前项目、工作区与阅读顺序

- 用户仓库： https://github.com/panxy1019/KT-ASCEND
- 当前本地工作树：`/home/admin/Documents/ChatGPT/KT/kt-round8-qwen35-prep`
- 当前分支：`codex/round9-heterogeneous-working-set`。
- 本文核查时 HEAD：`6a57b011c016db2a7331ea323e1493ab9bc3e115`，交接提交将在其后；请重新 `git rev-parse HEAD`。
- Round7 完整收口工作树：`/home/admin/Documents/ChatGPT/KT/kt-round7`。
- 本文下文相对路径均基于当前工作树，另行写绝对路径的除外。
- 下一份必读：同目录 `2026-09-20-02-ROUND9_EXECUTION_HANDOFF.md`。
- 活状态优先入口：`migration_bridge/round9/CURRENT_EXECUTION.md`；其中旧段落按历史保留，最新段落优先。

**最新用户决定：保留远端后台下载，暂停旧线程定时任务，迁移开发上下文到新窗口。**
定时任务 `round7` 已通过应用工具改为 PAUSED；不是删除，也不是停止 systemd 下载进程。
新窗口不要擅自恢复旧线程定时任务，避免两个窗口同时操作同一下载控制器。

## 2. 项目主线与资格边界

项目是 KTransformers × Kunpeng × Ascend 的异构 MoE 移植与工作集运行时。
参考过 `panxy1019/infra`，生产 serving/model source of truth 是 **kvcache-ai/sglang fork**。
主要集成点是 `KTEPWrapperMethod`，不是另写一套独立 Hybrid coordinator。

演进主线：CPU expert plane → Ascend runtime → 合成 Hybrid → DSV2 真模型 →
低卸载密度多层正确性 → upstream alignment → Deferred → heavy static → C64 →
Frequency/Dynamic → Qwen3.5 → 当前 Qwen3.8。

历史数值策略发生过明确的独立实验变更：

1. Round4A/5：冻结 `B_pair=2.1875, K=32`，属于已验证低卸载 profile。
2. Round6.5：heavy four-R32 超出该包络，历史结果保持 BLOCKED。
3. Round6.5A/6.6：排查实现、三路 reference，不能据此把旧 FAIL 改成 PASS。
4. 后续用户批准产品策略：CMMLU 256，观察准确率相对参考最多下降 5pp。
5. 新模型重新建立参考；系统、权重、ownership、生命周期仍是严格门禁。

CMMLU 产品筛查不是统计非劣性证明。不同后端回答可以不同，不代表允许错误权重、漏算、重复 shared expert 或同配置不确定性。

## 3. 历史各轮：结果、原因与文档入口

| 阶段 | 已建立的能力/结论 | 文档入口 |
|---|---|---|
| Round1/1.5 | Kunpeng CPU、KML 可用性调查，转向 LLAMAFILE/CPUInfer 的工程基础 | `migration_bridge/B_R1_SUMMARY.md`、`B_R15_DECISION.md`、`B_R15_LLAMAFILE_ARM_AUDIT.md`、`B_R15_LLAMAFILE_RUNTIME.md` |
| Round2A/2B/2C、Round3 | CPU expert、Ascend stream/transfer、合成 Hybrid、DSV2 Layer17/Expert8 真模型锚点；后续 Round5 重新回归 | `migration_bridge/02_CURRENT_ARCHITECTURE.md`；`migration_bridge/round5/11_R2A_R2B_R2C_REQUALIFICATION.md`、`12_ROUND3_REPLAY.md` |
| Round4A.1–.4 | BF16-visible 边界、FP32 routed partial 合并、独立 QD/HD 数值合同 | `migration_bridge/round4a2/`、`round4a3/`、`round4a4/ROUND4A4_SUMMARY.md` |
| Round4A.5 | raw ACL pointer copy 绕过框架依赖/生命周期导致多层不确定性；改 framework-managed Tensor.copy_ | `migration_bridge/round4a5/05_RAW_ACL_TRANSFER_ROOT_CAUSE.md`、`ROUND4A_FINAL_SUMMARY.md`、`ROUND4A_FINAL_GATE_MATRIX.md`、`p3/P3_FINAL_SUMMARY.md` |
| Round5 | UPSTREAM_FIRST 最小语义移植，P1/P2/P3 A3 重新通过 | `migration_bridge/round5/ROUND5_FINAL_SUMMARY.md`、`03_LOCAL_FIX_TO_UPSTREAM_MATRIX.md`、`09_ALIGNMENT_ADR.md`、`19_PATCH_RETIREMENT_MATRIX.md` |
| Round6/6.1–6.3 | Deferred runtime 与 numerical/capture 协议探索；完整 forward 关联、observer parity、采集边界成为重要资产 | `migration_bridge/round6_1/ROUND6_1_FINAL_SUMMARY.md`、`round6_2/ROUND6_2_FINAL_SUMMARY.md`、`round6_3/ROUND6_3_BLOCKED_SUMMARY.md` |
| Round6.4 | 接受官方跨层 Deferred 属于近似执行；最终 IMPLEMENTED_NO_SPEEDUP，默认 OFF | `migration_bridge/round6_4/ROUND6_4_FINAL_SUMMARY.md`、`20_FINAL_ADR.md` |
| Round6.5 | four-R32 历史数值包络失败，禁止靠换样本/专家/阈值求通过 | `migration_bridge/round6_5/ROUND6_5_FINAL_SUMMARY.md`、`04A_FIRST_FAILURE_INVESTIGATION.md` |
| Round6.5A | INHERENT_CROSS_BACKEND_NUMERICAL_AMPLIFICATION：微小后端差异累积、router Top-K 离散变化、下游放大 | `migration_bridge/round6_5a/ROUND6_5A_FINAL_SUMMARY.md`、`08_ROOT_CAUSE_ADR.md` |
| Round6.6/A/B | canonical reference、projection decomposition、exact controls；608 cases/128 experts 局部安全通过，旧质量 study 未通过 | `migration_bridge/round6_6/10_BACKEND_DECOMPOSITION_INTERPRETATION.md`、`round6_6b/ROUND6_6B_FINAL_SUMMARY.md` |
| Round6.7/C64 | 新的 CMMLU-only 5pp 产品策略；DSV2 R16/R8-C64 系统/质量通过，选择 Uniform R8-C64，R16 fallback | `migration_bridge/round6_7/10_PRODUCT_POLICY_ADR.md`、`round6_7_c64/FINAL_COMPARISON_AND_DECISION.md` |
| Round7 | Frequency、Dynamic、Deferred 独立评估；用户缩减性能范围后收口，最终 Uniform R8-C64 Static，Dynamic/Deferred OFF | **另一工作树** `/home/admin/Documents/ChatGPT/KT/kt-round7/migration_bridge/round7/05_FINAL/ROUND7_FINAL_REPORT.md`、`ROUND7_ADR.md` |
| Round8 | Qwen3.5 Q64 工程 reference → Q32 Static 产品基线；性能用户取消后审计收口 | `round8_prep/FINAL/ROUND8_FINAL_REPORT.md` |
| Round9 | Qwen3.8 来源获取/身份阶段；模型下载进行中，尚未全模型部署资格 | `migration_bridge/round9/`，下一份交接文档 |

### 3.1 不可变基线与 Git 注意事项

- Round4A tag `round4a-tp1-multiplacement-a3-verified`；tag object `9e2c9e220337f57b55f1fe9b50da0862ccc0cf64`；target `096a6ef860a01dc0a3c8f2cc5618a396425ee09b`。
- Round5 upstream parent `440df948027128318bc30a11a545433eb0c7d383`；upstream child `3424f35d0e60f03b9e75d9b90941f8f4cb944524`。
- Round5 aligned parent `be5b48621db32fb894402e3e886060dea263cdc9`；aligned child `412219b4b05a888808f0b88d2963a8b6164a0fda`；tag `round5-upstream-aligned-tp1-a3-verified`（以上来自冻结报告，使用前可只读复核）。
- Round8 closure commit `9c1f1dae92303c1a13078319ab2aa2b5026fdb9e`；用户后来单独授权推送，当前本地 remote-tracking `origin/codex/round8-qwen35-prep` 指向该 SHA。最终报告中的“no push”是当时状态，不代表后续未推送。
- Round9 本地提交未因此获得持续 push 权限。禁止自动 push、merge main、移动 tag。
- 当前 `third_party/sglang` gitlink `f02f304a25364d1e8fe9c52515a22f20f5089a52`；本地子模块显示未初始化。不要把 gitlink 当已部署 imported code。
- 其他工作树曾有用户自己的 llama.cpp 未提交修改；不要跨工作树 reset/clean。

### 3.2 Round7 精确结论，防止错误继承

Frequency CMMLU173/256 vs174参考；综合 NPU hit12.44%→26.27%，但主要来自 prefill，decode CPU assignments反而+1.26%，保留 Uniform。
Dynamic migration/held-out系统通过，但冻结短decode HOLDOUT未展示route收益；Dynamic CMMLU174/256的trigger为0，不能冒充触发迁移后的质量证明。
Deferred D1系统及CMMLU174/256通过；短请求描述性结果更慢，12个计时窗口仅2个资源合格，不能宣称完整独占性能资格。
Round7状态 `COMPLETE_WITH_USER_APPROVED_SCOPE_REDUCTION`，不是原始全门禁性能通过。

### 3.3 Round8 最终继承对象

`ROUND8=COMPLETE_WITH_USER_SCOPE_CHANGE`；Qwen3.5-35B-A3B **Q32-C64 Static**，Dynamic OFF，Deferred OFF；Q64是工程参考，不是All-NPU。

| 项目 | 冻结结果 |
|---|---|
| Model revision | `59d61f3ce65a6d9863b86d2e96597125219dc754` |
| 全CPU bank身份 | 40层、30,720 projection instances exact |
| Q64 CMMLU | 222/256，invalid0 |
| Q32 CMMLU | 225/256，invalid0 |
| Dynamic CMMLU | 224/256，invalid0；217/256触发 |
| Dynamic held-out | 128/128，8,320 updates，96 triggered requests |
| Static稳定性 | 两campaign，总2,304tokens，限测试范围 |
| 正式性能 | NOT_QUALIFIED_RESOURCE_AUDIT；后续SKIPPED_BY_USER_SCOPE_CHANGE |

核心身份：

- F32 bank manifest：`35ca0ac63a3646ba7b89675053bbead19724325fa9d73f99910ef3c156a84405`。
- Q32 mask：`b358399e6405d60b81d8231529b1ec6024ff7138cddab2e0cbac3ccf0b25f486`。
- CMMLU manifest：`365f557c0746db1e02ddee4ec66c7afb5e72eeb17e8dd8e0a47d1ed2b7af1b9d`。
- Runtime：.22、Kunpeng/910B3，历史NPU2、CPU128–191、NUMA4/5；C64=2×32worker，512intermediate分256+256，FP32 partial。新运行需重新资源准入，不保证该资源仍空闲。

Dynamic资源局限仍保留：BF16 full-context缓存、NUMA页回收/major faults、高启动成本；用户停止性能研究不等于这些问题已修复。不要恢复Round8性能去阻塞Round9。

## 4. 可复用组件：位置、作用与迁移注意事项

| 组件 | 实际路径/证据 | 可复用思路及边界 |
|---|---|---|
| LLAMAFILE expert算子 | `kt-kernel/operators/llamafile/moe.hpp` | gate/up/down、SwiGLU、effective vec-dot dtype、route累加；新shape640与低比特仍需实测 |
| worker pool/CPU子池 | `kt-kernel/cpu_backend/worker_pool.cpp`、`worker_pool.h` | CPU任务分片与生命周期；C64不是仅把线程参数改64 |
| Ascend vendor/callback | `kt-kernel/cpu_backend/vendors/ascend_npu.h`、`ascend_callback_worker.cpp`、`.h` | framework stream、callback/report生命周期；CPUInfer借用NPU流，不拥有它 |
| CPU transfer/buffer wrapper | `kt-kernel/python/experts_base.py` | pinned staging、copy_、提交/同步、buffer复用；禁止退回裸ACL pointer memcpy |
| SGLang KTEP | `round8_prep/runtime/kt_ep_wrapper.py`；子模块逻辑路径`python/sglang/srt/layers/moe/kt_ep_wrapper.py` | ownership、logical→physical mapping、partial merge、mask/generation；runtime overlay与gitlink分开记录 |
| C64开发与分片 | `migration_bridge/round6_7_c64/04_F32_TAIL_MICROKERNEL.md`、`05_F32_TP_PARTITION.md`、`06_FORWARD_MANY_TAIL.md`、`07_FP32_PARTIAL_MERGE.md` | DSV2的704+704、grouped-prefill、tail-safe kernel；不能直接宣称640/320或160 shape通过 |
| C64构建/回归 | `migration_bridge/round6_7_c64/tools/build_c64_extension.py`、`run_operator_matrix.py`、`audit_split_contract.py`、`audit_memory.py` | 构建身份、算子矩阵、NUMA/内存证据，需改成新模型独立实例 |
| NPU布局/权重恢复 | `migration_bridge/round7/tools/bf16_resident_layout.py`、`test_bf16_resident_layout.py` | resident/raw方向与ND/NZ规范化；shape/dtype相同不保证逻辑权重相同 |
| Dynamic失败修复证据 | `migration_bridge/round7/02_DYNAMIC/CPU_BACKING_FIX.md`、`FULL_CONTEXT_WEIGHT_IDENTITY_DIAGNOSIS.md`、`RESIDENT_TO_RAW_LAYOUT_REPAIR.md` | 全expert CPU backing不能遗漏当前resident专家；promotion原source、demotion可计算 |
| Qwen迁移ND/NZ回归 | `round8_prep/11_DYNAMIC/NZ_TRANSFER_FIX.md`、`tools/test_dynamic_nz_transfer.py`、`capture_dynamic_transfer_regression.py` | 明确矩阵方向、transpose/materialization、真实tensor identity；Qwen3.8需要新shape覆盖 |
| 样本抽取/转换 | `round8_prep/tools/extract_sample_expert.py`、`inspect_sample_shards.py`、`convert_all_experts.py`、`verify_checkpoint.py` | 从冻结checkpoint逐tensor审计，无缺失/重复/MTP混入；不能照抄Qwen3.5键名/packing轴 |
| 三路算子oracle | `round8_prep/tools/sample_reference_probe.py`、`llamafile_sample_probe.py`、`three_way_grouped_probe.py`、`multi_expert_probe.py` | NPU/production CPU/canonical CPU相同operand比较，区分kernel错误与backend数值差异 |
| GDN/卷积state探针 | `round8_prep/tools/gdn_primitive_probe.py`、`gdn_state_probe.py`、`conv_state_probe.py` | prefill/decode、chunk/reset、跨请求隔离；不能替代QSA/GR/PLE资格 |
| compact/restart/parity | `round8_prep/tools/q64_compact.py`、`q64_restart_and_screen.py`、`q32_compact_when_ready.py` | 32条compact、同路径重复与重启；新profile重新冻结 |
| generation/迁移审计 | `round8_prep/tools/qwen_dynamic_audit.py`、`audit_qwen_dynamic_results.py`；`migration_bridge/round7/tools/dynamic_audit.py` | weight/mapping/mask同generation、触发/不触发、重启；新模型不能继承旧threshold |
| forward关联采集 | `migration_bridge/round7/tools/associate_forwards.py`、`dynamic_telemetry.py`、`round7_hooks.py` | 所有forward/request/token边界完整关联，observer OFF/ON；timed path不带诊断hook |
| 冻结质量评估 | `round8_prep/tools/qwen_cmmlu_reference.py`、`qwen_cmmlu_q32.py`、`qwen_dynamic_evaluation.py` | 同题/模板/token预算/解析器，invalid也进入规则；新模型建立新参考分数 |
| 稳定性/资源准入 | `round8_prep/tools/qwen_stability_client.py`、`run_qwen_stability.py`、`round8_resource_guard.py`、`admit_q64_host.py` | owned PID/cgroup隔离、资源峰值/回收，不操作他人进程 |
| 性能工具 | `round8_prep/tools/qwen_static_perf.py`、`qwen_dynamic_perf.py`、`run_qwen_abba.py` | 同机独占、hook-free、固定预算、资源窗口审计；现有结果未合格不能作为速度基线 |
| 下载及完整性 | `migration_bridge/round9/tools/range_recovery.py`、`download_guard.py`、`verify_download.py` | 8分片×2range、独立断点、no-clobber SHA发布、最终全manifest审计 |

## 5. 必须保留的工程原则

1. 全局logical expert ID是唯一身份；CPU/NPU集合互斥且完备。
2. route weight、routed scaling、shared expert各一次；新模型从源码确定owner。
3. Tensor.copy_/stream/event满足生产者可见性与consumer结束后复用，不能用性能理由绕过。
4. CPU/NPU routed partial用明确dtype合并；权重存储类型不等于累加类型。
5. 迁移发布必须weight=mapping=mask generation，失败不能继续读半更新slot。
6. 实际import路径、Python hash、native .so、overlay、image/CANN/torch_npu比本地HEAD更接近运行事实。
7. 比较不同precision/topology时可以存在数值差异；同一配置不确定性必须调查。
8. 独立质量/held-out前冻结候选；不看CMMLU对错挑experts，不改失败样本或提高门限。
9. reused code ≠ inherited qualification。新模型、shape、设备组合要重新验证。

## 6. 给新窗口的第一条消息

```text
请接手KT-ASCEND Round9。工作目录：
/home/admin/Documents/ChatGPT/KT/kt-round8-qwen35-prep
先完整阅读docs/handoffs/2026-09-20-01-HISTORY_AND_REUSABLE_COMPONENTS.md
与docs/handoffs/2026-09-20-02-ROUND9_EXECUTION_HANDOFF.md，再读
migration_bridge/round9/CURRENT_EXECUTION.md和精度/C128补充计划。
当前.22后台8分片16连接下载保持运行；旧round7定时任务已由用户暂停。
先只读核对状态，勿重复启动。下载校验完成后按独立资源/语义/算子门禁
推进Round9；不要恢复Round8性能，不push/main/tag，不继承旧模型资格。
```

文档不含口令/token。已有连接方式或安全凭据若不可用，应报告访问阻塞，不能把历史对话密码写入交接文件。
