# Round12a current execution

Updated: 2026-09-27 UTC

Status: `ROUND12A_COMPLETE_KT_STYLE_VALUE_CONFIRMED`.

The tied-parameter-aware OffloaderV1 fix is qualified by v172. The first same-revision non-KT request passed in v173. The fair promoted-QSA KT/NON_KT/NON_KT/KT sequence v174-v177 passed exact quality, resource, cleanup and performance gates. KT median E2E was 2.982178643 s versus OffloaderV1 1204.699036151 s in block A and 3.386335781 s versus 1349.630512928 s in block B, a 99.752454% and 99.749092% reduction respectively.

The user-added Native SGLang TP8/old-QSA baseline also completed. v180-v183 are identical-code independent A/B/B/A repeats on physical0-7. Their process E2E medians were 3.428952584, 3.371347328, 3.489020653 and 3.325310193 seconds; max/min=1.049231636, all twelve measured outputs were exactly `[3897, 248046]`, cross-run per-device HBM spread was at most 2 MiB, and Host peak ratio was 1.000052514. This is repeatability evidence, not a fair speedup comparison against the promoted-QSA KT arm.

Every model block was fully archived and SHA-verified before targeted POSIX_FADV_DONTNEED. Nine retirement runs ended at zero registered resident pages without global drop_caches. Physical0-7 and physical12 are empty, no Round12 container/process/timer remains, and the registered training container identity/cgroup is unchanged. Automation remains paused.

Fresh final verification reran 148 tests and compiled 34 Python files. Both comparators exactly reproduce their frozen results under their canonical serialization, all archived SHA receipts pass, and a 2026-09-27 06:22:31 UTC read-only remote audit reconfirms physical0-7/12 cleanup and the unchanged training identity. The first audit parser false-positive for three exited legacy Round9 containers is preserved separately; no legacy or external container was modified.

Primary evidence: `ROUND12A_FINAL_STATUS.json`, `ROUND12A_KT_NONKT_ABBA_RESULT.json`, `ROUND12A_CROSS_BACKEND_SYNTHESIS.json`, `native_tp8/NATIVE_TP8_ABBA_RESULT.json`, and `ROUND12A_FINAL_REPORT_ZH.md`.

---

# Round11 current execution

Updated: 2026-09-26 UTC

Status: `ROUND11_HYBRID_PERFORMANCE_AND_BACKEND_QUALIFIED`

Round11 T0–T6 is complete. Final runtime remains HFINAL: threshold129, C128/MXFP4 on NUMA2/3/4/7, Static Q32 BF16/NZ29, chunk32, PLE ASYNC, Dynamic OFF, original QSA and original Layerwise shadow32. Hybrid forced-path evidence is qualified through prompt768 and Decode256; 1024 hit the resource guard and 769–1023 remain unqualified. The default path switch at 129 is unchanged.

Kunpeng attribution assigns 66.70% of the frozen group parent to native compute+merge and only 1.20% to queue wait; no bounded candidate was selected. Dynamic is transaction-correct but has `NO_FINITE_BREAK_EVEN`. QSA is the dominant B-route hotspot and its single component candidate passed numerical/component gates, but the model integration gate failed before any request and the candidate closes `COMPONENT_ONLY_NO_PROMOTION`.

T6 v147–v150 passed four independent unprofiled HFINAL processes, exact Decode32/64/128 outputs, five canary classes, two cancel/reuse controls, all guards and cleanup. BASELINE/FINAL labels use identical compute code; timing ratios are process variance, not a speedup claim. Final budget is 19/21 model starts, 80/108 generations, 4/5 aborts and zero retries. Exact post-hash cache retirement completed after each final block without global drop_caches. Physical12 is empty; no Round11 container or process remains; automation stays paused.

Primary evidence: `artifacts/round11/ROUND11_FINAL_STATUS.json`, `ROUND11_T6_FINAL_ANALYSIS.json`, `ROUND11_T5_FINAL_DISPOSITION.json`, `ROUND11_RESOURCE_RETIREMENT_RECEIPT.json`, and `docs/reports/2026-09-26-round11-final-zh.md`.

---

# Round10 current execution

Updated: 2026-09-26 UTC

Status: `ROUND10_AB_NARROW_QUALIFIED`

Round10 T0–T6 is complete under the frozen serial TP1, input≤128 narrow contract. HFINAL equals the H0 Hybrid129 static configuration: C128/MXFP4 CPU experts, Q32 BF16/NZ29 resident NPU experts, chunk32, PLE ASYNC and Dynamic OFF. No new performance candidate is promoted.

T1 qualified Hybrid policy, deterministic canaries, edge lengths through 128, multi-step Decode, cancellation and same-process reuse. T2 froze host-only cost attribution and a real replay fixture. T3 verified the current native Kunpeng worker path against an independent same-precision reference and retained the existing backend. T4 proved the current Hybrid→Dynamic transaction, then rejected enablement because its 64-token median was 4.558176× H0. T5 independently qualified current GDN, QSA and PLE semantics/state and selected no B-path performance candidate.

T6 v107 completed 8/8 generations and 1/1 abort with exact matching token outputs against the corresponding v95 rows. Its 64-token measurements were 5.409330s and 5.832713s, median 5.621022s versus historical H0 median 5.188200s (1.083424×); this is reported as same-configuration repeat dispersion and does not support a speedup claim. The supervisor passed, guard acknowledged exit, the owned container was removed, and physical12 is empty. The outer controller's post-cleanup receipt filename bug is preserved and reconciled; its generator regression is fixed locally without another model start.

Final Round10 budget: 4 model starts, 36 generations, 4 abort controls, zero automatic retries; T3 used 6 component processes and T5 used 3. Targeted cache retirement covered 179 unreferenced project files without global drop_caches. Automation remains paused. No push, merge or tag occurred.

Primary final evidence: `artifacts/round10/ROUND10_FINAL_STATUS.json`, `T6_FINAL_REGRESSION.json`, `RESOURCE_RETIREMENT_RECEIPT.json`, and `docs/reports/2026-09-26-round10-final-zh.md`. Live 129/130 Layerwise boundary, long context, concurrency and deployment remain outside qualification.

---

## 2026-09-25T18:30Z — Plan13 final: COMPLETE_WITH_NO_PROMOTION

R2 and R3 are terminal `NO_COMPONENT_GAIN`; R4 is `DEFERRED_NO_CURRENT_OVERLAP`. No Layerwise update/transport candidate is promoted and all remain default OFF/deferred. Independent threshold v89/v90 qualification completed: threshold64 Layerwise 64-token E2E samples708.326/702.309/694.657s versus threshold129 Hybrid6.131/6.116/6.149s. All path, output, repeat, resource and cleanup gates pass, so threshold129 is `HYBRID_THRESHOLD_BOUNDED_QUALIFIED` under the narrow frozen single-request contract. This path change is not Layerwise prepare acceleration.

v89/v90 each used one model start and seven generations, 0abort/0retry. Both supervisors PASS, guard ACK and containers removed. A post-v89 admission failed closed at NUMA3 MemFree17.42GiB before v90 launch; targeted retirement touched only unreferenced project tensor file cache, never global drop_caches or external tasks. Fresh admission-21 then passed and v90 ran once. Final physical12 has no task process; NUMA2/3/4/7 MemFree230.33/229.99/156.99/224.26GiB. Plan15 totals are8 component processes/144prepares and4 model starts/16generations. Automation remains paused; no push/merge/tag. Final evidence root: `artifacts/round9-fixed-shadow32-update-20260925/PLAN13_FINAL_REPORT_ZH.md`.

## 2026-09-25T16:58Z — R3 NO_COMPONENT_GAIN; R4 DEFERRED; threshold next

New admission-18 passed after resource recovery and launched v88 OFF exactly once. v88 PASS, 22 prepares, exit0, guard ACK, container removed; physical12 returned empty. Four-process v85/v86/v87/v88 comparator passes all exact/numeric/identity/mapping/allocation/cleanup checks but returns `NO_COMPONENT_GAIN`: high-reuse prepare ratios0.84797 and0.94286 both miss the frozen <=0.80 gate, and true low-reuse lifecycle first block1.05657 exceeds1.05. R3 remains default OFF and consumes no full-model budget. Component budget is now8 processes/144 prepares. R4 is `DEFERRED_NO_CURRENT_OVERLAP`; one-parent mutation cannot overlap its consumer, while independent host/device staging requires a new ownership subsystem without current benefit evidence. Next: independently build/freeze and run threshold v89 baseline64 then v90 candidate129 under fresh admissions. Scheduled automation remains paused.

## 2026-09-25T11:28Z — USER PAUSE / v88 NOT STARTED

Execution is paused at the user's request. Three fresh admission identities (12/13/14) completed the required 2x60s sampling and failed the unchanged continuous CPU idle >80% gate before container creation. Admissions 15/16/17 were interrupted on the pause request and are preserved as `KeyboardInterrupt`; they are not full admission failures. v88 has no evidence directory, component or prepare consumption. Post-pause verification found no controller/sampler/component process, no Plan15/R3 running container and no process on physical12. Actual component budget remains 7 processes / 122 prepares. On resume, use a new admission identity, rerun fresh 2x60s sampling, and launch v88 OFF exactly once only after PASS. Do not rerun v85-v87 or reuse admission-12 through -17.

## 2026-09-25T11:24Z — Plan13 execution resumed at R3 component closure

v86 ON and v87 ON raw run plus admission evidence are now archived locally and pass the frozen strict single-run validator. Together with v85, all three completed runs have exact frozen identities and paired group/sequence outputs, 22 prepares each, allocator cleanup, guard ACK and removed containers. Actual budget is corrected to R2 4 processes/56 prepares plus R3 3 processes/66 prepares, total 7/122. R3 remains 3/4 and benefit UNKNOWN. v88 OFF is still not started. Next action is a new 2x60s admission identity for physical12/NUMA2,3,4,7, followed by exactly one v88 launch only if every resource and identity gate passes. Automation remains paused; v85-v87 must not be repeated.

## 2026-09-25 对话迁移交接

当前完整接手入口：`docs/handoffs/2026-09-25-04-ROUND9_PLAN15_DIALOG_HANDOFF.md`。R2 已 `NO_COMPONENT_GAIN` 收口；R3 fixed-shadow32 slot reuse 的 v85 OFF、v86 ON、v87 ON 已在远端 PASS 并清理，v88 OFF 尚未启动，故当前只能标 `R3_COMPONENT_ABBA=3/4`，不能声明组件收益或晋级。物理12当前只读无设备进程，自动任务仍暂停。新对话先归档/校验 v86、v87 并修正账本，再 fresh admission 后只启动一次 v88；禁止重跑 v85–v87。

## Latest Plan15 state — 2026-09-25T10:02:50.371717+00:00

R2 ABBA v81–v84 complete and all resources retired: NO_COMPONENT_GAIN, no R2 request promotion. R3 fixed32 same-layer slot reuse implementation now selected; device gate pending. P4a v80 ON gate/capture PASS, old v66 FAIL retained. No task containers running from R2. Automation remains paused. See artifacts/round9-fixed-shadow32-update-20260925/R2_FINAL_DISPOSITION.md and BUDGET_LEDGER.json.

# Latest Plan15 execution — 2026-09-25T09:23:46.858899+00:00

P4a v80 RUNNING on physical12 with background guard; root `/mnt/data1/kt-round9/plan15-p4a-20260925-v80`, controller621549. v74 precontainer receipt failure and v75 observer profile-digest failure retained; v75 consumed1 model start,0 generations and cleanly retired. v80 is one authorized diagnosed script repair; current cumulative model starts2. Details and evidence links in `artifacts/round9-fixed-shadow32-update-20260925/CONTINUATION.md`. R2 components81–84 NOT_RUN. No promotion, no request benefit claimed. Automation PAUSED.

## 2026-09-25 Plan15 常驻授权生效

Plan15已覆盖Plan13人工审批门；当前继续执行，wall-time明确，无需再询问。新独立入口：[执行账本](../../artifacts/round9-fixed-shadow32-update-20260925/EXECUTION_LEDGER.md)。物理12/NUMA2,3,4,7准入01通过，正在刷新启动前准入02；P4a v74准备捕获真实Prefill及ON证据，尚未启动。R2默认OFF本地helper和私有adapter接线已实现，只有host合同测试，没有设备资格。自动任务仍PAUSED。

## 2026-09-25 Plan13：固定 shadow32 更新优化进入只读准备

用户已要求按计划13持续实施；新次数额度独立记账，旧预算不续用，自动任务维持暂停。新入口：[Plan13 CONTINUATION](../../artifacts/round9-fixed-shadow32-update-20260925/CONTINUATION.md)。

v71实际2557项仅可变execution-budget不同于启动前hash，其余2556项匹配；v72实际150项全匹配。镜像torch_npu版本/库hash已恢复，当前batch API文档只声明ND；R2暂作唯一待合同审查方向，尚无行为改动或新设备PASS。

当前等待计划第8节要求的本轮wall-time明确回复；建议组件1800秒、模型21600秒、请求1800秒。新启动/generation/prepare均0。物理12只读proc-mem无进程，尚非CPU/NUMA准入。详见新独立状态与预算账本。

# 2026-09-25 本轮有界验证已收口（不是全Round9完成）

物理12/NUMA2/3/4/7已重新准入。v71 Layerwise回归、Hybrid中途取消、reuse通过；v69 shadow32和v72 shadow320真实组件通过。v70原峰值记账失败保留，修复说明及新独立证据完备。320在14个同输入用例全部更慢，因此NO_PROMOTION，不进全模型；异步按证据依赖保持DEFERRED_NO_PROMOTION_OR_OVERLAP_EVIDENCE，不能称已实现。

所有本轮自有容器已移除且guard退出ACK，最终physical12无任务进程/约3121MiB系统基线。自动任务核实PAUSED，不恢复。38本地测试通过。不要重复启动v68/v71/v69/v70/v72；旧预算不重置。下一项行为变化须独立单变量合同，用户已授权持续开发原则，但不能复活失败候选或掩盖负收益。

完整报告：docs/reports/2026-09-25-physical12-hybrid-workset-round-report.md。证据及最终状态：artifacts/round9-workset-component-20260925/{FINAL_STATUS.json,PAIRED_COMPARISON.json,EXECUTION_LEDGER.md,ASYNC_MIGRATION_DECISION.md,SHA256SUMS.json}。

# 最新执行状态：v71 PASS，shadow32组件 v69 已提交

v71三项真实请求与回收全部通过，见artifacts/round9-physical12-20260924/v71-pass/RECEIPT.md。物理12、NUMA2/3/4/7新两轮60秒准入最低idle99.5907%、各node至少156.97GiB，训练CID/cgroup未变。shadow32组件controller PID504819，快照/mnt/data1/kt-round9/workset-component-20260925-v2，manifest SHA3e1661a2fa0d7818aefb4042cdc30fc1287c6cc497f3b2feda5f1e3534493068。输出evidence/20260925-workset-v69，勿重复启动。v1仅准备版本未启动。v69 PASS+回收后新准入再v70 shadow320，resident始终32。保留原同步，异步仍待真实成本证据。

# 当前：v71 正在执行 Layerwise 回归

物理12、NUMA2/3/4/7完成新两轮60秒准入；最低平均idle99.5619%。v68因allocator白名单遗漏36352而在请求前失败，已完整回收；最小修复及6项本地测试通过。v71 controller PID468361，完整容器ID3368ac758ed72d03354e1ecd2ae6a3cae0a0df00a766743351270ea08d61f60f，已加载并提交64-token Layerwise请求。后续driver自动执行Hybrid取消及reuse，不可重复提交。guard持续，自动任务保持暂停。组件32/320待前置通过及清理后独立新准入。

# 2026-09-25 06:01UTC v68已提交，勿重复启动

用户要求继续；物理12 fresh准入PASS，NUMA2/3/4/7两轮idle最低99.5974%，内存最低157GiB。controllerPID460231。远端/mnt/data1/kt-round9/hybrid-physical12-20260924-v68，输出evidence/20260924-hybrid-physical12-v68；先Layerwise/Hybrid取消reuse，后台guard。工作集仅独立本地开发，未接运行快照。

# 2026-09-25 最新：已恢复授权，CPU准入阻塞

物理12无进程，登记训练容器身份一致，2558冻结文件hash一致。完整2×60s的12窗口CPU空闲率41.63%–43.58%，全机最闲136物理核也仅约46.5%；无法靠换NUMA达到80%。NUMA2/3/4/7 free至少约165–219GiB。v68模型/请求/组件启动仍0，无自有设备占用。详见artifacts/round9-physical12-20260924/resume-20260925/BLOCKED.json。持续授权有效，后续先新准入再按原顺序；不自动恢复定时任务。

# 2026-09-25 已恢复执行

用户明确解除暂停，要求按既定顺序持续完成。正在110.123.0.3物理12刷新CPU/NUMA/HBM准入，训练容器完整身份不变，当前物理12无进程。NUMA候选2/3/4/7。v68尚无启动账本，不重复旧v67。自动任务仍暂停；本线程实时执行。

# 当前状态：用户要求暂停（2026-09-24）

等待机器资源空闲后续接。v68未启动、模型/generation/组件均0，新配置/旧证据已保存。禁止自动恢复定时任务或继续测试。下次恢复从物理12、NUMA2/3/4/7的新资源准入开始，再Layerwise回归、Hybrid取消/reuse、工作集组件及证据驱动异步搬运。

# 2026-09-24 物理12迁移（最新）

用户要求physical12并更换NUMA，顺序为Layerwise回归→Hybrid取消/reuse→工作集组件→有证据的异步搬运。新NUMA2/3/4/7连续两轮60秒已通过，48层MXFP4重新hash通过，新v68快照准备完成。**尚未启动：登记训练HBM实测15499MiB超过15360MiB，按共享合同拦截**。CPU通过不代替HBM准入，未改训练峰值上限。新阶段设备预算0消耗，自动任务保持暂停。

详情：artifacts/round9-physical12-20260924/{STATUS.json,CONTINUATION.md,EXECUTION_REPORT.md,WORKSET_COMPONENT_GATE_CONTRACT.md}。旧v67已回收，不重启旧controller；历史记录如下。

## 2026-09-24 v67结束：Hybrid两请求PASS，资源身份阻塞

最新报告：artifacts/round9-hybrid-threshold-20260924/PHYSICAL_RESULT_AND_BLOCKER.md。未知卡15 PID134438触发guard；原supervisor FAIL_CLEANUP，人工核验后自有容器/cgroup已回收。NUMA6仅15.834GiB<48GiB，外部新增vLLM容器，未修改其他任务。64token回归、取消/reuse未完成。禁止重启v67。新阶段授权继续有效；当前待资源/身份恢复，不是预算待批。定时任务保持暂停。

## 2026-09-24 Hybrid→工作集→搬运阶段已获持续执行授权

最新入口：artifacts/round9-hybrid-threshold-20260924/CONTINUATION.md 和 STATUS.json。
新阶段独立记账，旧预算不重置。v67已提交controller，未验证完成；不要重复启动。共享卡15规则、资源身份门禁及定时任务暂停不变。

## 2026-09-24 最终：资源已回收，终验竞态本地修复，原始FAIL保留

OFF3和ON5请求分别通过，ON原始driver在最后cost JSONL即时读取时报告cost coverage incomplete；随后五条完整记录经离线48层覆盖、互斥成本、epoch、cancel/reuse验证通过。原始driver/supervisor/controller FAIL不改写。prepare占正常请求host95.86%–98.75%，性能优化决定NO_CHANGE，无安全可消除部分的证据。

本地request_driver增加30秒有界证据发布等待，无generation/retry；40项相关测试+4项最终证据负例PASS，修复未部署／设备复验。预算2/2启动、8/8generation、1/1abort耗尽。自有v66容器及cgroup已回收，卡15无进程、整卡2878MiB；训练身份不变。2554冻结文件hash匹配。自动任务保持PAUSED，不要恢复或重启v66。

完整报告：artifacts/round9-request-cost/REQUEST_COST_REPORT_ZH.md。若需要完整driver PASS，需新授权窄ON终验预算；建议1启动/5generation/1abort、0重试/0warmup，OFF不重跑。没有新增性能优化设备预算。未push/merge/tag，旧08/09/M3结论保留。

---
## 2026-09-24 01:46 UTC — ON v66 running; shared admission restored

Latest user instruction restores shared resource admission and cancels automation; automation remains PAUSED. Fresh two60-second CPU/NUMA admission passed, with CPU meanidle96.78% and every candidate node above48GiB. ON v66 started exactly once at01:26, fullCID9e896c572c044bc64c97d0394240d1b8e835e4bb48cbed9a41051a6d76c8686c. Canary and short requests PASS; medium is in flight, then cancel/reuse. Do not restart controller/driver or submit requests. Backend resource guard remains active. Frozen runtime unchanged. Both starts consumed; final generation/abort totals come from remote cost-results and budget. Current host evidence locates cost in prepare, but no optimization selected and no full-run completion claimed. See artifacts/round9-request-cost/CONTINUATION.md.

---

## 2026-09-24 — shared admission attempted; NUMA2 blocked

User restored shared admission and cancelled automation (PAUSED, do not resume). Two60-second samplings complete:12windows CPU meanidle96.7804–96.8030%; NUMA minima2=25.461,3=227.570,6=231.749,7=228.919GiB. Frozen validator rejected NUMA floor before host-affinity/training final checks; those remaining checks must not be labelled PASS. No ON claim/model/generation launched; budget remains1/2,3/8,0/1. Preserve48GiB threshold and fixed profile; no drop_caches or external task modifications. Local on-admission-20260924 contains raw samples, blocked receipt, scripts and log. New UPSTREAM_COST_AUDIT.md and revision-bound upstream files document official mechanisms; no behavior optimization chosen without ON evidence. Fresh full admission is required after resources recover.


## 2026-09-23 OFF v65 verified; ON resource blocked

OFF completed all three requests with frozen token identities and raw-response validation: canary 285.928 s, short 170.655 s, medium 719.043 s. Sampled resource ceilings and logged request lifecycles passed. Supervisor explicitly stopped its container after driver exit 0, received OWNED_EXITED guard ACK, then removed it. Container exit 137 records that intentional stop; it is not a natural exit 0. Remote budget now off_verified=true, model starts 1/2, generations 3/8, aborts 0/1.

At post-run admission, approved training PID 679083 used 15799 MiB, above the unchanged 15360 MiB ceiling by 439 MiB. Full training CID/cgroup match. ON has NOT started. An independent postprocessing assertion initially required an empty card; that is too restrictive for shared mode and was corrected to distinguish owned cleanup from recognized training. This was not a runtime failure.

Next heartbeat: one bounded current training/card/identity check. If HBM qualifies, perform fresh two 60-second CPU/NUMA admission, validate frozen launcher identity, then launch ON exactly once. Do not rerun OFF or lower thresholds. Resource blocking waits for the next scheduled check. No training modifications. Local off-v65 receipts archive results, supervisor, model start, resource summary, verification, card snapshot and finalized budget; large logs remain remotely available. Actual device time/bandwidth/DRAM and optimization selection remain UNKNOWN/PENDING until ON attribution.

---
## 2026-09-23 最新：真实请求成本阶段 OFF v65 已启动

用户批准2model/8generation/1abort，新阶段预算独立于已完成M3。新observer默认OFF，68本地测试PASS。目标110.123.0.3物理15共享，训练fullCID不变。OFF v65完整CID3af298d4c5c71a81d44b8ff04f4b3a0cf0c89a2b055291454c848564c3fbe71d，已消耗1/2启动；由bounded supervisor执行3请求并清理。真实请求成本与优化选择尚未完成。用户要求生成期间不频繁查日志；Round9 heartbeat每30分钟检查一次后续接。唯一当前交接入口：artifacts/round9-request-cost/CONTINUATION.md、STATUS.json、EXECUTION_LEDGER.md；远端运行账本优先，勿重复启动。未push/merge/tag。

---

## 2026-09-23：最终收口（最新）

v64 13/13 PASS；5 canary、2 warmup、6 measurements，short/medium E2E中位数163.697/700.275秒。最终离线证据与身份核对PASS；资源独立回收PASS，训练未变。保留M2数值拒绝和用户随后授权baseline-only的历史；09按该窄范围完成。guard在容器删除后出现STOP_OWNED竞态，已记录PROBLEM_03并完成本地修复测试，未设备复验。prepare计数/TTFT/TPOT UNKNOWN。预算1/1启动、13/13请求耗尽，无新启动。Round9 automation PAUSED；未push/merge/tag。详见artifacts/round9-autorun/ROUND9_08_09_DELIVERY_ZH.md及final-verification.json。

---
以下保留历史记录。

## Latest — 2026-09-22 23:01 UTC baseline M3 five canaries PASS; performance running

Same v64 owned container and snapshot below; model1/1 consumed, no new start/retry. All five raw canary responses replayed offline: marker, arithmetic391, JSON, context quote, uppercase transform. All five retired before successor; prefix cache zero, finite logprobs and response/input-token identities verified. Request6/13 short17-token warmup in flight; medium64-token warmup then3short+3medium samples remain. No performance qualification or final cleanup yet. Actual results authoritative remote m3-results.json; local artifacts/round9-autorun/m3-v64 is a progress snapshot. Keep monitoring; do not restart. M2 rejection unchanged, baseline-only exception user-authorized. Resource/identity guard remains active.

## Latest — 2026-09-22 22:30 UTC baseline M3 explicitly authorized and running

User approved baseline-only continuation after rejected S3. Candidate stays disabled; historical numeric stop does NOT prohibit this newly authorized baseline. One model start used1/1,13requestmax/no retry; do not launch anothermodel. Host110.123.0.3 physical15shared unchangedtrainingCID. Modelkt-round9-full-model-v64 CID06fb8c36ef2061b2be14fb34e7d46490bada87724bbe8e755a61b9252a3dd260. Remote /mnt/data1/kt-round9/evidence/20260922-plan10-m3-baseline-v64 contains m3-supervisor.json/m3-results.json; authoritative requestsledger. Supervisor2196283 waitsreadiness thenruns5canary,2warmup,6measurements; cleansmodeloncompletion/failure. No readinessgeneration/HTTP. Snapshot /mnt/data1/kt-round9/plan10-m3-baseline-20260922-v1 sha bffe0bb4d1d7b46435779183f0dc0e5a0e539e0bb45ef48591a03dbcb30d2500. Runtime/provider/bank/profiles10identitychecks matchv63; profilerOFF. CPU12windowsmean98.16–98.39%idle andNUMA>=48GiB passed. See artifacts/round9-autorun budget/ledger. HeartbeatACTIVE forbaselineonly; Round7paused. No push/merge/tag.

## Latest — 2026-09-22 Plan10 HARD STOP: S3 numeric regression; resources released

S1 audit completed,1/2componentstarts. One private transpose-view candidate tested OFFv24/ONv25 (S3used2/6). OFFpasses08exacthash/oracle/unused22slots/allocationbaseline. ONexit1: caseArelativeL2=0.9835492231451818 >0.01. Plan10 numeric guard stops all further device execution; no S4/M3/model/HTTP. Do not automatically resume baseline qualification after this failure without explicit new authorization. Do not consume remaining repetitions as retries.

Candidate REJECTED_NUMERIC, production/Dynamic copier neverchanged, localenabledhelpernowfailclosedbeforetensoraccess. Original experiment/source retained under artifacts/round9-autorun/rejected-candidate and remote frozen snapshot. 3containmenttests+19existingobserverregressionsPASS. Unique internal NZ corruption mechanism remainsUNKNOWN; metadataalias checks were insufficient.

v23/v24/v25 exited andownedcontainersremoved; no own runningcontainer. Training exactCID/cgroupunchanged, physical15now8968MiB withtrainingPID904871using6126MiB; no task-ownedNPUprocess. ON in-process allocationreturn NOT_VERIFIED, separate from processcleanup. Round9heartbeatPAUSED by numeric hardstop;Round7remainsPAUSED. No push/merge/tag. All budget/results/sourcehashes in artifacts/round9-autorun. M3localpreparation mayexist but NOT_RUN. Currenttasknotcomplete.

## Latest — 2026-09-22 Plan10 S1 PASS; S3 candidate validation started

User approved automatic Plan10 M2/M3. New plan docs/plans/2026-09-22-round9-08-09-autorun.md; authoritative new budget/status/ledger under artifacts/round9-autorun. Old08 finished2/2; do not repeat. Round9heartbeat resumed with Plan10 rules, Round7 stays paused.

S1 v23 exit0, exact A/B outputs/content/cleanup match08. CPU dispatch observes repeated format_contiguous/clone/format_cast/Transpose and per-slot item/copy then format_fresh_view. Installed torch_npu commit912882db... binds op-plugin dedc316...; device subcost remains UNKNOWN. One of2S1starts used, v23 removed. Evidence and receipt in new artifacts.

One defaultOFF transpose-view candidate selected ONLY for private disposable component. No production/Dynamic wiring. Local5contract tests pass, not provider qualification. Independent review added frozen geometry eligibility, unchanged22slot hash and post-mutation failure exit. v1/v2 candidate snapshots never run; reviewed v3manifest996e0fd9243020d3135074348b5e0c58459a1aa2ff829e7811e47391b319edcc. S3 OFFv24 launch requested; verify remote /mnt/data1/kt-round9/evidence/20260922-plan10-layout-candidate-v1/run-ledger.json and container before any repeat. Max6S3starts alternatingOFF/ON, no retries. Both observerON; candidate flag independent. Latest2x60s fixed160physical-core aggregate idle85.34–87.40%, NUMA>=48GiB; launcher still rechecks sharedidentity/HBM. Host110.123.0.3 physical15 shared, fulltrainingCID unchanged. M3 NOT_RUN; baseline profile includes explicitPLEASYNC and BF16_COMPACT, don't silently switch. No push/main/tag.

## Latest — 2026-09-22 08 physical decomposition delivered; 09 evidence audit delivered

OFFv21/ONv22 both exit0, twoA/B BF16shadow cases validated, output/copy identities exact. ON222events VALID_HOST_DECOMPOSITION,prepare1042.999ms/sync747.372ms;deviceNZ832.899ms is separateview ofsamework,notadditive. A/BrelL2 .002422864/.002513433<=.01;allocatedbaseline629325824bytesrestored.2/2componentstarts consumed,0model/HTTP;bothownedcontainers removed;physical15back2878MiB,no process. Training/externaltasks untouched.

See `../../artifacts/round9-shadow-decomposition/DELIVERY_ZH.md`,SHADOW_PREPARE_DECOMPOSITION.md andphysical-20260922/ evidence. Independentgpt-5.6-sol mediumreview acceptedlimitedscope. DecisionNO_BEHAVIOR_CHANGE;nextspecificcandidateNZparentupdate layout necessity,notremoving sync.09originalevidenceaudit+physicalsupplementcomplete;09entireroadmap M2/M3/product/performance stillPENDING,newbudgetrequired. Round9heartbeatPAUSEDviaappafterauthorizeddelivery;Round7stillPAUSED. No push/main/tag. Preservealluncommittedartifacts.

## Latest — 2026-09-22 14:40 UTC resource admission passed; OFF launch requested

Fresh2x60s/12windows fixed160physical cores onNUMA2/3/6/7 conservativeSMT meanidle94.54–98.39%;nodeMemFree>=133.24/198.17/173.33/203.93GiB. Frozen shared candidate_resources preflight passed. No ownactivecontainer/08ledger beforelaunch;v21name available. Requested frozenOFFlauncher --version21; verify remote run-ledger andcontainer before further action. Do not launchduplicate. Evidence `../../artifacts/round9-heartbeats/20260922T144034/`. Existingtraining/externalcontainers untouched.

## Latest — 2026-09-22 14:06 UTC scheduled admission remains blocked

Fresh2x60s/12windows: optimistic best136physical-core idle67.45–73.37%,12/12fail>80%;NUMA4/5 minimumMemFree2.00/1.46GiB. Physical15HBM2878MiB,no own activecontainer/08run-ledger.0/2starts unchanged. Externalstage1rGGUFcontainers and training untouched. LowCPU GDN reuse audit found existing realadapter/pool paths andFP64oracle; historicalv4/v5notcurrentqualification (wrapperhashdiffers). See `../../artifacts/round9-heartbeats/20260922T140604/REPORT.md`. No newnumeric/devicegate. Heartbeatcontinues.

## Latest — 2026-09-22 13:29 UTC scheduled admission remains blocked

Two fresh60s rounds/12windows: optimistic best136physical-core idle upper bound65.97–73.21%, all below>80%;NUMA5MemFree minimum0.92GiB. Physical15HBM2878MiB, no own activecontainer or remote08run-ledger. No component starts (0/2),no model/HTTP. Training and externalstage1r container untouched. Local evidence verifier4negative tests PASS; no numeric qualification promotion. Evidence: `../../artifacts/round9-heartbeats/20260922T132934/REPORT.md`.30-minuteRound9heartbeat remains active;Round7paused.

## Latest — 2026-09-22 ordered low-CPU contract and test preparation

Completed ordered local work: component/state contracts; QSA boundary/mapping test preparation; GDN actual wrapper/pool interface and validation design;09gap supplement. See `../../artifacts/round9-architecture-audit/COMPONENT_STATE_VALIDATION_CONTRACT.md` and `LOW_CPU_PROGRESS.json`. QSA local run:2standard-library fixture tests PASS,3provider numeric tests SKIP because torch is unavailable (NOT_RUN, not numeric qualification). Existing qsa_module_tests already contains a513-block case; remaining gap is current wrapper/pool coverage. No runtime changes,remote jobs,NPU runs or budget expansion.08remains CPU-blocked0/2starts.

## Latest — 2026-09-22 low-CPU 09 architecture/metrics audit

User redirected work to local low-CPU development while cluster CPU resources are occupied. See `../../artifacts/round9-architecture-audit/09_METRICS_AND_OPERATOR_PLAN.md`. Revalidated19matrix rows/19unique saved evidence hashes and generated48per-layer Q32 map/metadata digests; no fresh device qualification. Audited QSA architecture and bounded optimization candidates, with GDN/GR contracts recorded in the same report.08 remains CPU-blocked,0/2physical starts. No SSH/model/NPU/remote jobs, timer changes or push.

## Latest — 2026-09-22 CPU/NUMA reselection measured, no qualifying set

User authorized selecting a different measured CPU/NUMA set. Surveyed all8NUMA nodes in two independent60s rounds (12x10s windows), excluding explicit external-container affinity and its SMT siblings. Even the optimistic per-window best136 different physical cores, taking the better SMT thread and relaxing4-node balance, reached only52.1–59.2% idle after48GiB/node memory screening;12/12windows fail>80%. NUMA5MemFree minimum8.9GiB. No fixed qualifying profile exists in these observations; none published/applied. No thresholds, affinity, training, other containers, cache or timers changed.08 remains resource-blocked with0/2physical starts. See `../../artifacts/round9-cpu-reselection/CPU_NUMA_RESELECTION.md` and `selection-result.json` in that directory. No new backgroundmonitor or retry.

## Latest — 2026-09-22 revised08 implemented, physical stage CPU-blocked;09 audited

Branch `codex/round9-shadow-decomposition`, baseline `a5b26444e`. User approved revised08 and parallel09. See `../../artifacts/round9-shadow-decomposition/SHADOW_PREPARE_DECOMPOSITION.md` and `shadow-decomposition-result.json` in that artifact directory.

- Default-OFF prepare phase observation, captured-input BF16 shadow A/derivedB and guarded OFF/ON harness are implemented in a new snapshot. Existing v63/v20 snapshots unchanged; no behavior optimization.
-63 local tests and78 frozen-image CPU-only tests pass. Five important independent-review findings fixed RED→GREEN. Core AST after stripping observation equals frozen baseline.
-Physical OFF startup denied before container creation: selected160 CPUs onNUMA2/3/6/7 busy; quantifiedmeanidle~40.6% vs required>80%. NPU15 idle does not bypassCPU gate.0/2 component starts consumed,0model/HTTP. No actualphasecost/numeric/physicalOFFON/cleanup qualification.
-09 existing evidence audit complete:19capability rows, explicit current/historical/not-revalidated distinctions andproposedNOT_RUNcanary. No M2/M3/quality/performance extension.
-Remote frozen source `/mnt/data1/kt-round9/shadow-decomposition-20260922-v1/source`; source manifest at parent. Evidence `/mnt/data1/kt-round9/evidence/20260922-shadow-decomposition-v1`.
-Next: re-admit same sharedcontract andrun OFF thenON withimmutable source/capture identities. Namev21was not created; verify availability before future launch. Never silently editfrozen snapshot. Budget2components1200s each,owned4GiB,allocator4%,Host64GiB,workspace512MiB. No extra model/HTTP/warmup/repeats.
-No ownedrunningcontainers orbackgroundmonitor. Training untouched; existingexternalRound7container untouched; timersunchanged. No push/merge/tag. Originalpausedhandofffile untouched.

## Latest — 2026-09-22 KT mainline audit and bounded attribution completed

Branch `codex/round9-kt-mainline-attribution`, isolated checkout `kt-round9-mainline`, baseline `57de93dfbb5775f1265378c1eae737edb8d5cc3e`.
Approved scope T0–T4 + decision + T6 is complete. T5 behavior experiment was not entered.

- v62/v63 normal5→1 and64→4 workloads returned identical token IDs with observation OFF/ON. No speedup qualification is inferred.
- Three normal v63 samples produced242 completed NPU event duration pairs; sampled providers include real GDN/QSA/GR/KTEP paths. Status remains PARTIAL_ATTRIBUTION: worker internals, DRAM bandwidth and calibrated overlap are unknown.
- v20 replayed a real layer0 decode activation through C128 MXFP4 CPU + Q32 BF16 NZ29 NPU. Full-model-vs-replay relative error0; three output hashes identical; mixed-precision oracle checks passed.
- v63 cancellation/reuse passed. CONTROL_SENT→CANCEL_SEEN339.116509s: response remains limited by the in-flight chunk, not low-latency cancellation. No next chunk after acceptance in this probe.
-49 related local tests passed. Default-OFF bounded observation and real replay harness added; qualified layerwise/dynamic/PLE arithmetic unchanged.
-2model starts,7HTTP requests,3profile requests,1component start used. All owned model/replay containers stopped/exited; exact training container remains running. No active training NPU PID was observed this round. Shared contention qualification remains pending.

Read `../../artifacts/round9-kt-mainline/ROUND9_MAINLINE_DELIVERY_ZH.md`, `KT_QWEN_ATTRIBUTION_REPORT.md`, `NEXT_ROUND_STATUS.json` and `NARROW_EXPERIMENT_DECISION.md` in that artifact directory. Remote aggregate evidence: `/mnt/data1/kt-round9/evidence/20260922-kt-mainline-v1`; v63 frozen observation build and separate replay source are recorded by distinct manifests.

No behavior optimization selected. Next proposal: separately attribute BF16 materialization, ND/NZ preparation and copy/retire costs inside expert shadow preparation. It has not been started. No resource lease or background continuation remains. Keep shared-mode admission fresh; old Round7 timers remain paused. No push, merge main or tag. Product quality, formal performance, real long context and deployment remain pending.

## Latest — 2026-09-21 P0 cancellation closure and P1 timeline complete

The approved stabilization scope is complete on `110.123.0.3` physical15 under
the restored shared-mode contract. v58 passed mid-prefill cancel, terminal abort
and same-process reuse while `/mzy-train` had a real NPU child. v61 repeated that
PASS and produced a bounded unified timeline: 2 requests, 2,043 events, zero
drops and three complete scheduler batch pairs. It observes a CPU-gather window,
but no H2D/expert-compute overlap and no repeated complete shadow identity, so no
new PLE or shadow optimization is selected; PLE Async remains default OFF.

v59 and v60 exposed two observer-only startup bugs: binding before private
transport preallocation and mutating a read-only slotted copy facade. Both are
fixed by binding after preallocation through outer forwarding proxies and have
regression tests. Device-gated Layerwise/Dynamic/PLE core hashes remain frozen.
The final v6 build receipt SHA256 is
`0a00966bcd9f96dca6b39e87474ce39fd902d65de46ce03c65c8855fdab250d5`.
The v61 Round9 container was stopped after evidence capture; the training
container was not changed. Product quality, formal performance, real long
context and deployment remain pending and were not run. Read
`62_CANCEL_CLOSURE_AND_TIMELINE_RESULTS.md` and
`ROUND9_CANCEL_TIMELINE_STATUS.json`. Old Round7 timers remain paused; do not
push, merge main or tag from this checkpoint.

## Latest — 2026-09-21 Framework Completion frozen

Final CPU source snapshot v16 passes331 tests and a fresh preserved-baseline
overlay install. Full-model gates now pass F1 Frequency, F2/F3 Layerwise route
telemetry, F4 Prefill-End Dynamic, F5 async PLE, F6 joint host budget, F7 Q32,
F8 actual4k pool sizing and F9 FP8-host-reference/BF16-stage QSA. F10 has an
exact-byte physical15 request-local context-tier component gate; no allocator
adapter or HBM saving is claimed.

F11 v43 passes HOLDOUT2 and two real contiguous streaming chunks. An HTTP abort
acknowledged during long prefill does not promptly interrupt queued chunks, so
mid-prefill HTTP interruption remains explicitly unqualified. v44 passes clean
restart, 391, exact 32+5 chunks and ordinary active abort/reuse. F12 v45 passes
the combined Q32 Layerwise Prefill-End Dynamic + PLE Async request with one
generation commit/final drain, four async drains and391. No Round9 model process
remains on physical15. F12/F13 reports bind the final receipts before commit/push. Old Round7 timers
remain paused; no Round8 performance work, main merge or tag is authorized.
`ROUND9_FRAMEWORK_COMPLETE = YES` under the bounded F0–F13 definition. Product
quality, formal performance, real long-context and deployment remain pending.

## Historical —2026-09-21 MXFP4 NUMA validation and F4 full-model smoke passed

Trial33 preserved a false runtime stop caused by clean file-cache growth. Trial34
uses corrected dual-condition pressure detection and passes on exclusive physical15:
two HOLDOUT requests, four real Prefill-End commits/drains, active abort and state
reuse. Peak owned anonymous memory per NUMA2/3/6/7 is23.086/18.063/15.072/
14.997GiB; minimum conservative available headroom is129.239/148.963/202.084/
200.732GiB. Owned cgroup high/OOM/failcnt remain zero. MXFP4 keeps a48GiB/node
startup floor plus8GiB runtime dual-condition protection; BF16/Q8_0 remain80GiB.
This is one exact short-smoke qualification, not a universal capacity claim.
See57 and the v34 NUMA summary. F5 full-model async PLE is next. No timer restoration.

## Historical checkpoint —2026-09-21 07:19 UTC

Follow-up CPU v6 supersedes v5 as latest tested CPU snapshot:300 tests PASS and
real overlay installer PASS, container exited0, no NPU devices mapped. Added
context component receipt validation and mutation negatives. See the machine-
readable FRAMEWORK_CAPABILITY_CHECKPOINT.json for exact scoped progress.

CPU v5:298 tests PASS; preserved-baseline fresh overlay installation PASS.
Context reference NPU v4:12 synthetic QSA layer cases PASS,355328bytes extra
allocation, BF16 exact output and FP8 decoded-reference exact output. Scope is
component-only; no allocator page was freed. See FRAMEWORK_PROGRESS.md.
F8 pool-measurement binding is implemented, full-model execution remains pending.
NUMA2 is still below80GiB free; no full-model v33 process has started.
No new push/merge/tag or timer restoration. FRAMEWORK_COMPLETE remains NO.

## Resource update —2026-09-21 05:52 UTC

Physical15 became idle and passed fresh exclusive admission for PLE v2 and joint
budget v3. Both component probes exited0 and passed; no shared mode was enabled.
Dynamic Q32 core v19 passes; CPU v4 passes274 tests and real-overlay installation.
Full-model v33 is NOT_STARTED: NUMA2 MemFree is below the unchanged80GiB minimum.
Project-owned unmapped file cache reclamation is exhausted; do not repeatedly
retry without a meaningful resource change or touch unrelated workload/cache.
See FRAMEWORK_PROGRESS.md for current scopes. F4/F5/F6 full-model qualification,
F8–F12 integration and F13 freeze remain open. FRAMEWORK_COMPLETE=NO.

## Historical resource update —2026-09-21 05:04 UTC

Physical15 is no longer idle: new PID3425392 belongs to `mzy-train`.
The exclusive guard stopped our full-model v32 after two successful Layerwise
HTTP requests. Abort/reuse is NOT_RUN. Device work is paused pending fresh
exclusive admission or user-confirmed shared mode and this training's peak.
Do not reuse the historical15GiB peak for this new process automatically.
Continue CPU framework development; no training container was changed.
See FRAMEWORK_PROGRESS.md and47_LAYERWISE_TRANSPORT_IMPLEMENTATION_AND_GATES.md.

## Latest —2026-09-21 Framework Completion sprint IN PROGRESS

User approved the revised F0–F13 plan and continuous execution with autonomous
repairs/documentation. Read43_FRAMEWORK_COMPLETION_EXECUTION_PLAN.md and
FRAMEWORK_PROGRESS.md first for this new sprint. Historical resource matrix
below remains frozen, including BF16Dynamic strictJSON failure.

Baseline commitdc52a63ca was already pushed for the prior documentation delivery.
New sprint changes are local work in progress; no new push/main merge/tag.
Automations remain paused. Exclusive110.123.0.3 physical15 only, fresh admission
per process, no shared training allowance until user changes that instruction.

F0 manifest frozen. F0a contracts and shared routes locally tested. Real F1:
trial24 feature-OFF passes4short requests; trial25 real48-layer route collection
passes2DEV requests/47tokens with identical output IDs and zero logprob difference
versus24. Independent frequencyQ64/Q32 manifests generated and replay checked.
Trial26 Q64Frequency passesHOLDOUT2 and activeabort/reuse; all48actual maps match,
2578resident replacements versusUniform. Trial27 cleanrestart PASS with identical
maps/output IDs. F1 scoped MXFP4/C128/Q64StaticFrequency is FULL_MODEL_SMOKE.
Remote F1overlay/framework source: framework-sglang-v3/python and
framework-dev-20260921-f1-v3 under/mnt/data1/kt-round9. Trial namespaces retain
20260920-full-model-v24..v27 for continuity; actual sample times are in receipts.

F7 Q32 realmodule v18 PASS; trial28 recorderdevice assumption failed beforefirst
forward and was fixed, preservedasFAIL. Fresh29Static/30Dynamic bothpass short
HTTP/actual48physicalmaps/abortreuse. Dynamic4transactions gen1..4. Ownedobserved
peaks25446/26242MiB. See46. Sourcef7-v3/overlay5; bothprocessescontrolledstopped.

F2 real privateBF16transport probev1 neverstarted (newcontainer name missing
monitor allowlist). Added onlytheexactlayerwise namespace;14guardtestsPASS.
Fresh probev2 PASS_LAYERWISE_BF16_TRANSPORT: six numericalcases (max32tokens),
source48identitysweep, privateweight/mapnonalias, abortreuse, constructorpeak
524289536bytes, executionextra616490496bytes. Componentonly; fullmodelpending.
Runtime/profile2/schedulerbinding underreview beforefreshmodelgate.
F5asyncring/F8context hostcode exists, integration/devicepending. F4hostliveprefill
transaction exists, scheduler/devicepending. F6traffic/F9–F13 stillpending.
Readlivecontainer/evidencebeforeanynewlaunch; nohistoricalqualificationinheritance.
ROUND9_FRAMEWORK_COMPLETE=NO. No CMMLU256/formal performance/real64k128k/soak.
Implementation issues and fixes:44_FRAMEWORK_F1_IMPLEMENTATION_NOTES.md.

## Historical —2026-09-21 01:26UTC: full resource matrix complete; jobs stopped

Exclusive110.123.0.3 physical15 full-model C128 matrix completed. BF16static(v10),
MXFP4static(v19),MXFP4Dynamic(v20),Q8static(v21),Q8Dynamic(v22) pass five drivers.
BF16Dynamic(v23) passes resource/system axis (firstHTTP,system8,actual4096,
abort/reuse) but FAIL_CANDIDATE_CONTENT_CANARY: actual fourth answer adds Markdown
JSON fences. First3canaries PASS; all4finite. No stripping/prompt change or false
PASS. Matrixv4 exits1/statusCOMPLETED_WITH_CANDIDATE_FAILURES, as designed.

v23 audit verifies14hashes,18transactions,21440expertcopies,16required requests
drained;128distinct pinned workers. Peak owned41062MiB,board43829MiB,hostcgroup
253952180224bytes. All final evidence collected locally. Delivery index binds
all6trial summaries/audits, audit program,matrix driver/status and device snapshot.

Finaldevice: physical15 no running process,2878MiB baseline. No Round9container,
server,guard or matrix process remains. Unrelatedmzy-train/kt-r7-a3-return-v1 remain.
Localround7 andround9 automation.toml bothPAUSED. No performance/push/merge/tag.
See39_CANDIDATE_RESOURCE_MATRIX_DELIVERY.md for finalscope and outstanding limits.
Quality256,candidate restart,same-placement full-model Dynamic numeric oracle
remain unqualified. Do not restart completed matrix or call allcandidatesPASS.

## Latest —2026-09-21 00:59UTC: Q8_0 Dynamic complete PASS; BF16 Dynamic last

Full-v22 passes all5drivers and strict4canaries, actual4096 and abort/reuse.
Actual128distinct pinned workers, normalstop. Audit verifies14bound hashes,
18transactions/generation18,21740expert copies,all16required requests drained.
Peak owned41060MiB,board43827MiB,cgroup269033795584bytes. Evidence collected locally.
Onlyv23 BF16 Dynamic remains in matrixv4; fresh admission required. Existingv18
strictJSONFAIL remains historical; v23 captures full raw responses before parsing.
Functional/resource scope only; no quality256/restart/numeric-oracle/performance
qualification. Timers remain paused; no push/merge/tag.

## Latest —2026-09-21 00:33UTC: Q8_0 static complete PASS; Dynamic running

Full-v21 C128/Q8_0/static passes all5drivers, strict4canaries, actual4096context
and activeabort+identicalreuse. Actual128distinct pinned workers. Normal controlled
stop; independent post-trial audit verifies13bound evidence hashes. Peak owned
39854MiB,board42621MiB,cgroup268915621888bytes. Evidence collected locally.
Fresh exclusive admission startedv22 Q8_0/request_boundary; thenv23 BF16 Dynamic.
Matrixv4 PID1501832 remains the one-shot runner. No performance or timer restart.
Complete candidates: v10BF16static,v19MXstatic,v20MXDynamic,v21Q8static.
Quality256/restart/same-placement full-model Dynamic numeric oracle remain pending.

## Latest —2026-09-21 00:22UTC: MXFP4 Dynamic complete PASS; Q8 static running

Full-v20 C128/MXFP4/request_boundary passes all5drivers and strict4canaries,
actual4096context,abort+reuse with drained boundaries. Stopped normally.
Independent post-trial audit verifies14evidence hashes,18consecutive Dynamic
transactions,16required requests drained,22072expert copies; no numeric oracle
claim. Actual128 distinct pinned workers. Peak owned41060MiB,board43828MiB,
container memory140692103168bytes. All files collected locally; qualification
scope remains functional/resource, not quality256/restart/deployment/performance.

Matrixv4 PID1501832 automatically startedv21Q8static after per-bank cache advice
and newCPU/NUMA/NPU admission. Thenv22Q8Dynamic andv23BF16Dynamic(rawJSONrepro).
Currently complete full-gate candidates: v10BF16static, v19MXstatic, v20MXDynamic.
Do not rerun these completed candidates. Oldv18BF16Dynamic remains strictJSONFAIL
with incomplete resource axis; v23 will produce independent full raw evidence.

## Latest —2026-09-20 23:55UTC: MXFP4 static full gates PASS, Dynamic next

Full-v19 C128/MXFP4/static passed all5 drivers: firstHTTP,system8 exact repeats,
strict4canaries,actual3840+256=4096,activeabort+identicalreuse. Normal controlled
stop, resource and content axes PASS. Peak owned39854MiB, board42622MiB, host
cgroup115188797440bytes(107.28GiB); actual128distinct pinned workers. Final hashes
independently audited PASS. Quality256/restart/full Dynamic oracle remain pending.

Nextv20 admission correctly rejected NUMA2free69.13GiB andNUMA7free77.20GiB.
Most usage was FilePages; only96 owned candidate low-bit GGUF files received
POSIX_FADV_DONTNEED (no data mutation/global cache flush). All4nodes then>80GiB.
Fresh one-shot matrixv4 PID1501832 now startsv20MXDynamic, thenv21Q8static,
v22Q8Dynamic,v23BF16Dynamic. Same dual-axis rules; each completed low-bit trial
advises only its stopped bank cache before next fresh admission. Evidence for
reclaim and interruptedmatrixv3 retained. No unrelated containers touched.
Log /mnt/data1/kt-round9/candidate-matrix-v4.log. Do not restart finishedv19.

## Latest —2026-09-20 23:45UTC: preserve content failure, separate resource axis

Full-v18 firstHTTP+system8(8/8) PASS, actual128distinct pinned CPU threads.
Known-answer multiply/capital/echo PASS; strictJSON canary parsing FAIL. Old driver
parsed before retaining row, so actual fourth answer is missing; do NOT infer it
from system8's separate fencedJSON responses. v18 stopped; context/cancel not run;
no resource/full-candidate PASS. Its monitor peakowned about41056MiB, no guard stop.

Raw-before-parse capture fixed without loosening strictJSON; numeric_finite now
recorded before content parsing. Independent review confirms35 requires capacity,
system and quality to remain separate. New finalizer reports resource_system_status
and content_canary_status separately. Any content failure keeps overall
FAIL_CANDIDATE_CONTENT_CANARY and nonzero exit; it can never become candidatePASS.
Any resource/guard failure still stops sequence. Classification regressions RED/GREEN.

One-shot matrixv3 PID1463485 starts fresh full-v19 MXstatic, thenv20MXDynamic,
v21Q8static,v22Q8Dynamic,v23BF16Dynamic. Each has independent full drivers/admission.
Only a captured, finite, nonempty content mismatch may continue to context/cancel;
all technical/nonfinite failures stop. Final matrix nonzero if any candidate fails.
V23 will reproduce BF16Dynamic with complete raw-canary evidence. No old result reused.
Log /mnt/data1/kt-round9/candidate-matrix-v3.log; samev8 isolated payload and new
frozen driver snapshots. No timers/performance/push/merge/tag.

## Latest —2026-09-20 23:24UTC: exclusive allocator correction, v18 starting

Full-v13 firstHTTP PASS and two real Dynamic commits, then FAILED allocator OOM:
402MiB allocation rejected by inherited39.5GiB per-process PyTorch cap while
21.72GiB device memory remained free. DockerOOMKilled=false. Matrixv1 stopped;
no subsequent combination launched. Evidence retained, not full-model PASS.

Candidate-only allocator is now42GiB, requires exclusive env+fresh validated
admission. Owned process guard44GiB, admission model45GiB plus8GiB reserve and
actual board baseline. Shared guard remains40.5GiB. Baseline runtime unchanged.
22 relevant tests PASS after RED regressions. Same qualified math adapter/gates;
new isolated candidate-sglang-v8 changes resource policy only.

One-shot matrixv2 PID1399939: full-v18 BF16Dynamic now starting, followed by
v19MXstatic/v20MXDynamic/v21Q8static/v22Q8Dynamic. Log:
/mnt/data1/kt-round9/candidate-matrix-v2.log; evidence/20260920-candidate-matrix-v2.
Stops at first failure; no scheduled tasks resumed. All qualification remains
functional/resource only, pending actual complete results.

## Latest —2026-09-20 23:20UTC: all batch device gates PASS, full matrix running

Expert-v15 BF16 / v16 Q8_0 / v17 MXFP4 all exit0 and pass64expert reverse/restore,
3migrations and9KTEP cases. BF16 observedowned peak2364MiB; explicit ND staging
bound400MiB is not a workspace/allocator measurement. Independent review noP1;
actual full-model resource guard remains decisive. ARM CPU host-v23:29PASS.

One-shot matrix PID1381883 (not scheduled): full-v13 BF16Dynamic currently loading;
then v14 MXstatic, v15 MXDynamic, v16 Q8static, v17 Q8Dynamic. Script:
/mnt/data1/kt-round9/round9_candidate_matrix_v1.py; log candidate-matrix-v1.log;
status evidence/20260920-candidate-matrix-v1/status.json. Stops on any failure.
Each run uses fresh exclusive admission, candidate-sglang-v7, matching new device
gate, five functional/resource drivers, controlled owned stop and final evidence.
No quality256/performance/deployment claim; no timer restart/push/merge/tag.

## Latest —2026-09-20 23:16 UTC: bounded layer migration validation

Exclusive physical15 on110.123.0.3 remains authorized; no training sharing until user changes it.
Fullmodel-v12 completed a real48-layer Dynamic transaction (424experts, generation1),
then a second8-expert transaction. Per-expert NZ resident writes made the first
boundary exceed4minutes. Controlled diagnostic stop; NOT a full-model PASS.
Raw logs/stacks/events/failure.json are retained. No resource guard trip observed.

Correction under test: stream original BF16 into one ND projection buffer per
layer (maximum400MiB device staging, existing64MiB host budget), then one NZ
index_copy_ per projection. All maps still publish only after all layer writes
and synchronization; exceptions still poison the controller until reload.
New device gate must reverse and restore all64 real resident experts exactly,
then pass existing3 migration identities and9 KTEP cases. Expert-v15 BF16 running;
new candidate-sglang-v7 isolated overlay installed. Q8/MX new gates and complete
full-model combinations remain pending. Driver boundary polling now bounded120s
instead of10s; no numeric acceptance threshold or result scope relaxed.

## Actual NZ29 migration fixed; fullmodel retry —2026-09-20 23:07 UTC

v11 fullmodel rejected layout before migration. Formal SGLang enables internal
NZ29 while early probes were ND2. Actual NZ probe reproduced slice.copy_ failure;
fixed by index_copy_ into original resident tensor. Expertv12BF16/v13Q8/v14MX all
PASS exact3 migrations+format29 invariance+real KTEP9cases. Matching gate now
mandatory in server/scheduler, including bank/source/native/adapter identity.
All three formats now have full48-bank SHA/type/source verification completed.
Fullmodelv12 BF16+C128+Dynamic is loading; v10 static remains functional/resource
PASS. See38 for failures and fixes. Corrected cgroup monitor passed real probes.

## Full-model C128/BF16 static trial passed —2026-09-20 22:58 UTC

See38_CANDIDATE_FULL_MODEL_TRIALS.md. v10 passed first HTTP, system8 exact
repeats, four canaries, actual4096tokens, active cancel+state reuse. Only own v10
stopped after completion. Boardpeak42624MiB, owned39856MiB; actual128 distinct
pinned scheduler workers. Exit cgroup telemetry defect documented and fixed for
future launches; inference-time container peak309354479616bytes, failcnt0.
v11 BF16+C128+Dynamic is now running under a fresh exclusive admission.
Q8 all48 static/Dynamic verification complete; MXFP4 conversion atlayer40.
Local implementation/evidence commit8a71454c5; no push/main merge/tag.
Candidate quality256, restart and same-placement full-model oracle remain pending.

## Physical15 exclusive execution authorized — 2026-09-20 22:44 UTC

Latest user reports training has vacated physical15 and explicitly authorizes
exclusive Round9 use until they request sharing again. Fresh npu-smi confirms
no physical15 process; baseline HBM2880MiB. No training allowance is reserved
for new exclusive admission. Unknown external process still stops our own run.
Old shared-HBM15GiB clarification is superseded, not a current blocker.
Q8_0 48-layer conversion complete; MXFP4 conversion continuing CPU-only.
Native low-bit FP32 TP-partial correction rebuilt; old binary strict C64/C128
comparison RED0.002165, corrected Q8/MX four-combination regression exit0.
Q8 new-native C128 full-bank KTEP+Dynamic local gate v7 PASS, actual ND2.
All candidate full-model claims remain pending fresh48-bank SHA/type/source
binding, independent entry-point review, and actual model execution.
Timers PAUSED; no performance/push/main merge/tag.

## Candidate device execution resumed — 2026-09-20 22:16 UTC

Latest user explicitly authorized fresh shared-resource candidate device gates.
See37_CANDIDATE_DEVICE_EXECUTION.md. Host110.123.0.3 physical15, training budget
15GiB retained even though no NPU training process observed at this admission.
C128 four32-worker pools NUMA2/3/6/7: wrapper and real KTEP local merge PASS.
Dynamic processed-tensor migration/local merge probes exited0; v6 measured ND2.
NO NZ qualification. Scheduler hook ordering passed CPU doubles only.
Low-bit24 native tile tests PASS. Earlier C64/C128 MoE tests are provisional:
review found premature BF16 TP-partial rounding; fix and strict cross-pool retest pending.
Both full512 layer0 low-bit banks converted. CPU-only48-layer conversions
kt-round9-q8-all48-v1 and kt-round9-mxfp4-all48-v1 running; do not duplicate.
Lowbit GGUF/KTEP local merge ran, but requires retest after native correction.
Candidate fullmodel and actual model scheduler integration remain PENDING.
Full-model shared-HBM admission awaits whether15GiB includes driver baseline.
Timers remain PAUSED. No performance/push/main merge/tag.

## Candidate host/framework development delivered — 2026-09-20 16:23 UTC

User authorized sequential development after reviewing35. See36 for exact
capabilities, commands, evidence and limitations. New independent modules/tools
implement candidate/C128 contracts, BF16-to-MXFP4/Q8_0 GGUF conversion, bounded
Dynamic state machine and concrete Torch KTEP adapter. Baseline files unchanged.
Local23 tests and ARM CPU-only27 tests PASS; independent review issues repaired.
All containers exited, no NPU devices mapped, no fullmodel/performance test run.
Candidate source lives in /mnt/data1/kt-round9/candidate-dev-20260920, isolated
from preserved v6 runtime. Both automations remain PAUSED.

Important remaining gaps: low-bit320/160 ARM grouped-prefill is explicitly
UNSUPPORTED; Dynamic production scheduler boundary installation is PENDING and
its fullmodel config rejects. C128 native128-worker/device qualification PENDING.
Do not call these candidate-model PASS. Framework host acceptance complete;
resource admission and real device/native tests remain separate.
Baseline CMMLU remains128/256,106 correct; second128 not launched.

## User redirected to development-first analysis — 2026-09-20 15:55 UTC

User prioritizes C128/lowbit/Dynamic framework development, deferring tests to
an idle window tomorrow and excluding performance tests. Concrete design and
code-gap analysis:35_DEVELOPMENT_FIRST_C128_LOWBIT_DYNAMIC.md. Candidate
implementation NOT_STARTED by this analysis; baseline preserved.

V6 CMMLU first128 complete106 correct/0 invalid; second128 NOT_STARTED.
Final256 reference remains INCOMPLETE, not a quality PASS. V6 stopped; docker
Running=false/OOMKilled=false/ExitCode137 (bounded stop). Guard exited after
teardown docker-top error; no owned Round9 worker active. Physical15 returned
to8969MiB board HBM with training4069720/6124MiB. Timers both PAUSED.
Do not launch v7 automatically. Retain exact v6 runtime+driver for remaining
baseline slice before candidate quality comparisons. Local evidence v6 saved.

## Full-model resource gate PASS; observerOFF quality trial starting — 2026-09-20 15:10 UTC

Trial v5 completed actual 3840 input + 256 output tokens, finite logprobs,
clean restart4 and post-long-request short reuse4 PASS against v3 compact.
Measured owned process peak39850MiB, board48708MiB, cgroup303213101056bytes,
failcnt0; training PID4069720 remained6124MiB. Evidence: physical15/
20260920-full-model-v5/resource-measured.json. Scope is observed frozen-profile
load and4096 resource workload, not long-context semantic qualification.
v5 stopped intentionally. v6 now RUNNING observerOFF, guard upgraded with
3 stable baseline samples and finally-stop even if logging fails. First HTTP,
current-container restart4, then first128 frozen CMMLU items chained fail-closed.
Quality driver fixed in34efa6551 to bind exact frozen manifest/commit, admission,
image/profile/runtime and current-container restart receipt before scoring.
Initial client command used a nonexistent /workspace path; no request executed;
corrected to mounted /round9/src path. Both schedules remain PAUSED.
Do not launch duplicate server. Remaining: observerOFF, two128 quality slices,
raw audit, final cleanup and record. No quality score yet.

## Actual4096 request decode in progress — 2026-09-20 15:01 UTC

Trialv5 firstHTTP andcleanrestart4 PASS againstv3compact; revisedinternal
4098context/4100pool admittedactual3840input+256output. Prefillcompleted,
decodecurrentlyrunning withmemoryobserverON. NofinalboundaryPASSuntilresponse.
Hostguardv5active, training6124MiBstable, ownpeak39850MiB atlastsample.
Furtherguardfix(logfailurefinallystop+3stablebaselinesamples)6testsPASS isLOCAL,
preparedforv6; do notclaimdeployedtov5. v6 plannedobserverOFF/restart4, then
CMMLUfirst128; v7 sameprofile second128. Frozen256order preserved, noquality
scoresyet. Runtimeidentity+profile/image/observerOFF boundinslicesforaggregate.

## Live trialv5 — 2026-09-20 14:54 UTC

Trialv4 stopped duringconstruction by non-atomicboard/process subtraction.
Training6124MiB unchanged; board28592-owned10864 falselyread17728nonowned.
Guard repaired:prestartsystembaseline+trainingPID estimate, sameimmediate
absolute board55.5GiB/owned40.5GiB limits;4regressiontestsPASS. See33.
**kt-round9-full-model-v5 RUNNING**, corrected4098context/4100pool foractual
4096workload, observerON, liveguardarmed. NeedfirstHTTPready,cleanrestart4
againstv3compact then3840+256boundary. Do notstartduplicatefullserver.
V3compact32/cancel/canariesPASS; CMMLU256frozen NOTSCORED;observerOFFpending.
BothschedulesPAUSED;training4069720untouched.

## Full-model retry with effective4096 slots — 2026-09-20 14:51 UTC

Trialv3 compact32, activeabort+statereuse and fourknownanswercanaries ALLPASS.
OwnedHBMpeak39850MiB; torchpeakallocated40705135616/reserved41502638080bytes;
cgrouppeak334439145472bytes,failcnt0. Exact3840+256 request rejected before
execution duef02guard slots; NOTcapacityPASS. See report32 for formula.
Trialv3 stopped intentionally; allfinalevidence saved. **kt-round9-full-model-v4
RUNNING** with internalcontext4098/tokenpool4100 for actual4096workload,
same321MiB roundedstatebudget. Liveguardarmed; observerON. NeedwaitHTTPready,
cleanrestart4 againstv3compact, thenactual3840+256context request. Afterwards
observerOFF comparison and CMMLU256reference remain. BothschedulesPAUSED.

## Full-model compact32 PASS — 2026-09-20 14:48 UTC

Trialv3 remainsrunning with liveguard. compact-v1 PASS32/32, fourprompts,
all repeated/prefix token trajectories identical, finiteoutputlogprobs.
Protocolsha f31bddd6d1b37bf5b53e162dce61eb4abd1aa91181c59e9793bcdb125c1e8831.
Actualactivecancel/state-reuse probe running, then3840+256token resource request.
CMMLU manifest frozen incommite66726f2f;ackwritten;NOTSCORED.
No restart/observer/4096resource PASS yet. Do notduplicatefullserver.

## First real full-model forward PASS — 2026-09-20 14:44 UTC

Trialv3 full load and actualHello+4decodeHTTP PASS, finiteoutputlogprobs.
Round9 compact32 RUNNING: /out/compact-v1 (fourprompts,repeat/prefixtrajectories).
CMMLU256 originalordereditems retokenized with frozenQwen3.8source;manifest
88efae7d68e7edf644add12909db89712dc001046b8624801dbb8ce7a8804858,
225–722prompttokens. FROZEN_NOT_SCORED. Full peakworkload notyetqualified.
Hostmonitor remains active; bothautomationsPAUSED. Continuecompact results,
then4096resourceboundary/cancel/restart/observer beforequalityreference.

## Live full-model retry — 2026-09-20 14:41 UTC

Commit4abb23f0d fixes real Ascend pool dispatch; RED-v3 reproduced wrong pool,
GREEN-v1 passed Qwen4 hybrid12layers plus ordinaryAscend48layers control.
**kt-round9-full-model-v3 is RUNNING** on .3physical15 with monitor armed,
logs /mnt/data1/kt-round9/evidence/20260920-full-model-v3. HTTPfirst probe waits
for readiness then Hello+4token fullforward. Monitor now recordsactualcgroupv1.
Optional ROUND9_MEMORY_OBSERVER=1 records perforward torchalloc/reservedpeaks.
No fullforward PASS yet. Compact32 client prepared but notstarted. Do not
startduplicateownedserver. BothschedulesPAUSED;training4069720untouched.

## First full-model load result — 2026-09-20 14:33 UTC

Trialv2 exited; no Round9 server remains. Original131shards loaded,48native
CPU backends initialized, HTTPready. Loadelapsed146.25s, loaderreported37.92GB,
observed ownedHBM peak39932MiB and board48851MiB. First HTTPforward FAILED:
Ascend general pool branch bypassed Round9QSAKVPool, missingQSA APIs atlayer3.
Fixing dispatch and adding actualinit_memory_pool regression before retry.
Monitor conservatively stopped during scheduler failure cleanup: process rows
vanished before boardHBM fell, so nonowned subtraction briefly read48851MiB.
This is a teardown accounting mismatch, not observed training growth or OOM;
Docker OOMKilled=false. Keep original raw log. Full forward/peak qualification
still NOT_PASS. Timers remainPAUSED. Source runtime edits for retry in progress.

## Live full-model trial — 2026-09-20 14:30 UTC

User15GiB envelope recorded. Five real decoder layers0/3/23/24/47 PASS45runs.
Production owned CPU staging and direct KTEP registration qualified. See31.
CPU admission fixed to distinguish160logical visibility/80physical capacity from
72quota and64workers+8aux. Static budget only, full peak NOT_MEASURED.
First real SGLang trial **kt-round9-full-model-v2 is running** on .3 physical15;
monitor armed before start, logs /mnt/data1/kt-round9/evidence/20260920-full-model-v2.
Trialv1 never started: host lacks PSI file; fixed optional telemetry with explicit
UNAVAILABLE status. No full-model loading/forward PASS yet. PyTorchcap39.5GiB,
owned budget41GiB, early monitor40.5owned/55.5board, nonownedboard15GiB.
Training4069720 untouched. Both timers PAUSED. Inspect current owned run before
any retry. User endpoint remains full-model testing, not this static admission.

## User resource answer — 2026-09-20 live continuation

User supplied **15G** for the requested verified maximum shared-training HBM,
including validation/save stages in this round. Conservatively interpret15GiB
(15360MiB). Receipt: evidence/physical15/20260920-admission/user-training-envelope.json.
This is user-provided input, not an assistant-measured peak or enforced reservation.
With existing8GiB reserve, Round9 has at most41GiB owned peak on64GiB physical15.
Resource envelope is no longer awaiting that numeric input; full admission still
requires actual component/load/workspace accounting and remaining runtime gates.

Layer0 real GR/GDN/router/TopK/KTEP/shared gate PASS9runs, crosspath output/state
relL2=0, peak1210348544bytes, exit0. Extending to QSA/middle/late layers now.
Formal owned CPU staging and direct KTEP registration fixes under verification.
Both timers remain PAUSED. No full model executed yet.

## Live continuation — 2026-09-20 13:57 UTC

Formal f02 model initialization now passes original model_path and binds owned
synchronous PLE reader after strict load audit. Red reproduced missing binding;
green metadata/row read + adapter-v5 all36 PASS. Shared expert/final mixer
report30 PASS. Actual layer0 GDN+GR+router/TopK+KTEP+shared integration probe is
being implemented. Latest physical15 training is active (~8969MiB HBM observed);
current idle assumption is invalid. Full model NOT_RUN, resource envelope still
unresolved; both scheduled tasks PAUSED. Last commits c08d35754/fbc24db89.

## Live continuation — 2026-09-20 13:47 UTC

Pool boundary8 PASS; real GDN/PLE/QSA-indexer/full-attention adapter v4 36 runs
PASS. Full attention FP64 oracle max relL2 0.00404127131, peak370674688 bytes.
Failures v1 alias load and v2 missing ForwardBatch argument retained, fixed in
fixture. See report29. All48 banks ready; KTEP Q64/C64 local gate PASS.
Shared-expert/final-GR probe in progress. Full model NOT_RUN; concurrent training
envelope unresolved. Round7/Round9 timers PAUSED.

# 2026-09-20 13:36 UTC — banks and local integration gates

48/48 original BF16 banks completed,73728 exact projection instances,225GiB
plus GGUF headers. Catalog .3:/mnt/data1/kt-round9/banks/all48-v1/catalog.json;
flat relative symlinks are usable through /round9 container mount. CPU jobexited0.
Real KTEP+nativeNPU64resident+CPU C64 local gatePASS9casesx3,maxmerge0.0006974445,
peakNPU1,048,747,520bytes; see28_KTEP_LOCAL_MERGE_GATE.md,commit7c92740f6.
C64wrapper/native repeated-route repair committed32e46aa62.
SGLang original48layer meta structure and actual loader mapping PASS:1658source
keys,1094destinations,128PLEparts,36FP32A_log. Negative missingQ/shiftedPLEboundary
rejected. Fixed logicalPLE rowcount vs config-aligned storage padding (90rows).
No full checkpoint/model allocations were done by this metadata-only fixture.
Real pool tests7PASS and model isolated contracts8PASS; pool last-slot boundary
and memory-saver-OFF followup currently running. Source overlay WIP is not full
model qualification. Agent prepares real adapter chunk/decode/reset fixture.
Full model admission/tests remainNOT_RUN; shared-training envelope unresolved.
Both schedules remainPAUSED. Inspect live owncontainers before resuming jobs.

# 2026-09-20 13:20 UTC — live continuation

Schedules round7/round9 rechecked PAUSED. Host .3 physical15 unchanged.
Actual BF16 C64 wrapper plus NPU transfer PASS:9cases (T1/10/32),3 bit-identical
repeats each,maxrelativeL2=0.0001138027073,after allocated/reserved=0.
Fresh native route-position repair fixes repeated-expert single-token weights;
previous FAIL16.1% preserved. See27_C64_WRAPPER_BANK_GATE.md. Not full KTEP merge.
PLE real f02 request/state pools:alloc/free/reuse,fork/clear,abort paths3PASS.
QSA real NPU page/ring/layer isolation + invalid page tests2PASS; runner integration
and model-level chunk/decode still pending. Source overlay is unqualified WIP.
CPU-only remaining layer banks job:kt-round9-bf16-bank-remaining-v2,CPU2/RAM12GiB,
no devices; sequential layer1..47,per-layer exact1536projection verification.
First bank-all attempt failed before output because Torch autoloaded NPU backend;
v2 sets TORCH_DEVICE_BACKEND_AUTOLOAD=0 as in successful layer0 job.
Inspect live job/evidence before launching anything duplicate. Full-model NOT_RUN.

# 2026-09-20 — Live execution through full-model testing

Latest user instruction: PAUSE round9 scheduled heartbeat and continue actively
in the current session, monitoring resources and fixing issues through full-model
testing. Prior stop-at-admission scope below is superseded. round7 stays PAUSED.
Do not recreate/resume a schedule. Active host .3 physical15, other constraints
unchanged. Full-model admission remains a prerequisite, not the new endpoint.
Synchronous PLE local gate PASS: see23_PLE_LOCAL_GATE.md, local commitd6e4638c8.
QSA strengthened device-v2 30cases PASS including real index numerics and signed
head ranking. GDN device-v4 18cases PASS after bounded private CPU staging
repair; prior transfer hangs preserved (see24_GDN_LOCAL_GATE.md). All are local
module gates, not full model. NPU image uses per-container OMP/BLAS threads2,
RAM8GiB, CPUquota2.
New Ascend-native extension built successfully in .3:/mnt/data1/kt-round9/build-ascend;
KT/KTEP imports and Qwen4 config PASS in runtime-v4; fresh Ascend callback/transfer 5tests PASS. Exact f02 retained in
sglang-f02-recovered; owned sglang-round9-port receives minimal config patch
and complete four-file KTEP overlay. Model port is in progress, not qualified. See25/26.
No owned jobs active at the12:33UTC resume inventory. Follow-up bounded GDNv3/v4
runs then started in current session; inspect live containers before new tests.
User question about full-model training peak/window remains unanswered;
current idle telemetry alone cannot qualify full-model loading.

# 2026-09-20 — Continuous progression authorized through full-model admission

User requests monitoring, continued development and fixes until full-model
resource admission. Thread heartbeat `round9` is ACTIVE every30minutes;
old `round7` remains PAUSED. Do not confuse those automation IDs.
Read `20_FULL_MODEL_ADMISSION_TRACKER.md` first, then the latest module reports.
Use gpt-5.6-sol / medium for delegated agents. Active target remains .3 physical15.
Before remote work check for existing owned jobs; do not duplicate active runs.
At successful full-model admission, report and stop this automation's work;
do not infer authorization to run full-model quality/performance from this scope.

# 2026-09-20 — RESUMED: physical15 shared development on 110.123.0.3

User explicitly resumed Round9 and resolved host ambiguity: use 110.123.0.3,
share physical15 with existing training. This supersedes the pause and .22 host
selection below. Preserve .22 uncommitted work; do not overwrite or discard it.
Any delegated agents must use gpt-5.6-sol with medium reasoning.

Fresh isolated ACL mapping probe PASS: only /dev/davinci15 exposed, one device,
user/logical0 maps to physical15, aclInit/aclFinalize return0. Host remains
UNION_POLICY and device-share=False. Earlier npu-smi -9005 is not sufficient to
reject ACL access. Bounded arithmetic and256MiB/1GiB/2GiB allocation/compute/free PASS.
GR real-weight local48-config probe PASS; full-model remains NOT_RUN.
Read `19_SHARED_PHYSICAL15_RESUME.md` for scope and remaining gates.
Evidence: .3:/mnt/data1/kt-round9/evidence/20260920-shared-resume/.

# Round9 continuation checkpoint — 2026-09-19

## 2026-09-20 — PAUSED_BY_USER

User explicitly requested pause. No Round9 jobs or agents may resume without
an explicit user resume. Read `docs/handoffs/2026-09-20-04-PAUSED.md`.
New uncommitted implementation and evidence exist ON .22; do not overwrite
them from this local checkout. All owned jobs stopped/exited; model data kept.

## 2026-09-20 — active host changed back to 110.129.0.22

Latest user instruction supersedes the physical15/A3 host selection below.
All subsequent development/testing belongs on `.22` through `110.120.0.3`.
Preserve `.3` model, migration support and `/mnt/data1/kt-round9` in place.
Read `docs/handoffs/2026-09-20-03-RETURN_TO_22.md` (repository-relative).
Fresh inventory: 8 idle 910B3 devices, no NPU processes. Initial device7 maps to
logical npu:0 in a single-device container. Bounded tensor/runtime probe PASS;
no Round9 model qualification claimed. `.3` maintenance question is superseded.
Source/architecture port, host-specific admission and CPU rebuild remain needed.
Existing approval and gate ordering remain valid; no automation resumed.

## 2026-09-20 — first approved execution batch: bounded CPU PASS, NPU blocked

Read `18_PHYSICAL15_EXECUTION_PROGRESS.md` and
`evidence/physical15/20260920-admission/summary.json` before continuing.
User approval remains valid; do not ask again for routine development steps.
ARM CPU-only BF16/C64 320+320 tail and FP32partial fix verified:15 final
profiles/90shape cases,3 identical repeats each,maxrelativeL2=0.00020935924.
One layer's512expert backing loaded;10selected real experts including0/511
numerically tested. Synthetic activations only; no model/NPU qualification.
35 local tooling tests PASS including review fixes. Native binarySHA
f2695802dbd0196aeb644286ce4b02d641200e16b135fb248a008b0ecf1eafbe.

Physical15 hostpolicyUNION_POLICY and device-shareFalse; isolated nonprivileged
single-die npu-smi query returned-9005. Access/logical mapping NOT_VERIFIED;
no NPU tensor/model execution or global mode changes. User has been asked to
arrange a maintenance window or retain CPU-only development; no answer yet.
All bounded CPU tests/observation jobs exited. No new persistent automation.

Live checkpoint structural censusPASS; mainrouted225GiB,PLEtables~95.368GiB.
SGLang gitlinkf02f304 unavailable from configured/officialremote and old.22tree.
Official541ddc37 fetched to/tmp/kt-round9-deps/sglang for audit ONLY; do not
silently replace gitlink or borrow old runtime qualification. Source/architecture
port and NPU admission remain prerequisites for subsequent full-model work.

## 2026-09-20 — user approved physical15 plan; execution started

User explicitly approved the detailed plan and requested execution. This
supersedes the review hold below. Stage0 started in the existing isolated
Round9 worktree. Native inline execution; no automation resumed.
Remote independent device query (no tensor/model run) returned-9005. Host
reports multi-die-policy UNION_POLICY and device-shareFalse. Investigate these
prerequisites without changing global modes or the active training container.
CPU-only environment/source work continues; model execution remainsNOT_RUN.

## 2026-09-20 — physical15 execution plan awaiting user review

User selected110.123.0.3/Card7/chip1/physical15 and explicitly requested the
detailed Round9 plan BEFORE execution. Read
`docs/superpowers/plans/2026-09-20-round9-physical15.md` (repository-relative).
Status: DRAFT_AWAITING_USER_REVIEW. Do not start development, conversion or
test processes until the user reviews/approves this plan. This update records
planning only; MODEL_EXECUTION remains NOT_RUN, destination identity remains
PASS, and shared-device/full-model admission remains unqualified.

Proposed order: shared access/resource/runtime admission → source semantics →
bounded BF16/C64 and format probes → QSA/GDN/GR/PLE/KTEP modules → Q64 static
unquantized-storage reference → MXFP4 with Q8_0 fallback → same-precision C128
→ PLE async/joint budgets → Frequency/Dynamic and independent Q32 → context
tiers → controlled Round9 performance and final per-profile audit.
Keep round7 automation paused; no Round8 performance revival, push/main merge
or tags. Do not alter the active training container or global device mode.

## 2026-09-20 07:20 UTC — destination identity PASS; card7 sharing assessed

Read17_CARD7_SHARED_TEST_ASSESSMENT.md. A3replication-state is
REPLICATION_IDENTITY_PASS. Destination receipt confirms144files/131shards/
1658tensors/359999963128payloadbytes and exact frozenmanifestSHA; all receipt
file statuses/names/sizes match manifest and published directory sizes.
No redundant full-weight hash run; model execution remainsNOT_RUN.

User proposes sharing card7/physical14–15 with active training. Both dies have
device-shareFalse; chip1 has0vNPU. Physical15 process2657094 belongs to privileged
mzy-train container exposing all16devices. HBM usage~13–14%, variableAICore;
hardware headroom is not reserved/isolation-qualified. Tentative targetphysical15
for2–4GiB bounded operator probes AFTER supported access/isolation is established.
No device mode changes, training modifications or NPU test launch in this audit.
Shared correctness evidence must be labeled; formalperf/fullmodel admission
still separate. Old blanket exclusive requirement is relaxed only for assessed
bounded correctness work, not for performance claims or uncontrolled sharing.

## 2026-09-20 — A3 development resources supplemented

Read `16_A3_DEVELOPMENT_RESOURCE_MANIFEST.md`. In addition to the background
model replication, A3 now has a verified Round8 compact handoff archive and the
final `.22` source download state/manifest/verifier/identity receipt. Existing
Round9 archive and full Git bundle remain verified. Large Qwen3.5 weights/GGUF,
download pilots/caches and the unqualified old runtime image were deliberately
not copied. The latter is frozen by digest only and may be exported after R9-A
selects an actual Qwen3.8 runtime base. Model replication remains automatic and
must reach `REPLICATION_IDENTITY_PASS`; A3 model execution is still `NOT_RUN`.

## 2026-09-20 — authorized post-download A3 replication

User requested one verified copy under110.123.0.3:/home/admin/models and the
resources needed for a possible later development migration. Destination has
~1.03TB free at admission; source/destination direct SSH is available. Planned
final path is /home/admin/models/Qwen3.8-Flash-Next/<revision>; transfer stage is
a hidden .<revision>.partial sibling and must never be used as a model result.

The A3 watcher waits for .22 DOWNLOAD_IDENTITY_PASS, then rsyncs the immutable
model over the private path, reruns the complete frozen-manifest SHA/tensor census
on A3, and only then atomically publishes the final directory. Its state and final
receipt live in <revision>.round9-support. A tracked Round9 support archive and
git branch bundle are also copied there. Model execution on A3 remains NOT_RUN.

## 2026-09-20 03:17 UTC — proposed development host110.123.0.3

User prefers subsequent Round9 development on110.123.0.3. Read
14_HOST_123_RESOURCE_PROPOSAL.md for read-only inventory and estimated budget.
320physicalcores/640logical(SMT2),8NUMA,~2TiBRAM,~3.2TiBfree/mnt/data1.
16x64GiB NPUdies are currently ALL occupied; exclusive NPU admission pending.
Proposed default:64expert+8auxiliary physicalcores,640GiBRAMreservation,
1.2TiBdisk,one64GiBdie; provisional limits await census/loading measurements.
SVE/BF16 CPU flags and SMT/NUMA topology differ from.22: requalify native path.
No migration/copy/reservation/launch performed; .22download and paused heartbeat
unchanged by this planning pass. Earlier.22-only restriction is no longer the
user's target preference; new-host actual resource/software gates still apply.

## 2026-09-20 02:26 UTC — download timeout repair and direct mirror selected

Latest user authorized download repair/speed optimization and a bounded .3 speed
test. See 13_DOWNLOAD_TIMEOUT_REPAIR_AND_ROUTE_SELECTION.md and
evidence/download-slowfix-mirror-20260920.json. This supersedes v3/v4/v5 below.

Active .22 service: kt-round9-range-recovery-v6, PID2594900. Four shards x two
connections, HTTPS hf-mirror.com DIRECT; HTTP/HTTPS/ALL_PROXY empty, NO_PROXY=*.
Same frozen revision and official SHA manifest; same stage/controller lock.
MemoryMax16GiB, swap0, CPUQuota400%, Nice19, idle I/O and300MB/s disk caps.
Old v4/v5 stopped during authorized repair; do not restart older controllers.
Local proxy tunnel remains available but is not required by v6. Heartbeat PAUSED.

Fixes: range hard deadline90->600s plus <1024B/s for60s stall abort; range URL
now honors approved HF_ENDPOINT instead of silently hardcoding official host.
20 tooling tests PASS. Deployed range_recovery.py SHA256:
f1a2f7ca1b1d4499ac2dce87ccd2b127868fe249b07dd516825a6ba3f1dff5b2.

02:23:04->02:26:05 committed saved bytes190754811064->193661071464:
~16.12MB/s, no new failures, complete files62->64; newly published62/63 pass
official whole-file SHA. Full model identity still PENDING. .3 eight-connection
128MiB screen12.61MB/s provides no observed advantage over this .22 window;
no full model download, container or persistent worker started on .3.

Do not infer permanent network recovery or exact ETA from this short shared-link
sample. Current transport no longer needs local proxy for progress. No NPU work,
model precision change, source revision change, automation resume or push/tag.

## 2026-09-20 — user-requested GitHub repository study

Read12_REPOSITORY_STUDY_AND_REUSE.md; immutable file identities are in
evidence/repository-study-20260920.json. Remote usermain remains5feb57f91;
publishedRound8branch is9c1f1dae. OfficialKTmain isf7607c0f. Its commit tree
now independently verifies SGLanggitlink541ddc37 and llama.cppa94e6ff8;
this supersedes the earlier official-gitlinkPENDING note below.

Source audit identified a concrete BF16/low-bit C64 preparation gap: current
moe-tp.hpp admits32-aligned multi-pool slices only for all-F32 weights;
non-F32intermediate640 with multiple pools is rejected. Groupedforward_many
and FP32partial rules also contain F32-only paths. Resolve through bounded
format/shape/merge probes before full conversion, not by changing dtype labels
or deleting guards. Native --kt-method BF16/MXFP4 is not LLAMAFILE storage.

Documentation-only research; existing download/automation/runtime untouched.
R9-A identity/resource/semantic/operator qualification remains incomplete.

## 2026-09-20 01:35 UTC — new-session takeover, R9-A still in progress

Read11_TAKEOVER_SOURCE_PREFLIGHT.md and its two evidence JSON files.
InitialHEAD2aab965f7c9e00051055375114a5f38abba25ccc; cleanRound9branch.
User-specified interactive access through110.120.0.3 then sudo/root then
admin@110.129.0.22/sudo was exercised. Existing proxy tunnel left unchanged.
round7 automation remainsPAUSED; no new automation or scheduled continuation.

.22v3 controller2233665 stillACTIVE,16GiB/swap0,OOM0. At01:34:16:
61/131published size-complete shards176125078536bytes +8992587776 committed
unpublished bytes=185117666312/360000192888 (51.42%). FullidentityPENDING,
no final verification receipt. Actual remote manifest is ROOT/download_source.json.
Three deployed scripts and manifest SHA256 match local;18toolingtestsPASS.

New32MiB range timeouts appeared01:33–01:34 at90s, curl28; failures retained.
Existing bounded retry advanced shards66/67/68 again by01:35. No restart,
timeout/config change or claim of fixed underlying network. Preserve controller.

Recovered clean upstream semantic source4ccff141dbe992794f9da6c3aa23535b4f72000d
in /tmp/sglang-qwen4-audit-20260914;15files checked against committed blobs.
PLE config[2]→layer1 is explained by explicit layer_id+1 source. Packed loader
uses expert axis0, gate/up splitdim=-2; real headers/weights still pending.
kvcache pin541ddc37 committed python/sglang has no Qwen4Exp/qwen4_exp names;
native registration gap, external/Transformers fallback untested. Dirty alternate
worktree untouched. OfficialKTgitlink stillunresolved, no runtime upgrade.

No full conversion/NPU launch/C128 allocation. Next: finaldownloadidentity+census,
independent resources and source semantics, then precision/shape/operator gates.
Round8perf stayscancelled; no push/main/tag; no inherited modelqualification.

## 2026-09-20 01:16 UTC — user-authorized download memory limit increase

Only owned kt-round9-range-recovery-v3 service MemoryMax increased live from
8GiB to16GiB via systemctl set-property --runtime; MemorySwapMax remains0.
Host MemAvailable1036681788KiB (~989GiB) before change. Old cgroup max events
4539,OOM0; anon167776256bytes,file8142893056bytes. This addresses file-cache
reclaim headroom, not a proven network-timeout fix. PID2233665 unchanged, no
restart. If service recreated, explicitly preserve16G; old commands show8G.
All other resource limits and eight-shard/two-range concurrency unchanged.
Heartbeat remains PAUSED. User has not requested automatic scheduling resume.

## 2026-09-20 — user-requested new-window handoff, heartbeat PAUSED

User requested background download continue but scheduled task stop. App tool
confirmed automation `round7` PAUSED. Do NOT automatically resume the old thread
heartbeat. Remote v3 downloader and local proxy tunnel remain active; preserve
both. Next window reads:
- docs/handoffs/2026-09-20-01-HISTORY_AND_REUSABLE_COMPONENTS.md
- docs/handoffs/2026-09-20-02-ROUND9_EXECUTION_HANDOFF.md

These reconcile historical scope reductions, deployed overlays, latest BF16/
MXFP4/Q8_0 precision plan and independent C128 study. Old instructions below
to keep heartbeat active are superseded. No new model test started at handoff.

## 2026-09-20 — authorized production eight-shard / sixteen-connection download

Latest user explicitly authorized8shards x2connections and autonomous recovery.
Active owned unit now `kt-round9-range-recovery-v3`, controller2233665 on .22.
Code41dfcd8a9. KT_SHARD_WORKERS=8, KT_RANGE_WORKERS=2. Oldv2 stopped; do not
restart older units concurrently. Same frozen source/proxy and same stage lock.
18 tooling tests pass. Per-file progress is now `range-recovery-v1/model-*.progress.json`;
old progress.json is stale. Global download-state reflects last active transfer,
not all children. Inspect owned systemd cgroup for all processes.
First eight active shards53–60 each committed64MiB around1789864388; shard52
completed and officialSHA PASS. Native/model qualification unchanged, pending.
Second complete batch finished1789864451.9951 vs1789864388.9713:536870912
additional committed bytes in63.0238seconds =8.52decimalMB/s. At1789864494
unitACTIVE, memory.current3571154944bytes, no reported retry/timeout; shard58
already entered its third committed batch.52complete shards before this wave.
Per-range3attempts; failed shard transport jobs retained for bounded retry waves
at4/2/1shards after current wave ends. Integrity errors do not become transport
retries. Final whole-model audit still mandatory. No unconditional restart loop.
CPUQuota400%, memory8GiB, swap0, idle I/O, disk caps300MB/s unchanged; no
global network changes or other-user task changes. Download remains background.

## 2026-09-20 — user-authorized four-shard/two-connections test complete

See10_MULTISHARD_DOWNLOAD_SCREEN.md and evidence/multishard-abba-v1.json.
Test code0e597f68e;16 tooling tests pass. Four-shard x2 =8 connections
measured8.403/7.827MB/s vs two-connection baseline6.971/7.308MB/s.
Geometric gain13.63%, zero retries, all per-file prefix hashes match.
Benchmark ended; production kt-round9-range-recovery-v2 restarted unchanged
with two connections. Eight-connection production not implemented/enabled by
this bounded test. Preserve shared controller lock; never launch duplicate
downloaders against same stage. No whole-model identity pass inferred.

## 2026-09-20 — reduced range overhead / dual connection screen

Current owned service: `kt-round9-range-recovery-v2` on .22, workers=2.
Old v1 deliberately stopped for the user-authorized bounded comparison; do not
restart it or older HF units concurrently. Same stage `range-recovery-v1`, same
frozen revision and proxy tunnel. No global network policy changes.
Code: 63ceafadb (immediate child completion + bounded ABBA), 4bd8f3f90
(tiny-tail redirect fix). 14 tooling tests pass. See 09_RANGE_THROUGHPUT.md.
ABBA 128MiB per cell: single 5.272/5.245MB/s; dual 6.737/5.774MB/s.
Paired geometric gain 18.60%, no retries, all four byte hashes equal.
Selected workers=2 by predeclared rule, not a dedicated-network guarantee.
The 176-byte shard50 tail exposed curl max-filesize rejecting an intermediate
redirect; transport cap now remains 32MiB, final range/length remain exact.
At epoch1789863353 tail recovered and shard50 entered full SHA verification.
Shard50 subsequently SHA PASS (receipt evidence/range-recovery-shard50.json).
Shard51 progressed335544320 ->738197504bytes between epochs1789863411.3593
and1789863469.2425: effective committed throughput6.956MB/s. Published50shards,
148352668760bytes. Controller2208417 healthy. User additionally asked about
parallel different shards: not yet tested/enabled; current two connections
operate on one shard. A future comparison must fix total connections to avoid
confounding shard scheduling with concurrency and preserve current healthy run.
Whole-model identity remains pending; no conversion/model launch authorized
by these transport results. Preserve failure headers and receipts.

## Bounded range recovery replaces failed HF snapshot downloads

Read08_RANGE_RECOVERY.md. Current owned unit is kt-round9-range-recovery-v1
on .22, script range_recovery.py, code commits9e0616d1a/3f2331e29.
Official proxy tunnel remains needed. Old HF unitsFAILED: do not restart them
concurrently. Preserve original HF incomplete and all earlier pilot failures.
11toolingtestsPASS; exact tail range206 verified. New script validates32MiB
ranges, max3attempts per range, SHA-before-publication, final wholemodel audit.
Recovered missing metadata by frozen SHA/Gitblob. Real shard00048 progressed
from2862612480 to3097493504bytes at1789862291; full-shard verdict pending.
This repairs download recovery semantics, not a proven network hardware fix.
VERIFIED: shard00048 recovered3355443376bytes, officialSHA
dbbf3a6e2b81c8cf9eb152c03e07550741b6e143121fbe579e3b32cdf8836a33 PASS.
Receipt range-recovery-v1/model-00048-of-00131.safetensors.verified.json.
At1789862358 controller2184176 automatically advanced to missing00050;
00049 was already present. Keep healthy range controller intact.

## 23:48–23:50 UTC ModelScope comparison and renewed interruption

User authorized speed test only. ModelScope Qwen/Qwen3.8-Flash-Next master
model-00029 HEAD X-Linked-Etag matches frozen officialSHA40d2f229...03ec.
Master lookup is discovery, NOT a frozen alternate snapshot; no bytes published.
Sequential bounded/discarded tests: MS64MiB10.179093s=6.59MB/s;
HFofficial via proxy64MiB9.579291s=7.01MB/s; MS next64MiB failedcurl92
HTTP2 INTERNAL_ERROR before body; HTTP1.1 retry17.023648s=3.94MB/s,206/TLS0.
No demonstrated ModelScope speed advantage. Shared traffic and downloads
make this exploratory, not an isolated bandwidth benchmark. No source switch.

IMPORTANT current official-proxy-v2 downloader FAILED again at1789861747.6861756.
RemoteProtocolError received2865353719 of3355443376bytes. Prior peer-close
received2870525122, a similar transfer amount; investigate long connection/
range-resume behavior rather than claim stable transport or blindly restart.
Last count47 complete shards139706348856bytes. Preserve incomplete cache and
journals. Wholemodel identityNOTPASS; no conversion/NPU launch.

## 23:39 UTC heartbeat — peer-close recovery

Official proxy v2 had stopped at epoch1789860842 after46 complete shards,
136350905480bytes. Failure was httpx.RemoteProtocolError: peer closed body
after2870525122 of3355443376bytes, NOT the prior TLS badrecordMAC error.
Local proxy tunnel still active, remote~2.5TiB disk and~986GiB available RAM.
Preserve journal and2862612480byte incomplete cache. One same-config service
restart at epoch1789861174 resumes the owned official proxy v2 only.
No source/code/mask/quality changes, no concurrent old downloader. Wholemodel
identity and tensor census remain pending. Do not interpret service restarts
as evidence of a rootcause fix or automatically loop indefinitely.
Recovery verified25s later: partial2862612480 ->3072327680bytes, RUNNING,
controller2165205/child2165219. Latest completed shard00047 size1716681904
passed complete SHA against official manifest:
eb79852b4266e63fc7050e79349ece1eb3160eedb13bdb5f39b4c768e0741a4f.

## Official-source proxy recovery v2 — current owned downloader

Original mirror download failed TLS at43/131 completed shards130962633280bytes.
Mirror-through-proxy v1 then failed HF FileMetadataError (missing Hub metadata),
before payload progress. Both unit journals and original partials retained.
Added allowlisted KT_DOWNLOAD_ENDPOINT selector, preserving proxy environment,
revision/fileallowlist/cache and final SHA/census validation. Red test reproduced
missing behavior; full Round9 tooling suite9/9PASS. Commit2e65f3e40.
Deployed guardSHA ae38169ab632fb7364c67809bd7d7ce04d66f41c7c0fad7dba460d51a23a90e1.
Previous script retained as download_guard.pre-official-proxy.py on .22.

Current owned unit: kt-round9-official-proxy-download-v2 on .22, HTTPS_PROXY
loopback17891, endpointhttps://huggingface.co; same8GiB/swap0,CPU400%,IO300MB/s.
Local user unit kt-round9-proxy-tunnel-v1 provides reverse SSH17891 ->local7890
through authorized jump. It depends on this desktop/proxy remaining online;
no host-wide routes/exports/firewalls changed. Port17890 remains untouched.
Do not restart old direct/mirror units concurrently. No automatic unbounded retry.
New endpoint recovered RUNNING/payload growth; underlying TLS rootcause is not
proven fixed. Final wholemodelSHA still required. Independent existing shard
model-00001 size1040155944 verified against officialSHA
3a092d3a9a542093a9c3524023f05f9e1289ccf23324735ce32ae70bf58700ec PASS;
this is not wholemodel or new transport full-shard qualification.

## Candidate download/storage host110.120.0.3 inspected

User asked about downloading there and mounting into .22. Read-only inspection:
hostname server-00; /mnt/data ~1.5TiB free, root443GiB free; MemAvailable~713GiB.
Only active physical interface eno3np2 negotiated1Gb/s. Existing NFS exports
include /mnt/data/train_data to110.129.0.0/16 and kcc-training-workspace to
110.129.0.0/24; no export/mount/container settings changed, no files deleted.
.22 route to110.120.0.3 still uses its shared enp189s0f0/gateway110.129.0.1.
New-host mirror64MiB probe8.753s=7.67MB/s, TLS verified206; no demonstrated
speed advantage or complete-shard stability qualification. Prefer bounded full
shard+SHA pilot before migration. Network-backed CPU expert paging is not an
appropriate unqualified substitute for local runtime storage.

.22 original downloader nowFAILED at epoch1789859816.7288632, same
SSL DECRYPTION_FAILED_OR_BAD_RECORD_MAC. Journal preserved. Previous RUNNING
records above/below are historical, not current. No blind restart this audit.

## Network diagnostic 23:12–23:14 UTC

Default internet route on .22 uses enp189s0f0, negotiated1000Mb/s full-duplex;
25Gb/s enp195s0f0np0 is not the default internet path. During20s sample
1789859552..1789859572, RX5361553433378 ->5363430342111bytes
(93.845MB/s,750.76Mbps). rx_errors37398879 ->37435506 (+36627).
Later CRC/frame/fifo/missed counters were0, length135; generic error source
unresolved, not proof of physical cable fault or TLS causality. Driverhns3.
Read-only socket/process ownership audit found multiple concurrent curl
download connections belonging to UID1007/session2574, NOT project's
kt-round9-model-download cgroup. Do not stop or alter them.
Bounded tests during ongoing shared traffic: mirror64MiB16.397s=4.09MB/s;
Cloudflare32MiB12.303s=2.73MB/s. Not isolated bandwidth certification.
Shared1Gbps ingress pressure is directly observed; proxy through same NIC
cannot bypass it. Keep owned healthy downloader intact; infrastructure/network
changes or other users' scheduling need coordination, not unilateral changes.

## Local proxy feasibility test — keep healthy download unchanged

User requested investigating local proxy. TLS-verified curl range probes of
frozen model-00029 (outputs discarded, no local model storage) succeeded206.
Local HTTP proxy127.0.0.1:7890 ->official:16MiB/3.058s;64MiB/7.067s
(9.50MB/s). Local proxy ->mirror:16MiB/3.657s. Temporary loopback-only SSH
remote forwarding .22:17891 ->local7890 ->official:16MiB/3.099s;
64MiB/8.669s (7.74MB/s). These bounded probes are not full-shard identity
proof or isolated sustained-throughput benchmarks. No demonstrated speed gain
over current downloader9.44MB/s; keep it running. Probe tunnel PID3277177
terminated afterward; existing occupied17890 left untouched.
Latest completed shards42/131,127607189904bytes; download RUNNING. Proxy
remains a possible alternate transport if failure recurs, not selected default.

## Download throughput check and online quota increase

User requested faster download and ETA. Online systemd runtime quotas changed
only for kt-round9-model-download: CPU100% ->400%, read50MB/s/write30MB/s
->300MB/s both. Verified effective properties; healthy worker was NOT restarted.
Keep memory8GiB/swap0, resource pause guards, TLS checks and official SHA audit.
Serial worker remains because prior serial/parallel TLS failures are unresolved.
Sample epoch1789859230..1789859260: model completed+partial bytes
126535541520 ->126818657040, gain283115520/30s =9.437184MB/s.
Completed shards40/131,122309780240bytes; partials not identity-qualified.
Estimated remaining transport~233.18GB (~6.9h at this short-window rate),
plus full identity audit. ETA conditional on sustained transfer/no TLS failure.

## User-authorized download resume and Round8 publication

User explicitly authorized resuming download and pushing Round8 results.
Original single-worker kt-round9-model-download service restarted; confirmed
RUNNING with controller2124622/child2124635 and actual partial-file growth.
This is a resumed attempt, NOT a demonstrated fix for the historical TLS fault.
Keep original curl pilot failure/partials; do not run concurrent downloaders.
Complete official identity verification remains required after download.

Round8 branch codex/round8-qwen35-prep pushed to panxy1019/KT-ASCEND and remote
ref verified as9c1f1dae92303c1a13078319ab2aa2b5026fdb9e. All255 archived artifact
hashes matched before push. No main merge/tag; Round9 branch not pushed.
Frozen Round8 ancestry includes pre-existing Round9 preparation commits.
GitHub accepted with warnings for two historical Round7 archives over50MB;
do not rewrite frozen history merely to remove these nonfatal warnings.

## Formal Round9 entry — supersedes historical checkpoints below

User explicitly requested formal Round9 entry. Phase: R9_A_IN_PROGRESS.
Branch: codex/round9-heterogeneous-working-set, based on frozen Round8 commit
9c1f1dae92303c1a13078319ab2aa2b5026fdb9e. Reuse the existing isolated linked
worktree /home/admin/Documents/ChatGPT/KT/kt-round8-qwen35-prep to avoid another
checkout on the small local disk; the Round8 branch remains at its final commit.
Automation path remains valid. Do not alter the frozen Round8 artifact set.

Remote inspection at 2026-09-19 22:57:51 UTC on authorized .22:
- 38/131 shards size-complete: 115754320832/360000192888 weight bytes.
  Size completion is NOT SHA verification or model qualification.
- Both kt-round9-model-download and kt-round9-curl-pilot-v1 are failed.
  Pilot stopped at 1409286144/3200016032 bytes with curl exit56 and TLS
  decryption failed or bad record mac. Preserve all partials and raw journals.
- No project model container running; unrelated mindspeed-mm and verl untouched.
- Remote disk about2.6TiB free; MemAvailable1034805008KiB. These totals alone
  do not grant runtime admission. Local disk cleanup leaves about4.7GiB free.

Next: investigate common transport failure without disabling TLS or blindly
restarting either failed service; validate complete files against frozen official
identities. Continue lightweight source/shape/precision preparation independently.
No full conversion, full-model NPU launch or C128 expansion until their gates pass.
Read precision amendment05 and C128 plan06 before backend implementation.

## Round8 revised-scope closure — next authorized stage is Round9

Latestuser cancelled Round8performance testing and requested finalaudit/closure.
Read round8_prep/FINAL/ROUND8_FINAL_REPORT.md and ROUND8_FINAL_STATE.json.
Round8COMPLETE_WITH_USER_SCOPE_CHANGE; formalperformanceNOTQUALIFIED remains
historical, not a pendinggate to resurrect. StaticQ32-C64selected,DynamicOFF.
Proceed independentRound9entry, no inheritedmodelqualification.
Curlpilot failed22:43:43 at offset1409286144(exit56), no whole-shardSHA receipt.
Preserve partials/headers/journal; no concurrent oldHFservice. Downloadtransport
still needs repair. No Round9fullconversion/NPU until identities/admissionPASS.
Localpipcachecleanup restored~427MiB; do not download model locally.

## 22:39 UTC — ONE shard alternate-client pilot running

Read07_DOWNLOAD_TRANSPORT_PILOT.md. Newownedunit kt-round9-curl-pilot-v1
started22:38:40; stateRUNNING,CURL_PILOT_RANGE_0. Preserve if healthy.
Selected model-00029-of-00131.safetensors3200016032bytes, officialSHA
40d2f2291ff4800373087cd00f1f95c3bd69331c32f39cfdd5d43516764b03ec.
Own output curl-shard-pilot-v1; no model publication. ExistingHFunit remainsFAILED.
Localcommit2cdee26d0precededdispatch;8toolingtestsPASS. Guard interlock remains
active, sameCPU/memory/IOcaps, noNPU. Audit entireSHA before extending transport.
Do not infer wholemodelverified from SINGLE_SHARD_PILOT_VERIFIED.

## 22:25 UTC — reservation released; bounded transport probe

Round8diagnostic exited, prep container stopped; PAUSE_DOWNLOAD marker removed
only after checking both. Download service remainsFAILED, not silently restarted.
Official frozen config HEAD on .22 still fails connection(code000,4.21s).
Mirror frozen config HEAD via systemcurl returns200 withTLSverification0.
Systemcurl rangedGET of frozen model-00001-of-00131.safetensors returned206,
exactly1048576bytes in1.431s, TLSverification0. Outputdiscarded; no cached model
artifact replaced. This proves a bounded alternate-client range succeeds, NOT
whole-shard integrity or a TLSrootcause fix. Next use a separately identified,
bounded resume/identity experiment if replacing HF transport; preserve HF partials,
officialmanifestSHA validation, same resourcecaps and pauseinterlock.
Do not disable TLS, change source revision, or run uncontrolled full transfers.
Desktop free disk~12MiB; keep large artifacts on persistent .22, manifests local.

## 22:00 UTC — serial download also failed; no blind restart

Unit failed21:40:26 with the same TLS DECRYPTION_FAILED_OR_BAD_RECORD_MAC,
even withONEworker. Thus concurrency reduction did not eliminate failure.
38completedshards/115754320832bytes last observed; full identity NOT verified.
Preserve partial cache and journal. No certificate bypass/force-redownload.
PAUSE_DOWNLOAD recreated before Round8 single-cell resource diagnostic.
Investigate transport independently; do not resume transfer until diagnostic
ends and reservation is released. Existing revision/resource limits unchanged.

## 21:40 UTC — bounded serial transport diagnostic

TLSReadError/badrecordMAC recurred21:28:52 after38completedshards,115754320832bytes.
Retainjournal; performance ended,notpaused. Downloadguard now usesONEworker,
samefileallowlist/cache/fullrevision,TLSverification andfinalSHAchecks. This is
transportisolation,notaprovenTLSrootcausefix.6localtestsPASS, noforce-download.
GuardSHA62209856e4f6340dbf25c31f41b56dd53df81273f79982c7d19080ec5a66ff21,
matcheddeployed. Failedservice restartedafterexactstatuscheck. Verifyhealth;
do not blindlyrepeat if sameerrorrecurs. Resourcecapsandpauseinterlockunchanged.

## 21:22 UTC — timing ended, download reservation released

All12Round8performancecellscollected, controllerinactive,allownedlanesstopped.
PAUSE_DOWNLOAD removed. HF resumed then failed TLSReadError/badrecordMAC;
sameownedservice restarted withsamepinnedrevision/cache/caps. Preservejournal,
noforce-download,nocertificatebypass. Verifylivehealth. Round8 strictresource
auditNOTQUALIFIED duehostswap/pressure andsamplingbounds,finalclosurepending.
NoRound9NPU/fullconversion yet. Earlierpauseinstructionsbelowarehistorical.

## 18:30 UTC — download paused for formal Round8 performance

Latest: performance-v2 failed cold Dynamic watchdog300; preserved and replaced
by separately frozen performance-v3 with bounded watchdog1800. Reservation
continues through the entire v3 retry; do not resume just because v2 isFAILED.

Explicit PAUSE_DOWNLOAD marker is present; guard status PAUSED and owned HF
process group stopped (T), verified before dispatching kt-round8-dynamic-perf-v2.
Keep paused through all12cells, including between server restarts. Resume only
after campaign ends and timed work is absent.33completeweightfiles totaling
102,539,898,376bytes observed before pause; full identity NOT yet verified.

User also requested consideration of C128. Read `06_C128_EXPERIMENT_PLAN.md`.
Include an independently admitted C64/C128 comparison AFTER precision probes;
no immediate expansion, no alteration of existing C64, no inherited PASS.

## CPU precision replan requested by user

Read `05_CPU_PRECISION_REVIEW_AND_REVISED_PLAN.md` before any full CPU-bank
conversion. User questioned full F32 deployment and requested a new official-
source-backed precision plan. Proposed route: bounded F32 local reference;
probe BF16 weight storage/F32 compute as first unquantized model reference;
MXFP4 product candidate, Q8_0 fallback. No format is yet device-qualified.
Do not blindly start450GiB F32 conversion from older instructions. Complete
format/shape/C64 probes and freeze the new reference identity before model
quality use. Do not relabel BF16 storage execution as F32 execution. Current
source download and Round8 runtime stay unchanged.

## Superseding authorization: download/check/deployment preparation now

User explicitly authorized source-model download, verification and isolated
deployment preparation while Round8 runs. This supersedes metadata-only limits
below and in EXECUTION_PLAN.md, but does NOT start Round9 full-model qualification.

Remote root: `/home/admin/kt-round9-qwen38` on .22. Download source manifest
`manifests/download_source.json` freezes official Hub file sizes and LFS SHA256
or Git blob IDs at de4b8e4d43b917e7706784d8bb445c9af86a3540. Mirror is transport
only. Full131shards plus metadata:360023351514bytes. No precision substitution.

Owned systemd unit `kt-round9-model-download.service` downloads then streams
identity verification. Four network workers, CPUQuota100%, MemoryMax8GiB,
swap0, Nice19, idle I/O, disk write30MB/s/read50MB/s cap. No NPU.
`download-state.json` reports RUNNING/PAUSED/FAILED/DOWNLOAD_IDENTITY_PASS.
Raw output is in this unit's journal; verification receipts are under evidence/.
Service is independent of this desktop session; no auto-delete on failure.

Before ANY formal Round8 performance:
1. Create remote `PAUSE_DOWNLOAD` marker in Round9 root.
2. Verify download-state status PAUSED and child process group stopped.
3. Wait for outstanding disk activity to settle, then resource admission/timing.
4. Remove marker only after the complete isolated performance campaign ends.
The guard also pauses if Round8 prep container runs, MemAvailable<128GiB,
diskfree<150GiB or Docker state is unknown. Only its own process group is signaled.
Do not treat a contended timing interval as isolated if reservation was omitted.

`kt-round9-qwen38-prep` is a separate preparation container, no devices/network,
2CPU/8GiB/no swap, immutable existing image, read-only Round9 mount. Its package
inventory probe exited0; stopped afterward. It is NOT a Qwen3.8 serving image:
installed transformers5.3.0.dev0 vs source config5.8.0.dev0 requires compatibility
work. No Round8 package/image/imported path was modified.

Full conversion and NPU serving remain gated on Round8 closure plus independent
Round9 resource and architecture admission. Current source-download admission
does not imply the roughly545GiB F32+PLE runtime is admitted.

State: PREPARED_WAITING_ROUND8. Read EXECUTION_PLAN.md and latest user input.

1. Continue Round8 from
   `/home/admin/Documents/ChatGPT/KT/kt-round8-qwen35-prep/round8_prep/10_HANDOFF/CURRENT_EXECUTION.md`.
   Preserve healthy .22 heldout-v3 and CMMLU-v2. Audit before isolated formal
   short64/32 + medium256/64 performance; no long requests. Do not revive
   cancelled old performance matrices. No new heavy Round9 work yet.
2. Close Round8 with actual static system/quality/stability/capacity evidence,
   authorized performance and Dynamic outcomes; positive Dynamic/Deferred
   benefit is not required. Commit final report/state and reproducible handoff.
3. Establish new isolated Round9 branch/worktree; record path here. Existing
   `kt-main-integration` branch `codex/qwen38-flash-next-bringup` has dirty
   submodule/untracked documents and MUST NOT be overwritten.
4. R9-A must finish source and resource audit before large downloads/conversion.
   Metadata revision is frozen; weight-header and full tokenizer identities
   remain unverified. Official fork file lookup failed; resolve actual source
   tree, do not treat guessed qwen4_exp.py path as proof of absence.
5. Advance ordered milestones in EXECUTION_PLAN.md. No unearned READY states.

Preparation changed no production runtime or model configuration. Existing
Round8 imported-code/overlay identities remain separate from local gitlinks.
