# Round6.4 execution checkpoint

## Current pointer — supersedes historical entries below

Active: `/workspace/results/r64-quality-dev-v1`, OFF-only8train+8dev, capacity1024,
normalEOS, sourcev2. Do not interrupt/duplicate. Round3anchorv2passed after
retainingCANNPYTHONPATH; v1failurearchived. See11_RETAINED_REGRESSION.md.
Batch campaign v2 is FAILED at b4 OFF repeat1/cohort0/s6_en_01 token55.
Both batch2 profiles completed10repeat+2restart PASS; all4 initialpilotsPASS.
Read09_BATCH_QUALIFICATION.md before acting. Partialaudit325forwards/20requests
PASS integrity; cohort0 actual schedules equal; earliest retainedrouter/logit
difference atposition1. Full diagnostic reproducedsamefailure; Layer7 decoder
inputhidden/residual already differ beforefirstCPU, prefillcommon tensors exact.
Batch4 remains EXPERIMENTAL_NOT_QUALIFIED withpre-L7NPU/KV investigation boundary.
See09_BATCH_QUALIFICATION.md; do not rerunbatch4long/qualityunprompted or refit.
OriginalD1batch1soleprimary continues; not a postfailureprimary substitution.

Future tools deployed separately to `/workspace/results/r64-next-tools-v2`;
fixtures to `/workspace/results/r64-next-fixtures-v2`. Active campaign still
uses capacity-tools-v1; no overwrite. Future wrapper now resolves its own directory.
Staging `/home/admin/r64-next-tools-0tGiCF`; root PTY92973.
53 local tool tests PASS; own-subtree orphan reaping also1A3behavioraltestPASS.
Future v2 OwnedServer now uses new run_owned_process.py wrapper(subreaper),
not the active batch campaign's oldguard. No extra process signals, no productiondiff.

Next: finishOFF-only quality dev8+8 then32+32; inspect format/capacity and
freeze protocol before primary results. HD3/FD3 descriptive_entry.json committed;
retained old quality evaluator and exact historical manifest deployed as smoke.
No primary outputs, primary contract not frozen. Quality/performance/stability
still NOT_RUN. Continue through final ADR; no early final at the batch milestone.

## Approved authority

User approved complete Round6.4 plan and 2pp absolute per-benchmark quality margin.
Read approved plan and `00_BASELINE_AND_SCOPE.md`. No repeated approval necessary.
Old Round6.3 stays REJECTED; do not run its candidate-order evaluator as this round's gate.

## Workspace

- `/home/admin/Documents/ChatGPT/KT/kt-round6-deferred`
- branch `codex/round6-4-official-deferred-approximation`
- start `ba3494d9aedcc0e5623157832134e1e32a8ea44a`
- approval commit `404274e17`; compact tooling commit `c8a08e5f0`
- child `f02f304a25364d1e8fe9c52515a22f20f5089a52`, unchanged
- inherited llama.cpp dirty: preserve, never stage/discard.
- No push/main merge/new READY tag yet.

## A3 current

SSH admin@110.123.0.3; root PTY was available in session 92973.
Only container `kt-r6-deferred-ascend`, NPU0, CPU0–15.
Never stop other users' tasks or mutate host libraries.
Actual checkout+overlay remains distinct from local HEAD (see preflight provenance).
Local SCP staging `/home/admin/r64-stage-4a6dFz` on A3.

Preflight `/workspace/results/r64-preflight-v1`: 19 passed, EXIT_ZERO; archived locally.
v2 accidental duplicate metadata startup cancelled before tests, not evidence.

**Current runs complete; no active model at last check**

`/workspace/results/r64-compact-v2`
driver `/workspace/results/run_r64_compact_controls_v2.py`
driver log `/workspace/results/r64-compact-v2-driver.log`
Sequential remaining P2_B0/P3_B0/D1_NODEFER/D1, 20 requests per profile.
v1 P1_B0 20/20 PASS archived locally; v1 stopped before P2 at EADDRINUSE port probe.
Tool-only fix `3e10a183d` uses SO_REUSEADDR+bind/listen, no REUSEPORT; 5 local tests PASS.
Do not repeat P1 or relabel original v1 INCOMPLETE. No production changes.
Each profile root has guard/result.json, report.json, requests.jsonl, readiness.json.
Driver stops on first failure, guard cleans only owned server.
Top-level progress.json currently written on driver completion/failure, not per request;
inspect driver log and active owned PID before considering missing file a failure.

## Immediate continuation

1. Check healthy run; archive each final report/log/hash. Do not launch competing model.
2. Audit production diff/reference and finish WP0 freeze work concurrently offline.
3. Complete missing R2A/expert safety, nondefer compact and D1 system/no-hit prerequisites.
4. Adapt retained D1 runtime runner to new approved protocol (old r63 runner wrongly
   requires rejected candidate). S6 12×10×64 +8/16/32, restart12×64, observer OFF.
5. Explicit oracle four preselected engineering requests + actual 1000-forward lifecycle.
6. Batch1 runtime PASS then early batch2/4/pilot. No batch>4 or D2/D3 expansion.
7. Freeze GSM8K test full + CMMLU stratified2048, source/revision/templates/parser,
   paired score/exact interval implementation, synthetic tests and power scenarios.
   Contract currently explicitly forbids primary quality until this preparation complete.
   Old context512 insufficient for GSM input plus512 output; explicit equal OFF/ON
   capacity adaptation and validation required, never truncate prompts silently.
8. HD3/FD3 unobserved-input audit; use diagnostics not old logit exact gates;
   proceed quality/performance/stability/ADR per approved plan.

Same-path/system failures halt affected configuration. Finite ON/OFF numerical drift alone
does not block quality. 2pp is approved and immutable; confidence-crossing is INCONCLUSIVE.
No quality/acceleration claim until actual evidence. Default Deferred OFF.

## Scheduling

Existing heartbeat id `round6-3` updated to Round6.4 prompt and ACTIVE, every 10 minutes;
display name `Round6.4 近似加速执行接续`. No duplicate monitor.
Quiet on unchanged health; notify on milestones/failure/completion. Continue useful local
preparation while A3 runs. Pause after scope completion or terminal unresolvable blocker.

Actual compact model_runner SHA (read on A3):
`71b1daaf2a78afde91794963eb5c6a0779bb6ed931841e0a56f8b8bf34ce88d9`.
Deployed compact driver SHA:
`0af222c0b46b6cb44e1faafee3175e4df68d67cea91e254389c5f4acf29f2e9d`.
Previous deployed SHA above is v1, not v2; hash the versioned v2 driver on collection.
Latest status: P1_B0 complete; v2 remaining profiles launched, check before any new work.

## Latest completion update

- v2 completed P2_B0/P3_B0/D1_NODEFER/D1, each20 PASS. Combined with v1 P1:100requests.
- All v2 artifacts archived under round6_4/evidence/r64-compact-v2.
- v2 driver SHA e0c93f787fb758e300cdca7ca5a54f96cfc1b709294d995167243ec059777e75.
- r64-cpu-regression-v1 completed18passed/3historicalprivateAPI deselected; actual1000
  CPU expert forwards exact/finite, RSS delta0, all10samples identical. Archived.
- Tests from `/workspace/round5-reference/legacy-tests/kt-kernel/test/per_commit/`
  test_cpu_detect_metadata.py + test_llamafile_{gguf_e2e,routed_correctness,cpuinfer_lifecycle}.py.
- New local runtime runner `round6_4/tools/run_d1_runtime.py` and shared lifecycle
  adapted in commit927779fcf;12localtests PASS. NOT_DEPLOYED/NOT_RUN.
- Next: finish same-operands expert safety (true1408 path) and frozen whole-request
  no-hit (round6_2/fixtures/runtime_controls.json input[1000], include causal/lookahead
  boundary). Existing control utilities under round6/tools and r63 fullcapture tooling
  can be reused with explicit schema adapter, not old numerical-candidate gate.
- Then run new D1 runtime runner. Needs contract file+rawSHA, frozenS6, actual preflight
  provenance with allfiles still matching, CANN env, NPU0. Prefer versioned deployment
  directory containing run_compact_controls.py and run_d1_runtime.py together.
- No need to rerun successful100compact or19preflight or18CPU tests absent code changes.

## Active continuation — 2026-09-16

- Whole-request no-hit `r64-nohit-v1` PASS: immutable [1000], one target plus
  lookahead, both L7/L8 no CPU route, partial zero, visible merge exact, full
  logits byte-exact OFF/ON. Both capture/index final integrity PASS. Archived.
- Real expert `r64-expert-safety-v1` PASS: all8 frozen experts × token shapes1/4,
  10 repeats each, actual2048×1408 weights, maxrelativeL2=0.003913061993474726
  <=.01, allfinite/exactrepeat. GuardEXIT_ZERO, no production change. Archived.
- **ACTIVE D1 full runtime** `/workspace/results/r64-d1-runtime-v1`, driver log
  `/workspace/results/r64-d1-runtime-v1-driver.log`. Deployed tools in
  `/workspace/results/r64-runtime-tools-v1/`. 156 same-process requests then12
  cleanrestart. ObserverOFF/productionoverlap, sameactualsource aspreflight.
  Check healthy process/log before any new NPU work; do not duplicate/restart.
- Quality source downloaded outside repo `/home/admin/Documents/ChatGPT/KT/r64-quality-sources`:
  GSM8K740312add88f781978c0658806c59bc2815b9866, CMMLUefcc940752ea4a1ea94d2727f11f83858d64fc8e.
  `haonan-li/cmmlu` redirects to `lmlmcat/cmmlu`, record rather than silentlyhide.
  Draft `/home/admin/Documents/ChatGPT/KT/r64-quality-draft-v1`:1319GSM+2048CMMLU,
  seed6404, inputmax223/334, requiredcontext735. StillNOTALLOWEDprimaryexecution:
  pendingdev/parser/capacity/overlap audits. NoONtestoutputs observed.
- Local analysis venv `/tmp/r64-quality-venv`: numpy2.5.3 scipy1.18.1
  pyarrow25.0.1 tokenizers0.23.2. `hf` uses separateinstalledCLI1.28.0.
- Paired interval implementation+5synthetictests prepared (allPASS), not final
  protocolfreeze. Conservative exact marginal bounds on pairedgain/loss with
  withinendpointBonferroni; bootstraponlysupplemental. Needs method assumptions,
  prospectivepower, CMMLUstratification discussion documented beforeformalfreeze.
- Still next: runtimecompletion, four engineeringoracle requests, actual1000
  deferredforwards, batch2/4, HD3/FD3, quality/perf/stability/ADR. Do not call
  Round6.4 complete at this checkpoint.

## Newest active state

- `r64-d1-runtime-v1` now **RUNTIME_PASS**:156sameprocess+12restart, allfinite/
  exactprefix, observerOFF. Archived locally; source+contractbinding inprogress.
- `r64-engineering-oracle-v1` failed TOOL_MANIFEST_SCHEMA:missing retained
  r63-forced-history-v1 wireversion. Ownservercleaned, no productionchanges.
  Failurearchived, unit3PASS, correctedtool deployedversionedfilename.
- **ACTIVE** `r64-engineering-oracle-v2`:4preselectedS6 histories fromsuccessful
  runtime; productionthenprivateexplicit. Driverlog `/workspace/results/r64-engineering-oracle-v2-driver.log`;
  permodedetail`production/process.log` and`explicit/process.log`. Checkhealth,
  don'tduplicatelaunch. Commandchainscompare_engineering_oracle.py oncapturePASS.
- Tools: `/workspace/results/r64-oracle-tools-v1/run_engineering_oracle_v2.py`,
  compare_engineering_oracle.py +run_compact_controls.py. Oldv1scriptpreserved.
- PreparedNOTRUN: run_d1_lifecycle.py (6fixedS6×64×6cycles,metadata-onlytrace,
  originalcontext512, actualroutedtaskcount>=1000 andterminaldrain, RSSsamples).
  Deployedin`/workspace/results/r64-capacity-tools-v1` withowned_server.py.
  Run afteroracle closes, via`run_capacity_test.sh run_d1_lifecycle.py`.
- PreparedNOTRUN batch2/4 tools and fixed16engineeringmanifest SHA
  `4fdb040ef1d2b100688761c931331705cb7a8c9d0223c96c09d4e704f4b7b24a`.
  run_batch_capture.py usesatomiccohortAPI; audit_batch_capture.py joinsALL
  physicalcapturestoRID/KV/history, protectsuniontop5,workerdrain,actualbatch.
  Startrepeats1shortbefore10+restart; don'tclaimgenericbatchPASSuntilaudited.
- Quality drafts nowalsoatA3`/workspace/results/r64-quality-draft-v1`.
  `run_quality_dev.py` OFFonlydevpilot8+8 first; primaryexecutionstillfalse.
  Thereare128historicalGSMoverlaps,2historicalCMMLUoverlaps,2CMMLUdev/test
  overlaps. Devselectionavoidsmatchingtestquestions; primaryrowsunchanged.
- Keepworking: this is not Round6.4 completion. No newpush/READYtag.

## Oracle closed; actual task lifecycle active

- Oracle v2 ENGINEERING_ORACLE_PASS:15600common tensors byte-exact,
  260full-logit comparisons exact,22private deferred forwards. Archived.
- ACTIVE `/workspace/results/r64-d1-lifecycle-v1`, driver log same path plus
  `-driver.log`. Originalcontext512,metadata trace,6fixedS6×64×6cycles.
  Count actual routed immediate/deferred CPU tasks and worker completion, not
  token proxy. Do not launch another model while this owned run is healthy.
- Latest owned_server.py including explicit offline environment deployed.
- Next: lifecycle audit; batch2 pilot OFF/ON and transaction audit; batch4;
  repeat/restart; dev-only quality pilot; frozen heldout/quality/perf/stability.

## Lifecycle closed; batch campaign active

- `r64-d1-lifecycle-v1` PASS:2340physical,3042actualroutedCPUtasks,
  258deferredtasks,4680synccomplete,8/8experts,pendingdrained. All36requests
  exactpriorreference. RSS plateaus afterrequest10; notzero-leak claim.
- ACTIVE `/workspace/results/r64-batch-campaign-v1`, initialb2OFFpilot.
  `campaign.json` recordsactive/completed. Driverlog adds`-driver.log`.
  Sequence4pilots(2OFF/ON,4OFF/ON),thenperprofile10repeat+2cleanrestarts,
  allcaptureandtransactionaudit. Failclosed,noothermodelparallel.
- Prepared primary quality runner NOTDEPLOYED/NOTRUN;3freeze-rejection testsPASS.
  Formalprotocolstillpreparation, no primary outputs permitted yet.

## Batch v1 tooling failure; v2 active

- v1 model-loadfailure beforeanyrequest:adapterexecv argv0barepython escaped
  virtualenv. A3readonlyA/B confirmed systemprefix+kt_kernelNone versusfullpath
  virtualenvprefix+installedmodule. No productionbug evidence/no productiondiff.
- Fixedadapter fullargv0; readinesswrapper nowwatchesguardexit;6localtestsPASS.
  v1 archived. Onlyownedfailedserveranditsreadinesshelperwere stopped.
- ACTIVE `/workspace/results/r64-batch-campaign-v2`, driverlog suffix-driver.log;
  samefrozenbatchmanifest andsequence. Do notlaunchanotherNPUmodel.

## Four batch pilots passed; long repeats active

- All4shortpilotsPASSgeneration+transactionaudit. b2all520physicalactualbatch2;
  b4all260physicalactualbatch4. ONdeferredroutes85/44 respectively. JSONarchived.
- ACTIVE v2 `b2-D1_NODEFER-repeat`, latest52/160requests, allownreferenceexact.
  Driverautomaticallyauditsandstarts2cleanrestarts, thenotherprofiles.
- LocalpreparedNOTDEPLOYEDafterv2launch:descriptiveholdouttools,
  primaryqualityrunner+summary,stabilitycampaign,performancephase+SSEtiming.
  PerformanceinputsrawSHA b383aa539815f8b6931446709163accec617fb809160223ee0e8b310dd0e8693.
- IMPORTANT localowned_server/capacityadapter nowalsohaveperformanceOFF-observer
  option; these latestchanges are NOTin activeA3capacity-tools-v1. Do not overwrite
  runningcampaign's deployedtools. Deployfuturetoolsetversionedaftercampaign.
- ReadonlyhostRSS/HBMmonitorprepared;initial10secondsamplingverifiedscope;
  latestCPU-util additionsNOTDEPLOYED. Neverstopsprocesses/installsanything.
- Formalqualityfreeze stillPENDINGdevpilot+protocolreview. No primaryoutputs.
