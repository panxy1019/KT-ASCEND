# Round7 autonomous continuation

## Latest override: parallel DEV system probes (2026-09-18)

09:12: abbreviated-v1 exitedHOST_NOT_FRESH_EXCLUSIVE_BEFORE_MEASUREMENT after
30minwait,6/12completed; ownedreapingcomplete,containeronlysleep. HoststillWAIT
outsideCPU~28cores. Prepared --buckets medium,long --blocks1 recovery support,
not yetdeployed. Retainshortblock4clean fromv1; rejectENTIREmediumblock fromv1
forresourcecontamination; replayall4medium+all4long innewdirectoryafterquiet.
Join onlywith explicit sourcehash/protocol/runtimeidentity and fullwindowaudit,
neverduplicate short or pickfastcells. Formal36stillNOT_COMPLETED_BY_USER_SCOPE_CHANGE.
Do notstartNPUjobs while hostbusy. Local123testsPASS (verifycommit).

08:26 checkpoint: abbreviated-v1 6/12 completed. Allshort4 + mediumturn0
EXCLUSIVE; mediumturn1 window CONTENDED_OR_NOT_YET_QUIET. HostoutsideCPU
now~95cores. Preserveallraw; do not promote isolated performance on this window.
Runner nextcell admission mustwaitfreshEXCLUSIVE (30min timeout). No external
workload modification. Inspect nextheartbeat for recovery/timeout; missingresource
gate is not evidence of Dynamic no-benefit. No threshold/model changes.

USER SCOPE CHANGE: shorten ABBA. Use tools-v27 --blocks1 serial12cells, preserve
threeworkloads/tokenbudgets/frozen1024 and isolation. v2 only2/36 complete at
inspection; stopownedrunner preserveasSTOPPED_BY_USER_SCOPE_CHANGE, startfresh
cache-static-dynamic-abbreviated-v1. No parallelformalperformance. Original3block
gate NOT_COMPLETED_BY_USER_SCOPE_CHANGE. Do not let formal evaluator promote
12cells or claimrepeatability/defaultON. SeeABBREVIATED_PERFORMANCE_SCOPE.md.

07:26+ DEPLOYED tools-v26 source5bf71fd25, archiveSHA
81786f1f5f0e9b69b619faa0cb7d57586f49ddfde07c8874eb6d3d10e8def774.
SCP/remoteSHA matched;120A3puretests PASS. Originalcontainer fullyidle after
v1interrupt (docker top only sleepinfinity). New hostobserver-v5 PID2140873,
output host-resource-monitor-v5; oldv4 untouched. Dispatched fresh
cache-static-dynamic-abba-v2/log usingtoolsv26, originalNPU2/CPU128–191,
threshold1024/frozenACK. Premeasurement admission waits up to30minutes;
fullwindowaudit stillrequired. Keephealthy run; inspectlog/protocol/hostlatest.

07:25 override: ABBA-v1 stopped owned performance PID204459 with SIGINT after
6/36 cells. Actual host pswpin124→161→163 occurred inside short0turn3;
do NOT promote attempt-v1 as isolated performance. All raw files retained.
Old observer additionally latched WAIT by comparing cumulative counters to
startup values. Interval comparison repair retains60squiet, cgroup zero swap,
memory reserve and full measurement-window rejection. New pre-measurement
admission waits for fresh EXCLUSIVE after warmup, outside timing. Local120tests
PASS. Deploy separately as tools-v26/observer-v5; fresh ABBA-v2 with same frozen
targets/36cells/candidate1024. No threshold, quality, model or timing denominator
changes. No latency-based sample removal. A/B workers remain idle.

06:31: freshV4hostEXCLUSIVE blockers[],age7.34s after60squiet; dispatched
`cache-static-dynamic-abba-v1` on originalNPU2/CPU128–191, toolsv25,
leftUniform/rightDynamic1024, frozenACK, observerOFF,36cells3ABBA×3workloads.
Log`/workspace/c64-dev/round7/cache-static-dynamic-abba-v1.log`; outputsamewithout.log.
Hostsamples `host-resource-monitor-v4/samples.jsonl`. A/Bremainidle. Do NOTstart
parallelwork now. Audit allactualwindows afterfinish; entranceEXCLUSIVEnotblanketPASS.
Afterformalperformance complete: evaluate_performance+isolation, selectscheduler,
thenDeferredD0/D1system/quality/performance. Preservehealthy longABBA.

06:27 UPDATE: DynamicHD128PASS COMPLETE/reaped; independentlyreassociatedraw,
verifiedmigration identities andmetrics. Trigger32/128,832updates,115158810624bytes.
ActualStaticdecodeNPUhit12.11605235% vsDynamic11.87967415%; CPUhits+0.2689663%.
No causalplacementgain shown, but frozen formalABBA stillrequired, no refit.
Archive76b2ba7966a123c8de0f36edbfb7de5fa41932c0ee0e35f1857df96d161affb4 downloaded/audited.
Fresh same-source hostobserverV4 PID1987362, source9e92b9cd8981a104277a851d7e45883ec5a1239935d7d0ad8a1498b70cd4c063;
outputhost-resource-monitor-v4. V3historicalunchanged. WaitfreshEXCLUSIVE60s beforeABBA.
All3ownedmodelserversreaped/idle. DefaultUniformStatic/D0unchangedpendingperformance.

06:09 UPDATE: StaticHD128PASS complete/reaped, raw2176/retained2048/tails128/unknown0.
Archive38cee92cfa74ec2af8d8a6893d56198cbf7a80943525803709d1abb89d565a74 downloaded/audited.
DynamicHD106/128 healthy; continue. CMMLU174/256PASS alreadyarchived.
BeforeformalABBA startfresh hostobserver campaign (same source/criteria), preservev3:
v3 latches swapcounter differences sinceinitialhoursago, not proof ofcurrentactivity.
Requirefresh60squiet/noothermodels beforemeasuring; don'trewriteoldeligibility.
See `02_DYNAMIC/CACHE_STATIC_HOLDOUT_RESULT.md`. Rootsession95982.

05:40 UPDATE: DynamicCMMLU complete174/256,invalid0,delta0pp PASS; actualtriggers0/256,
updates0: shortrequestguardrail NOTlongtriggerquality. Rawunique256/scoringsum/
lifecycle/reaping audited; archive279c1126b682f77152f5f22bbcb1aec037289c6cf353e30848872c496c82bc45.
StaticHD71/128 and DynamicHD70/128 healthy; keepbothrunning, noformalperf yet.
ROOT SSH session now95982 (93900stalled; do NOTsendmorecommands there).
See `02_DYNAMIC/CACHE_DYNAMIC_CMMLU_RESULT.md`; localarchive downloaded/verified.

05:18+ UPDATE: DEVscreen-v2 COMPLETE48/48; allfourprocessesCOMPLETE/reaped.
Independentaudit rebuilt exactratios and selected1024; candidate localcommit
`0be78ade3835d7adadc2f984fc39e63dd817af12`, ACKcommitab836745e.
Frozen selectedSHA bca98065f7e852961d5d4c3273861812865bb9597c6cf18162fe13d203b2e98a.
Rawarchive e98674efbd666c9d2fc9345739ecaeec35798775a068c520f7d4d69b06badf27.
Tools-v25 archive97b21b257d9ab4bfae99225cd9c76504984ce6c1739741cd32d0dfec75fe3cf6
deployed original `/workspace/c64-dev/round7/tools-v25` (visibleROonworkers).
118LOCAL+A3puretestsPASS. Explicitoperatorreusebinding A3PASS; nofakefreshoperatorrun.

THREE BACKGROUND TASKS NOW DISPATCHED (preservehealthy):
- A/NPU4: StaticHD128 via static_qualification, base `/workspace/parallel-worker/runtime-v8`;
  log `/workspace/parallel-worker/static-holdout-v1.log`, summarybase uniform-holdout-summary.json.
- B/NPU6: DynamicHD128 at frozen1024;
  output `/workspace/parallel-worker/dynamic-holdout-v1`, matching .log in parent.
- Original/NPU2: DynamicCMMLU256 at frozen1024;
  output `/workspace/c64-dev/round7/cache-dynamic-quality-v1`, matching .log.
  Systempredecessor completed1024, operator-source runtime-v7 explicitlybound.
  Quality parallel withHD is independentguardrail, cannot overrideHDsystemfailure.
Alluse toolsv25+samefrozenruntime-v8. NOformalperformance duringparallelphase.
AfterHD/quality audit+archive, advance isolatedStatic-vs-DynamicABBA, thenDeferred.
Candidate/ACK path under tools-v25/migration_bridge/round7/02_DYNAMIC/candidate-cache-v2.
DEV medianratio1024=1.008675, long1.18727; notspeedupclaim orformalperformance.

04:26 UPDATE: all256/512/1024 fullsystemPASS; A/B chain exit0, allreaped.
Coordinator joined exactartifact/corpus/mask provenance and automatically started
`dev-threshold-screen-v2` on originalNPU2. Current PID198035 threshold_sweep,
Static0 initialcases; A/B nowidle. Do NOT startotherheavycompute duringDEVtiming.
Old pauseddispatcher retired; do notresume. Sourcecollection
`parallel-cache-v8-probes-collected-v1/COLLECTION_PROVENANCE.json` PASS.
Archive `raw/round7-parallel-probes-complete-v1.tar.gz` contains full512/1024
plus joinidentitymanifest; SHA58447f0122018efa17c052bbf21ca9139a6148e601e68fa65c213572b1b5a09b.
AfterDEVfinish: localcommitcandidate+ack, then user-authorizedparallelHD/quality.

04:07 checkpoint: runtime-v8 Dynamic256 fullyPASS (A5/B2/OFF4, allCOMPLETE/reaped).
New A/B compact32+restart4 bothPASS;512/1024 bothProcessA2requests, healthy.
See `02_DYNAMIC/PARALLEL_CACHE_SYSTEM_CHECKPOINT.md`. New local explicit
quality operator-source binding not deployed: later use `--operator-evidence-base
/workspace/c64-dev/round7/runtime-v7` with new tools after checking source hashes.
Do not fake/copy a fresh operator summary into v8. Worker old `cpu_affinity`
metadata is inherited; actual cfg/argv/topology are correct. Preserve activefiles.

User explicitly authorized512/1024 on the two new workers; do not keep them idle.
Runtime-v8 Uniform compact32 + restart4 PASS on original worker. Existing256
child193606 continues. Original shell dispatcher191211 was SIGSTOP'ed ONLY to
prevent duplicate512/1024 and automatic timing during concurrency. Do not resume
that old dispatcher. The host join coordinator retires that exact parent after256
PASS, leaving healthy child untouched.

- `kt-r7-parallel-a-v1`: physicalNPU4, CPU0–63, NUMA0/1; compact/restart then512.
- `kt-r7-parallel-b-v1`: physicalNPU6, CPU64–127, NUMA2/3; compact/restart then1024.
- Both now use tools-v24 (source2ce653c13), independently writable
  `/workspace/parallel-worker/runtime-v8`; identical original runtime artifact
  paths remain read-only mounted under `/workspace/c64-dev`.
- Worker logs `/workspace/parallel-worker/qualification.log`; exitcode file and
  threshold summaries only after completion. Host directories
  `/home/admin/kt-r64-migration/round7-parallel/<container-name>`.
- Host `parallel_probe_coordinator.py` joins three PASS + reaping + artifact and
  corpus identities, then serially launches original-worker
  `dev-threshold-screen-v2`. Concurrent probe durations NEVER select threshold.
- After DEV completes: retrieve/audit candidate, LOCAL COMMIT and exact ACK;
  only then parallel Static HOLDOUT / Dynamic HOLDOUT / quality as predecessor
  gates allow. Do not rerun whole historical Frequency/R64 campaign.
- Formal ABBA still isolated, no simultaneous inference on added workers.
- Container relocation is not yet a new A3 PASS until its own gates finish.

V23 hit pre-model GUARD_SCOPE; preserved in both workers' failed-guard-scope-v1.
V24 scopes guard to owned new directory, retains generic guard/subreaper.
Host join coordinator PID1725839/log
`/home/admin/kt-r64-migration/round7-parallel/coordinator-v1.log`.
See `02_DYNAMIC/PARALLEL_WORKER_ADMISSION.md` and the join-coordinator log.

This is an execution handoff, not completed qualification evidence. Follow the
user's full Round7 mandate in this task. Ordinary software failures require
preserved reproduction, official-source investigation, minimal repair, regression,
and rerun; do not promote missing gates.

## Frozen lineage and boundaries

- Worktree: `/home/admin/Documents/ChatGPT/KT/kt-round7`.
- Branch: `codex/round7-static-frequency-dynamic-deferred`.
- Base: `bc2e5d53e05d97db83b40c09a6f80cddff0549ab`.
- No push, merge, tag, upstream wholesale upgrade, or Qwen production port.
- TP1, batch1, Graph OFF. CPU64 means two CPUInfer subpools of 32,
  CPUs128–191, NUMA4/5; it does not mean NPU TP2.
- Uniform R8-C64 selected; R16-C64 fallback. Static Frequency has global
  budget208 across 26 MoE layers; uneven and zero per-layer counts are legitimate.
- CMMLU frozen manifest: `bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04`.
  Frozen R64=174/256, product gate >=162/256, invalid=0.
- GGUF: `20114dba7790a9b6be40a17ffb3881c516b81ea86b8f2fcfd3ea61d052e96fe2`.
- Keep all historical numerical rejection/no-speedup classifications unchanged.

## Remote execution

- SSH: `ssh -J admin@110.123.0.3 admin@110.129.0.22`, then sudo root.
  Credentials are already in the user messages: do not write them to files/logs.
- Owned container: `kt-r67-c64-development-v1`; physical NPU2 maps logical0.
- Container root: `/workspace/c64-dev/round7`.
- Host backing directory:
  `/home/admin/kt-r64-migration/payload/kt-artifacts/round6-deferred/r65-r67-c64-development-v1/round7`.
- Actual deployed parent `e3084d309d6f39b8a9f7185d36ac27e4b77526d7`,
  child `eaa58ccea4ee372b8b207dd16f398eafe3da37ff` plus recorded overlays.
  Local parent gitlink is distinct: `f02f304a25364d1e8fe9c52515a22f20f5089a52`.
  Audit actual imported source/binary hashes, not only local HEAD.
- Model full revision `604d5664dddd88a0433dbae533b7fe9472482de0`.
- Native extension SHA `a5497fa085d704d8f67484b3936eaf448080f5d445899d88e4e76bea4305f077`.

## Healthy current stage — do not restart

2026-09-18 03:25: OriginalDEVscreen48/48COMPLETE selected1024 (longratio3.198),
cachefeasibilityPASS all168experts exact/controlH2D/stalebinding; warmstaging
31.7–32.0x faster, NOTmodelgain. Fulloriginalscreen+1024+cachearchiveSCP/hash/tar
PASS7c556cef6d5543c2d399d021eb505b5edec7a80215ea282cd497b1e80de61749.
NEWruntime-v8 privatecacheport/tools-v22; wrapper3e5d6a38c375ed6e8d538cd4e3eea3b8dad2d725313f8554fe5a1b3e2ff0fbb7,
cachehelperfc67bac2d5e1f6b78a520e4fa6f0b3e711fb72970ffa9f27032d0b9aa02a689b,
CPU461/stagerdb411/residentac2 unchanged.108LOCAL+A3puretestsPASS.
Ownedchain cache-runtime-v8-qualification-v1.log nowlaunchedUniformcompact→
cache-runtime-v8-probes/threshold-256/512/1024→dev-threshold-screen-v2.
NooldPASSsummarycopy. Keephealthy. Checkactualsourcebinding(inference tensor
versioncounter potential), memory growth and exactparity. OnlyafternewallPASS
andDEVscreen complete audit/commitcandidate+ack; noHD/CM/perfautomatic.
Needbindexistingoperatorgate forlaterquality via explicit provenance if newruntime
doesnotcontainoperator-summary; do notfabricatefreshoperatorrun. OldR64/Frequency
remainvalidprior evidence forcache-onlychange. SeeSTAGING_ACCELERATION_EXPERIMENT.
CurrentrootSSHsession93900;old14125stalled,don'tuse. WorktreeHEAD9c8859187+docs.

User now explicitly authorized acceleration. Originalall3DEVprobesPASS; original
DEVscreen-v1 healthy (lastStatic0 completed5/12). Preserveitsresults, do notcommit
anack/launchHD automatically: newcachefeasibilityexperiment queuedAFTERscreenexit.
See02_DYNAMIC/STAGING_ACCELERATION_EXPERIMENT.md. Separatecache-tools-v1 source
3c596d289dc328222fbda73acf59501bcf8461bc, archivec3a842148e34b93babeb2d19f1458a922fc22200d4a0bfb3c1192e8efcc723af,
SCP/hashbeforeextract;106LOCAL+A3puretestsPASS. Ownedflockwatcher
cache-feasibility-auto-v1.log waitsselectedJSON+originalscreenprocessgone, then
isolatedL1/13/26 all56cachebenchmark outputcache-feasibility-v1. No production
importchange. Auditcold/warmidentity/H2D/binding/memory beforeprivateport; newruntime
needsaffectedDEVreplaybeforefreeze. Don'toverlapformalperf/newmodelwithqueuedprobe.

2026-09-18 01:34: DEV512 fullprobe PASS, all3processesCOMPLETE/reaped,
all26layersworkloadshift and restart/observerOFFparityPASS. Completearchive
SCPexit0/local+A3SHA/fulltarPASS:
5ce1599953ee0387b499272fa6b5c0cd07c821ff40d80bf9ce249f1eeac38fd4.
DriverRUNNINGdynamic-dev-probe-1024, processA5requestscomplete, processB
restarttriggeractive. Keephealthy. ExistingDEVscreenwatcher waitsall3PASS;
no candidate/HD/quality/perf/Deferredyet. SeeDEV512_RUNTIME_V7_QUALIFICATION.md.

2026-09-18 01:00: DEV256 fullprobe PASS: no-trigger/realtrigger/weight/checkpoint/
generation/repeat/AtoB26layers/restart/observerOFFparity. All3processesCOMPLETE
and reaped. Completearchive local/A3hash+tarPASS SHA
ca6fbd3f16d8496b7706f6fcd3bde406ee28281a06ce933f0e81b9abd789497f.
SummarySHA1f1031df24da8b206c42b8ba6f4f7d4bf36d208fd9ba030ae74994519cd42536.
DriverhealthyRUNNINGdynamic-dev-probe-512;1024follows. Do notduplicate/restart.
Watcheronlyfitsafterall3PASS;no chosen threshold/HD/CM/perf/Deferredyet. See
02_DYNAMIC/DEV256_RUNTIME_V7_QUALIFICATION.md. FutureStaticHD mustusefreshv7
cohort beforecausalDynamicHDcomparison;oldv2HDnotpostfixoracle.

2026-09-18 00:44: runtime-v7 predecessors freshlyPASS operators100/100,
Uniform/Frequency32+restart4 each, freshdirectR64174/256invalid0. Closedarchive
local/A3hash+fulltarverified SHA
c1aee04bddc48c51969a45908b1c83462ba936b15ff02fd3e2a38c1f1c491845.
OriginalDEV256 healthy: processA5requests/78updates, processB2restartrequests/
26updates, all104weight/checkpoint/generation exact. Nowobserver-OFFparity
firsttriggerprefill; noPASSsummaryyet. Keephealthy;512/1024followautomatically.
Do notduplicateDEVscreenwatcher orfitbeforethreePASSprobes. See
02_DYNAMIC/RESIDENT_LAYOUT_PREDECESSOR_CHECKPOINT.md. NoHD/CM/perf yet.

2026-09-18 00:08: actualNZ29 raw/infer primitive regressionPASS exact/maxabs0;
NDfixed3jobsPASS. Losslesscompleteprimitive/source/prov/releasearchive LOCAL+ A3
hash/fulltarPASS afterSCPexit0 SHA
c9685a6b740b9e0e1eba29726c03841407810733561fae7905e6ad08645c0285.
NEWbacking-requalification-v6 healthyRUNNINGuniform-compact runtime-v7/tools-v21.
operators100/100PASS max .0017029279396063187. FirstUniform32requestsfinished/
gracefuldrain; restartseparateprocessloading, no Uniformgatepromotedyet. Keephealthy.
NEWownedflockwatcher dev-threshold-screen-auto-v1.log waitsforEXACTdriverstatus
PREDECESSORS_COMPLETE_AWAITING_DEV_THRESHOLD_SCREEN, thenrunsv21DEV-onlyscreen
to dev-threshold-screen-v1 (samefrozenDEV, Uniform, all3PASSprobesrequired).
Failure/timeoutpreventsfit; noHD/quality/perf/ackautostart. Don'tduplicatewatcher
orstartscreenmanuallyifalreadyrunning. Screencompletedcandidate requiresactual
localGitcommit/contentverification+ack BEFOREheldoutconsumers. No chosen threshold
yet. SeeRESIDENT_TO_RAW_LAYOUT_REPAIR.md. Localdisk~170MiBfree; largearchives
durablyA3+smallhashmanifestGit, notuncontrolleddownload/rootcleanup.

2026-09-17 23:49: scopedresidentPhase1port prepared/runtime-v7 created, helper+wrapper
hashesinRESIDENT_TO_RAW_LAYOUT_REPAIR.md. tools-v21 source72de5bcc9d5508ddce497d7057e444ede4286540,
archive1cbf2d133903af90c1b5fcef4b9065cdf35ab27251ed1453c36e77e9f26fd734,
completeSCP/hashbeforeextract;104LOCAL+A3puretestsPASS. Ownedprimitivegrid-v1
healthy (lastfixedNDinferstarting). RequestedNZ downgradedtoND format2; doNOTclaim
NZPASS fromv1. Newownedbackgroundchain resident-layout-recovery-chain-v1.log
waitsforownedgridparent161137exit; checksfixed3NDresults; runsv21actualNZraw/infer
withtest-onlyallow_internal_format=True andrequiresreportedformat29/exact;
onlythenfreshbacking-requalification-v6 runtime-v7 tools-v21, alloperators/static/
freshR64/DEV256-512-1024. Newchainlaunched, inspecthealth/PENDING stage. Never
restarthealthygrid/chain/model oroverwritev6/v7/oldreleases. No thresholdfit/HD.
After3probesPASS/reaped v21threshold_sweep --base runtime-v7 --devcorpora-v2DEV
--profileuniform --probesbacking-requalification-v6 --probe-prefixdynamic-dev-probe-
--output<newDEVscreen>. Candidate fullcommit+matchingack beforeHD/CM/perf.

2026-09-17 23:43 continuation: tools-v18 sameDEV256 againstunchangedruntime-v6
FAILED/reaped. ActualresidentE0/E8/E56 checkpoint exactBEFOREprepare in both
L1/L2, allfalseAFTERprepare rawcontext; CPUstagedE1/E63 remainexact. Thisisolates
resident→raw boundary. Full diagnosticarchive SCPexit0/hash+fulltarPASS SHA
b9cb551ce76301abc92915c58453526bd52164fba9fc959dfb01ff4a0d0fba21.
Localrootdiskfull encounteredgitaddarchive. Onlynewuntracked175MiBarchive moved
to /tmp/round7-dynamic-weight-source-diagnostic-v1.tar.gz; verifiedfullcopy remains
canonicalA3hostrawdir. BinaryNOTcommittedGit; smallarchiveprovenancemanifest records
thisexplicitly. No old/userdata deleted. Rootnow~175MiBfree; do notdownloadhuge
newarchives toworktree untilcapacityresolved. KeepdurableA3archives+smallJSONGit.
MinimalPhase1NPUonly fulltensorNDnormalization+explicitinfer→rawtranspose
commit e7d51bf5309416ee5fcdc6c05f4d2e6750c60c1f prepared;103LOCAL+A3puretestsPASS,
actualsourceguardsPASS. tools-v19 archiveeb0ab5276437c1fbe21494dcbaa964c8023128832853b902d18388eac8f7df85
completeSCP/hashbeforeextract. runtime-v7 privatepreparation currentlystarting;
repairednative461... unchanged. RepairA3validation PENDING: reducedactualPhase1
ND/NZ raw/infer source matrix then allorderedpostfixgates/newdriver.
No thresholdselection/HD/quality/perf/Deferred. Originalruntime-v6 immutablykept.

2026-09-17 23:18: v5/runtime-v6 FAILED/reaped inDynamic256 afterclosedFreqcompact
andfreshR64174/256invalid0. ActualLayer1generation1 passedauditor; Layer2failed
FULL_CONTEXT_CHECKPOINT_EXPERT_IDENTITY, only1update record. No512/1024/HD/fit.
Losslesswholefailedprobe+Freq/R64source/provenance archive localSCPexit0/hash+tarPASS
SHA528cc3636b7fce044cb109794e19314707fee1f35611aab83fdb981fb77ac29d.
Reducedactual_phase1 resident_inference→raw test raises aclnncopy161002(shape
4x12 vs12x4); independentcompatibilitygap NOTYETprovenactualLayer2rootcause.
Newtest-onlyauditor capturesfixed0/1/8/56/63 prepareentry/outputcheckpoint shapes/
identity and failingprojection tensors/metadata withoutrelaxingassertion or
productionchange;99existingLOCALtestsPASS. Nextdeployseparatetools-v18, same
runtime-v6 andsameDEV256fixture,newdiagnosticoutput. No needrepeatunchanged
productionpredecessors beforedebug replay, butafterrealfix rerunrequiredgates.
SeeFULL_CONTEXT_WEIGHT_IDENTITY_DIAGNOSIS.md. No active modelrun atcheckpoint;
reducedprobecompleted/reaped. Preserveoriginalv17/v6/v5results.

2026-09-17 23:01 checkpoint: v5/runtime-v6 operators100/100PASS(max .0017029279396063187)
andUniform32+restart4PASS. Allfourreference trajectoriesexactvsruntime-v5.
FullUniformraw/restart/lifecycle/source/provenance/operator-summary archive now
locallypreserved SCPexit0/hash+tarPASS SHA
6b0cb8281f978a6925d323bdb605b0d77b29490e124e708a254e551439619fe4.
Restartserver intentionalSIGTERM/ownedguardreaped; no newrequestfailure. Parent
driver nowhealthyRUNNINGFrequencycompact→directR64→DEV256/512/1024. Keephealthy.
NochosenDynamicthreshold/HD/quality/perf yet. SeeRAW_LAYOUT_RUNTIME_PREDECESSOR_CHECKPOINT.md.

2026-09-17 latest: pairedlayout/pinning fix standaloneA3PASS. Raw12cases/48writes,
inference12/48, checkpoint+H2Dexact/repeats/other-slot/negativeLOSSYsafe. Actual
runtime-v6 allocator CPU BF16buffers pinnedtrue/pointerscorrect.99puretestsA3PASS.
Tools-v17 source d7fa260996e0442baff32bf992140b5915af61fe/archiveSHA
33180a97bfd0fb76706f0d7847236b9510a5f2f381f536b9d7a2608020a02029,
completeSCP/hash-beforeextract. PrivatewrapperSHA
61354c6672446f604d11797347d6ece5b6bf83183765f4ccd8fba257b257faf4,
helperSHA db411ab5e5245cb76180857dcb8f18e13074da7c97a0eaa4749a50ae76fdd68f;
CPU461...unchanged. Losslessoriginal+fixedprimitive/staging/source/releasearchive
SCPexit0/hash+tarPASS SHA35afe5820f03161655eb04560c673c710ba7ed21560d1eb7842310bde97033ff.
NEWdriver /workspace/c64-dev/round7/backing-requalification-v5/progress.json
confirmedRUNNINGphaseoperators, tools-v17/runtime-v6, noresume/noPASScopy. Then
Uniform/Freqcompact→directR64→DEV256/512/1024. Keephealthy. No chosen threshold/
HD/Dynamicquality/perf yet. After3probesPASS/reaped usev17 threshold_sweep
--base runtime-v6 --probes backing-requalification-v5 --probe-prefix
dynamic-dev-probe- --dev corpora-v2/SCHED_DEV.jsonl --profile uniform --output
<newDEVscreen>; candidate localcommit+matchingack beforeHD/CM.
FuturehostobserverV3 nowA3deployed/exclusive; source9e92b9... samples
host-resource-monitor-v3. Identity-verified obsoleteownedV2PID287781 SIGTERM,
alloldlogs preserved. Onlyfutureformalwindows useV3; oldStaticwholeformalpending
unchanged. SeeBF16_RAW_LAYOUT_AND_PINNED_BUFFER_REPAIR/HOST_PERFORMANCE_OBSERVER.

2026-09-17 22:39 continuation: v4 FAILED/reaped at firstDynamic256trigger,
LLAMAFILE_BF16_STAGING_ASCEND_LAYOUT. Operators100/100, both32+restart4 compact
lanes and directR64174/256invalid0 had closed. Full failure archive verifiedlocal
SHA979182fe2e5fa8700fb17e8080131245561e8e14fd04038683e37092acd94cc3.
Actual reducedSharedFullContext allocator probe completed: CPU BF16 raw buffers
are unpinned, pointerscorrect,CUDAfalse. Minimal pairedraw/inference staging+
frameworkpinned BF16NPUTP1allocator port prepared;99LOCALtestsPASS and actualsource
guardsPASS. A3fixedvalidation PENDING. No active model driver at checkpoint.
Nextstage separate tools-v17/runtime-v6 (neveroverwritev16/v5), bothstaginglayouts
independent exactness+actualallocationprobe, then allorderedpredecessors and
Dynamic256/512/1024 in backing-requalification-v5. NoHD/thresholdselection yet.
SeeBF16_RAW_LAYOUT_AND_PINNED_BUFFER_REPAIR.md. Keephealthynewruns intact.

2026-09-17 22:02 latest: healthyv4 closedoperators100/100PASS(max .0017029279396063187)
andUniformcompact32+restart4PASS, fourreference trajectoriesexactvsruntime-v4.
NowFrequencycompact RUNNING, then directR64→DEVprobes256/512/1024. Keephealthy.
NewlosslessUniformraw/provenance/operator-summary archive locallypreserved,
SCPexit0/hash+tarPASS SHA162d4a2b91835cd3b91f1dbf21cdd4edc025365ef07d828b13ca690c59310dce.
SeeLLAMAFILE_STAGING_PREDECESSOR_CHECKPOINT.md. No newmigrationqualification.
FutureobserverV3 preparedLOCALonly, commit918b5256bfdc9a444d794f646d662663534d8a63,
sourceSHA9e92b9cd8981a104277a851d7e45883ec5a1239935d7d0ad8a1498b70cd4c063,
97localtestsPASS. Itattributesactualsnapshotcgroup/process-start-ID, retain
outsideprocessdeltas. Do NOT reinterpretoldV2Staticcontaminatedwindows asPASS.
Notdeployed yet; activehostV2/tools-v16 unchanged. Beforefutureformalperf,
stageversionednewhostobserver/A3puretests/smoke+freshcompletewindows; stoponly
identity-verifiedobsoleteownedobserver notmodel/businessjobs. Keepcriteriafixed.

2026-09-17 21:43 latest: minimalstagingport standaloneA3PASS12experts/48writes,
checkpoint+H2Dexact/other-slotsafe/lossyF32rejected.94testsLOCAL+A3PASS.
Tools-v16sourcea1c822f75ddd8b853b46520f6323f67bc2365d32/archiveSHA
6f87d6b0dfcc20ef4f5d77e6dd077273c7ddba3a17889f0c859110320e1155f6.
CorrectedSCPexit0/host-containerhashbeforeextract (firstuploadroot-owneddir
permissiondenied, notadeployment). Private runtime-v5 wrapperSHA
5762b3386e5bf8e9ee0e36ee922fd38fe073a1b75bb1a7ab3330dc01b0e814cf/helperSHA
b623670d0043a4d55359e4497fe06521aaed059d433fcb76105b98dd82948f44,
bothprovenance/importbound. CPU461...unchanged. Standaloneregression lossless
archive locallypreserved SCPexit0/hash+tarPASS SHA
8e91dec9f052d1a0f39227b87fe62cb53b86cd4b2cc234a964ef82983b07aecb.
NEWdriver /workspace/c64-dev/round7/backing-requalification-v4/progress.json
RUNNINGphaseoperators;logbacking-requalification-v4.log. Noresume/noPASScopy,
allorderedpredecessors→originalDEV256/512/1024. Keephealthy. Originalv3failure
stillfrozen/reaped; noHD/qualityfit/noDynamicqualifiedyet. AfterallprobesPASS,
usev16threshold_sweep --base runtime-v5 --probes backing-requalification-v4
--probe-prefix dynamic-dev-probe- --dev corpora-v2/SCHED_DEV.jsonl --profile
uniform --output <newDEVscreen>, thencandidate localcommit+matchingack beforeHD.
No thresholdselected yet. SeeLLAMAFILE_BF16_STAGING_COMPATIBILITY.md.

2026-09-17 21:38: v3 FAILED/reaped at firstthreshold256 trigger. Exacterror
LlamafileMoEWrapper missing submit_write_weight_scale_to_buffer, nowA3_REPRODUCED
notjuststaticcandidate. Allpreviousv4predecessorsPASS (bothcompact+restart and
directR64174/256invalid0); no512/1024/HD. Containerdocker top onlysleep.
Losslessfailedprobe+progress+v4source/provenance preserved locally, SCPexit0,
hash+tarPASS SHA565a84a404e2dbffc24a678d1a80b429edce51d00ce68857418b5e38e588d9a0.
MinimalTP1F32GGUF→BF16pinnedstaging seam prepared with sourceguards, preserves
officialeventfences/nativewriterfallback/KTEPcoordinator.94localtestsPASS;
notyetA3deployed orqualified. Nextnewtools-v16/runtime-v5, independentexact
stagingregression then allorderedpredecessors/probes. Freezehelper+wrapperhash,
do notoverwritev4/v15 orinjectcheckpointweights/disabletrigger. See
LLAMAFILE_BF16_STAGING_COMPATIBILITY.md. NochosenDynamicthreshold.

2026-09-17 21:18: v3 closedFrequencycompact32+restart4PASS, fourtrajectories
exactvsruntime-v3, directR64again174/256invalid0PASS. Nowthreshold256DEVprobe
RUNNING; initialshort128tokenprefill/16decode no-trigger request finished.
LosslessFrequency+R64raw/process/provenance/summaries locallypreserved,
SCPexit0/hash+tarPASS SHA
c04d08594040787dcb8bc831dea3818295f019bb2f8922bc76780f6a79ac79b5.
No actualmigrationPASSyet; keephealthyprobe, inspect actualschedulererrorif
itfails (especially LLAMAFILEmissing weightwriteinterface notedbelow).

2026-09-17 21:02: healthyv3 advanced operators100/100PASS(max .0017029279396063187)
andUniformcompact32+restart4PASS, fourtrajectories exactvsruntime-v3. Sessions
reaped; nowFrequencycompact RUNNING. Keephealthy. NewlosslessfullUniformsessions
+provenance/operator-summary checkpoint archive locallypreserved SCPexit0/hash+
fulltarPASS SHA fdde863413ae342b69e77a3e0ecf455bbd72bb54f921da41694d09940f40f015.
Newoperatorarrays remainA3; archive onlyoperator-summary/progress. No triggered
Dynamicrequalification yet. SeePRIVATE_CONTEXT_PREDECESSOR_CHECKPOINT.md.

2026-09-17 20:45 latest: correctedv2 directR64PASS174/256invalid0 and resolved
CLIexact. Then firstDynamicthreshold256triggeredrequest failedbeforepublication:
BF16fullcontext pinsCPU255 outsidecontainer128–191, OSError22. Allownedchildren
reaped; no512/1024/HD. Lossless entirev2/rawlogs/provenance/v13archive locally
preserved, SCPexit0/A3-localhash+tarPASS SHA
a8521b8ad6738add53625e5263127b8cc93036d03f887c0a68638add3437b244.
Minimal BF16-only affinity/device-stream port commit
0f732bce2c0d679d42855ceb282a05b923b2bc4a, no weight/mask/math change.
Originalactualchildsourceimmutable. PrivateSGLangcopy runtime-v4 containsported
wrapperSHA ea7fe7a6c46edd6eb0f6225a90f0892e0375a7752a9e7720c36711cf099cfcc5,
explicitcfg/import/provenancebinding; repairedCPU461...unchanged. NoPASSresults
copied into v4. Toolsv15sourceabove, archive
aaa73d6b6b4401d3b1146f6568d0b131e9dc3af46a1272cf92bf2b2c79b69a81,
fullSCP/hash-beforeextract;89tests LOCAL+A3PASS. Newordereddriver
backing-requalification-v3 launched runtime-v4, no resume, allpredecessors rerun:
operators→Uniformcompact→Freqcompact→directR64→DEVprobes256/512/1024.
Readv3progress/log; keephealthyintact. Usev15threshold_sweep --probe-prefix
dynamic-dev-probe- afterallprobesclose; candidatecommit/ack beforeHD. Nochosen
threshold, Dynamic/Deferred notqualified. SeeBF16_FULL_CONTEXT_COMPATIBILITY.md.

Additional CODE_INSPECTED seam to check if next triggeredprobe fails:
LlamafileMoEWrapper has no submit_write_weight_scale_to_buffer/sync method;
only utils/amx.py implements that Python interface; llamafile/moe.hpp and
moe-tp.hpp do not expose nativewrite API. Official BF16context currently calls
that interface forCPUownedweights. Do not claim this as A3reproduced yet or
stophealthyv3. If reproduced, preservefailedprobe, implement minimal F32GGUF
→BF16 pinned fullcontext staging adapter with exact frozen weight identity and
device-layout orientation, keeping official slot/event fences and KTEPcoordinator.
Loader get_undequanted_tensor_and_ggml_type returns read-only byteTensor backed
by mmap; requireGGML_F32 then view(torch.float32), correctlogical expert slice,
gate/up concatenation and NPUtranspose matching auditedcheckpoint orientation.
Do not reinterpret F32bytes asBF16, inject diagnostic checkpointweights or use
raw pointertransfer. Existing loader all4992projection identity evidence stays
frozen; newstage needs exactcheckpoint/source/target regression beforepromotion.

2026-09-17 20:25 checkpoint: v1 stopped at fresh R64 adapter mismatch,
175/256 invalid0. Both post-fix compact lanes PASS32+restart4. Historical
R64 uses direct Ascend withoutKT; v12 mistakenly used KT64 and expandedlimits.
Test-only repair commit5a673f1a70dc5e472be72f6b0cf30a7eafff4352,
releasev13 SHA6971ddcd396fa0b1e279404f1af0970585e1f3b247996c58187c0fffe879eb8d,
completeSCP host/container verified;82puretests LOCAL+A3 PASS.
New driver backing-requalification-v2 is launched with --resume-from v1,
retained three PASS summary hashes, private repaired native and new output.
Read progress/log there; initial CANN environment imports take minutes.
Do not restart healthy stage. Sequence exact historical directR64(required174,
resolvedCLI exact) then DEVmigrationprobes256/512/1024. No threshold/HD yet.
Bothcompact/full175failure archive nowlocallypreserved, SCPexit0/hash+tarPASS,
SHA b0a4be94684d862ab553affde27506de745c978d733d53f5be2a046e2de616bb.
See R64_REFERENCE_ADAPTER_REPAIR.md. No production/native change in this repair.
Future threshold_sweep explicit --probe-prefix dynamic-dev-probe- binds retained
probe summary hashes; deploy a separate future release, never replace livev13.

20:25: v2 progress RUNNING fresh-r64-quality, retained gate hashes recorded.
Actual corrected resolved argv equals the historical direct R64 argv exactly;
noKT reference confirmed. Server startup/imports still in progress at inspection.
Future releasev14 staged separately, source
adf0cd9e5c5f1633122babf6385c470b7532f6e9, archiveSHA
93f6023266e35fd5df0f0461cc555b51fcedd07e342a2d5720d10ad35848009c.
CompleteSCP/hash-before-extract;85puretests LOCAL+A3 PASS. Activev13 untouched.
After allthree migrationprobes PASS and reaped, run v14 threshold_sweep with
--probes backing-requalification-v2 --probe-prefix dynamic-dev-probe-;
new DEVscreenoutput then localcandidatecommit/ack beforeHD. No threshold chosen.

Latest19:50: repaired operators100/100 PASS(maxrelativeL2 .0017029279396063187),
Uniformcompact32+restart4 PASS and four trajectories exact vsoriginalruntime-v2.
Newlossless completeoperator arrays/weights/partials/logs/results/provenance
archive locallypreserved, SCPexit0/local-A3hash+fulltar verified SHA
142e187455ef6723b031cab984c13610eba12d1321c7f593b76fe091dee6986a.
Driverbacking-requalification-v1 nowhealthyRUNNING frequency-compact;
stagesoperators/uniform-compact. Do notrestart. ThenfreshR64(required174)→
DEVmigrationprobes256/512/1024. SeePOST_FIX_PREDECESSOR_CHECKPOINT.md.
Servingraw stillA3 untilbothcompactclose; do notclaimlocallyarchivedyet.
Futurethreshold_sweep currentlyexpects --probes root/threshold-{256,512,1024},
whilecurrentdrivercreatesdynamic-dev-probe-{256,512,1024}. BeforeDEVscreen,
addexplicit testedprefix adapter or immutable read-only aliases afterprobesclose;
do notoverwritecurrenthealthy v12driver or silently change scientificinputs.

Latest19:30: original CPU-backing defect has passed repaired A3 standalone
L1/L13/L26 eviction regression,18cases/180forwards finite+10repeatexact;
max canonical relativeL2 .0017457476763079638 <=.01. New nativeSHA
46182176279b072b4ab52d21bb8cc0ecf4d217399d486155c7fcd4f0690e21e2.
Lossless newbinary/header/build/provenance/results archive preserved locally,
SCPexit0/hash+tar verified SHA
68ade2a57257a8f9a85c0cf22abcc4c2ecc133279fb269474ddc01ea9cb5d72c.
Originalbinary/source/staticresults unchanged. Full-model Dynamic not qualified.

Newtools-v12 sourcecommit43b6c67631adf8506744611bec091203729ff812,
archiveSHA1eaac5e364c3994aa6852da575eb2e064a19b418ce2cadb3ca03b2c66a3600b4;
fullSCP host/containerverified;78puretests LOCAL+A3 PASS. Newruntime-v3 created
with private_CPU_runtime_root/hash-provenance binding, copied only profile/config
inputs, NOT old PASS results. Its predecessor driver is now RUNNING phaseoperators:
/workspace/c64-dev/round7/backing-requalification-v1/progress.json;
logbacking-requalification-v1.log. Usesv12/runtime-v3/newbinary environment.
Healthy driver must not restart; stageorderoperators→Uniformcompact→Frequency
compact→freshR64 CMMLU(required174/256)→DEVmigrationprobes256/512/1024.
Onsoftwarefailure preserve/diagnose/repair/newrelease/resume, no thresholdfit/HD
before predecessors. On completionstatus
PREDECESSORS_COMPLETE_AWAITING_DEV_THRESHOLD_SCREEN, proceed DEV-onlyscreen,
commit selectedthreshold and matchingack beforeHD/CM. No Dynamic/Deferredmodel
quality/performance gate has run yet. SeePOST_FIX_RUNTIME_PROTOCOL/CPU_BACKING_FIX.

Latest19:15 update: original L1 backing eviction FAIL reproduced, E0/capture1
finite/10repeat exact but canonical relativeL2=1.0 (gate.01). Full failure/archive
now committed; SHA e9ba345f59f50e78edec94c2d1e77b1a3cea90aa0d1280b82b9f8df51db31a16.
Minimal load-only skip removal commit4bfe9c44f, builder commit
b18b72b67bfd83883178e3aec538c9052f9915f1. New isolated backing-fix-build-v1 is
currently BUILDING, logbacking-fix-build-v1.log, build.json in that directory.
Do not restart duplicate/overwrite original native/source/runtime. Build patch
SHA be1489cd8efd7c57238fa90616206d4e082d98e00118cb6f8ec1877516d2430b;
host/container verified after complete SCP. Use its with_full_cpu_backing_env.sh
for post-fix standalone eviction L1/13/26, newoutputcpu-backing-fixed-v1/Lxx.
Wrapper script already execsPython; invoke `bash script probe.py`, not executable
directly or with an extra Python argument. Need repaired binary provenance,
operators/Static compact/freshR64 before full-model Dynamic. Current v11 model
launcher still hardcodes original runtime: implement a minimal provenance-bound
new-runtime adapter in a NEW tool release before launching repaired full model.
Do not claim fix closed or Dynamic qualified until post-fix regressions pass.
See02_DYNAMIC/CPU_BACKING_FIX.md. Static selection Uniform is frozen;
performance archive entire36+originalwindowaudit is preserved and committed.

2026-09-17 19:05 checkpoint: Static driver completed36/36 and reached
STATIC_EVIDENCE_COMPLETE_AWAITING_AGENT_SELECTION. Original-A3-mtime audit-v1
accepts30/36 windows; six first post-completion samples report outside CPU busy,
so whole campaign remains PENDING_HOST_ISOLATION, not formal-isolated PASS.
Complete exploratory medians show small gains, but independent decode routing
regresses under Frequency. Static selection retains UNIFORM_R8_C64; see
01_STATIC_FREQUENCY/STATIC_BASELINE_SELECTION.md. Frequency system/CMMLU PASS
and raw audit are preserved. No threshold refit or quality-based mask selection.
Archive creation is in progress; verify its SHA and complete SCP before claiming
local preservation. Next: original-binary backing eviction regression1/13/26,
then minimal repair if reproduced, fresh R64 if common loader changes. Use v11
future tools and Uniform profile for Dynamic control.

19:12 update: performance archive now fully preserved locally, SCPexit0,
full tar validationexit0, local/A3 SHA both
5eadc608619bf19c9bd6d84a0155928040b2408161ff360ca3c2187e6262dfc1.
Static selection commit ee2b17f53 (resolve full before reporting). Owned container
had only sleep infinity: completed Static processes fully reaped. Original-binary
L1 eviction probe is now running with Uniform profile under v11:
cpu-backing-original-v1/L01/result.json and cpu-backing-original-L01-v2.log.
Do not launch a duplicate. Initial v1 launcher permission-denied before test is
retained, corrected v2 uses taskset/bash wrapper (wrapper already execs Python).
If L1 fails, preserve it and original binary before fixing. If PASS, proceed
separate processes L13/26. No full-model Dynamic run has started.

2026-09-17 15:46 heartbeat: Static performance16/36 at inspection, healthyRUNNING;
host V2 EXCLUSIVE blockers[]. No extra NPU test launched. Prepared post-DEV
Dynamic consumers now enforce explicit threshold commit acknowledgment via
`frozen_threshold.py`: matching selected JSON/profile/DEV hashes and full local
candidate commit, no quality/HD fitting. Formal performance may no longer rely
on a bare --threshold integer. See `02_DYNAMIC/THRESHOLD_COMMIT_FREEZE_PROTOCOL.md`.
All71 local tests PASS; new v11 is staged separately, not model-qualified.
Never overwrite v6/v10. Future Dynamic HOLDOUT/quality/performance and
Deferred-on-Dynamic require --threshold-ack along with --threshold-freeze.
Actual Git candidate-content verification must precede creating the ack; runtime
checks are a handoff binding, not cryptographic proof of local commit contents.

Latest future tools release-v11:
`/workspace/c64-dev/round7/tools-v11/migration_bridge/round7/tools`.
Source commit `49a2bf8cec85ce781a2c66ea2a2a7fdeac8306c5`, archive SHA256
`352a12ba1d790d2a90ec7dfa3b899f5429395b65fc4525f0f1208de37bced051`.
Complete SCP exit0 and host/container hash verification preceded extraction.
All71 pure-Python tests also pass on A3 Python3.11 (0.166s); no torch import,
NPU probe or model qualification. Use v11 for later gates only after healthy
Static stage completion/reaping. All original-binary backing and Dynamic
predecessors still required. Active v6/native/masks/corpora are unchanged.

2026-09-17 15:28 checkpoint: observer-OFF Static ABBA is healthy,14/36 completed;
host V2 EXCLUSIVE, blockers[], owned cgroup swap0, no active host swap change.
New offline separation of closed HOLDOUT captures shows Frequency benefit is
prefill-dominated: prefill CPU assignments -16.0494%, but decode CPU assignments
+1.2582%, decode NPU-hit rate -1.1058pp. Do not select a Static baseline until
the whole36-cell performance and independent window audit close. See
`01_STATIC_FREQUENCY/PREFILL_DECODE_ROUTING_CHECKPOINT.md` and machine output
`stage-routing-v2/static-stage-routing.json`.

Prepared future Dynamic tools had a Python3.11 IntEnum classification defect:
actual DECODE mode string is "2", not "DECODE". Repaired locally using independent
exact-history decode indices; future observer also records name/is_decode.
65 local tests PASS. See `02_DYNAMIC/DECODE_MODE_TOOLING_REGRESSION.md`.
No Dynamic A3 gate or production change has run. This repair is not present in
v6/v9; v10 is now separately staged with verified complete SCP and archive hashes.
Do not overwrite active tools or run extra NPU work while Static owns the device.

Release-v10 staged (not a model driver) at
`/workspace/c64-dev/round7/tools-v10/migration_bridge/round7/tools`.
Source commit `110c091800aebe9e587ceb838a8c6651704e9669`;
archive SHA256 `e5d95ccd3beb759298572edbf0763d0f0eeb59cbb97e1b8732f7c3c23c12de22`.
SCP exit0, host and container SHA verified before extraction. All65 pure-Python
unit tests passed both locally and under A3 Python3.11, without importing torch
or launching any additional NPU work. This is tool verification, not Dynamic
full-model/system migration qualification. Active v6/native unchanged. Use v10
for future gates after Static completion/reaping; retain v9 window-audit adapter.

DEV statistics collection attempt2 completed all256 prompts and its full
integrity audit: raw4352/attributed4352, kept4096, proven unused tails256,
unknown0, each routed layer1498272 assignments. Lifecycle/reaping COMPLETE.
Candidate freeze commit `391b332ff0dfe648cec0bb4113b9daeabb8571e9` contains
the exact original .pt bytes in the lossless DEV archive and readable mask JSON.
Full artifact hashes are in
`01_STATIC_FREQUENCY/FREQUENCY_CANDIDATE_COMMIT_ACK.json`.
The frequency budget is208, per-layer counts4–12, no zero-resident layer in
this candidate. Operator representatives and Uniform compact/restart/parity
have passed on A3. Frequency compact/restart/parity also passed; its exact
structure and separate lossless archive are preserved. Uniform and Frequency
HOLDOUT have now both passed all128 requests and complete forward
attribution. Uniform NPU assignment hit rate0.12440549718339645, CPU assignment
rate0.8755945028166036, CPU weight mass0.8705099846154037. These are observer-ON
route diagnostics, not formal performance. Frequency NPU-hit rate0.26265173490281274,
CPU route rate0.7373482650971872, CPU weight mass0.7438615086765767. Frequency
quality now PASS173/256, invalid0, delta-0.390625pp vs frozen174. Lossless quality
archive and paired-ID audit are committed in the quality checkpoint. Driver is
now `uniform-vs-frequency-performance` (observer-OFF3ABBA36 measured requests).
Performance and final static selection are not yet promoted. See Frequency
HOLDOUT/quality checkpoint documents for their lossless archives.
Preserved system evidence is in
`01_STATIC_FREQUENCY/STATIC_SYSTEM_CHECKPOINT.md` and its lossless archive.
The serial qualification driver has been launched in the owned container:
`/workspace/c64-dev/round7/static-phase-v1/progress.json`, log
`/workspace/c64-dev/round7/static-phase-v1.log`. It uses release-v6, base
`runtime-v2`, HOLDOUT `corpora-v2/SCHED_HOLDOUT.jsonl` and
`frequency-candidate-commit-ack.json`. Healthy driver must not be restarted.
Read this new progress path rather than completed DEV progress for monitoring.
On failure, preserve the first failed stage and logs, repair/reproduce before
resuming; do not overwrite earlier raw directories. Stage order is operator
representatives→Uniform/Frequency compact→Uniform/Frequency HOLDOUT→Frequency
CMMLU→isolated performance if quality passes. Agent static selection follows.
Use read-only checks of:

```
docker exec kt-r67-c64-development-v1 sh -c 'grep -n "phase\|status\|error" /workspace/c64-dev/round7/static-phase-v1/progress.json; tail -5 /workspace/c64-dev/round7/static-phase-v1.log'
```

Collector log: `collect-dev-v2.log`; run directory `runtime-v2`.
Actual server stdout is `guard/process.log`, not empty `launcher.log`.
Driver is backgrounded through `flock` and an internal exclusive lock.
Do not launch a duplicate or run additional NPU tests while this stage owns NPU.
Do not replace active release-v6 tool files with later local edits: deploy future
tools into a separately versioned directory after current stage reaping.
Attempt1 is preserved in `runtime-v1` and rejected for an extra-forward integrity
assumption. Attempt2 exact IDs/positions attribution proves an unused final-token
forward, excludes it from fitted statistics, and rejects unknown/duplicate/missing
passes. Observer ON/OFF parity passed. Finite/repeated pilot outputs passed.

## Continue in this order

1. Finish DEV256; audit all extra forwards and recording startup boundary.
   Require `DEV_STAT_COMPLETE`, clean owned reaping, raw/clean statistics and
   attribution hashes. Do not fit on partial statistics.
2. Run `freeze_frequency_profile.py`: official global-topK mask208; record all
   per-layer counts including zero; pull artifacts and commit freeze before
   HOLDOUT/CMMLU. DEV/HOLDOUT already frozen independently in repository.
3. Run representative operator audit and static Frequency compact/repeat/restart,
   observer parity, masks/mappings/ownership, finite and local L2<=0.01.
4. Compare Uniform/Frequency HOLDOUT route metrics and separate observer-OFF
   isolated performance. Run Frequency CMMLU256. Choose static by routing/perf,
   not correctness scores. Frequency failure retains Uniform and continues.
5. First execute the standalone `cpu_backing_probe.py` on the original binary:
   actual multi-subpool LLAMAFILE load skips initial NPU-owned expert slices;
   Dynamic eviction could expose an uninitialized CPU backing pool. This is a
   CODE_INSPECTED defect candidate, not yet reproduced. Use predetermined
   initially resident experts and real operands, preserve pre-fix failure, then
   minimally repair and rerun original regression/operators/static compact if
   needed. A common CPU weight-loader repair also requires fresh R64 reference
   reproduction per the user's common-loader-change rule; preserve the original
   174/256 result and frozen criterion, bind any new native/runtime paths to new
   provenance before launch. Do not run it concurrently with healthy current serving or overwrite
   the live native binary. See `02_DYNAMIC/PRE_MIGRATION_SOURCE_AND_BACKING_AUDIT.md`.
   Continue Dynamic deployed-code audit and DEV-only threshold256/512/1024 sweep; freeze
   threshold before HOLDOUT/CMMLU. Validate no-trigger, real triggered full-prefill,
   exact copied BF16 weights, wrapper/device mappings/masks, same-generation
   publication, deterministic selection, A-to-B adaptation, restart to static.
6. Static vs Dynamic short+256/medium+512/long+1024 decode, three ABBA blocks.
   CMMLU256 and triggered long smoke. No benefit is legitimate, default OFF.
7. Selected scheduler D0 vs D1 system/perf/quality and last-layer effective0.
   No benefit/rejected approximation leaves Deferred OFF. Do not expand D2+.
8. Artifact SHA manifest, final ADR/report/state, selected scheduler handoff.
   Qwen3.8-Flash-Next research is already recorded; immediate port blocked by
   qwen4_exp integration, not a blocker for Round7 completion.

## Tools and current edits

`migration_bridge/round7/tools/` contains collector, attribution validator,
frequency freezer, static qualification and operator audit. Thirteen local tests
passed after adding pure migration and SSE timing checks. With Deferred trace
assertion tests, the current local suite now has sixteen passing tests. `dynamic_audit.py`,
`dynamic_probe.py`, `performance_screen.py` and new hook wiring are not yet
deployed at this checkpoint. Review actual trigger integration before deploying. Diagnostic
weight readbacks/synchronization must remain absent from timed performance.
`threshold_sweep.py` now prepares DEV-only twelve-case/64-decode exploratory
selection after all three threshold migration probes pass. It freezes the
selected threshold before any held-out use. `runtime_quality.py` reuses the same
immutable evaluator for Dynamic and Deferred. Migration component telemetry,
component timing and final orchestration still need integration.
`deferred_probe.py` now tests actual D1 route execution, terminal0/draining,
repeat/prefix/restart/observer parity on the selected Static or Dynamic base.
Its trace seam stays OFF in timed performance and quality. New session configs
verify retained imported-source/native/corpus hashes before launching; the large
RO GGUF retains its initial full hash with a size check, not a claimed rehash.
The prepared performance screen supports D1 on a selected
Dynamic base via `--dynamic-base`; never silently evaluate D1 on the wrong base.
HOLDOUT routing uses official recorder metadata to exclude proven unused final
forwards and rejects unassociated routes, just as DEV fitting does.
New routing flags distinguish per-layer assignments from whole-model token CPU
touch rate. These later tools remain undeployed until current collection finishes.

`run_static_phase.py` can run the following serial gates in one detached driver
after DEV completion and a candidate local commit freeze. It requires a JSON
ack containing `local_candidate_commit` (full40 SHA), `frequency_profile_sha256`
and `frequency_activation_sha256`. Create this only after verifying the actual
local commit contains the matching frequency artifacts; no self-referential
commit-SHA claim is needed inside that commit. Driver gates DEV reaping and
hashes, then operator→Uniform/Frequency compact→HOLDOUT→Frequency CMMLU→isolated
ABBA. A Frequency quality rejection skips its costly performance and preserves
Uniform. Software failures return to the agent for repair; never erase failed
run directories. Driver completion still requires agent selection/documentation
before Round7B. The local suite has eighteen passing tests after freeze checks.

Tool release-v6 is executing the static driver at
`/workspace/c64-dev/round7/tools-v6/migration_bridge/round7/tools`.
Its archive SHA256 is
`af6eeeb22ff796525d15c389757e7296320561caed9d3088de5a26ddfd43750b`,
built from local commit `5aa2d89af` (resolve full SHA before reporting).
The completed collector's release-v5 directory was not modified. Current local
future Dynamic tools add independent HF checkpoint source identity and the CPU
backing eviction probe; twenty-one local tests pass. Those additions are not
deployed or A3-verified. Use a separately versioned release-v7 for future gates;
do not replace v6 during its healthy static run.
Release-v7 is now staged (not executing) at
`/workspace/c64-dev/round7/tools-v7/migration_bridge/round7/tools`.
Archive SHA256:
`0cc75662b081a5748d534c8c9323e89bbc96c614a9be6aca7b797fbcd08289dd`,
from local commit `ba5b10d0be82da6c0673c0d1294741183b67de96`.
After static serving reaps, invoke `cpu_backing_probe.py --layer 1
--profile /workspace/c64-dev/round7/runtime-v2/profiles/frequency.json
--output /workspace/c64-dev/round7/cpu-backing-original-v1/L01` through the
frozen `with_c64_env.sh`. Preserve original binary failure before repairing;
subsequent layers13/26 and post-fix runs need separate new output directories.
Never use a probe result alone as full-model Dynamic qualification.
Local future tools now also include `dynamic_holdout.py` and
`dynamic_route_summary.py`: frozen-threshold 128-case HD routing, decode-only
exposure excluding full-device prefill/unused tails, actual migration event
correlation, and initial-mask-on-same-routes counterfactual diagnostics. The
counterfactual does not replace the separate Static-vs-Dynamic comparison.
Twenty-six local tests pass. Those additions are not in staged v7 or running
v6; deploy a new release-v8 after review and current-stage reaping. See
`02_DYNAMIC/HELDOUT_ROUTING_PROTOCOL.md`. Actual CMMLU trigger-rate telemetry,
performance host-isolation evidence and later-stage orchestration still need
integration before claiming complete metrics.
Further local edits add `dynamic_telemetry.py`, a default-off lightweight
actual-update/component host-call recorder for untimed quality and parity.
Dynamic probe OFF trajectory replay enables this seam and requires real events;
runtime quality reports actual CMMLU trigger rate through exact request windows.
No extra NPU work or weight readback is introduced by this seam; it is not a
weight/generation certification and stays OFF in timed performance. Current
suite29 tests pass; additions are not running on A3 and should be included in
future release-v8. Current release-v6 remains untouched.
The current local suite has33 passing tests after read-only host contention
observer/window validation. A host-only observer is launched separately:
`/tmp/round7-host-resource-observer-v1.py`, samples mapped to
`/workspace/c64-dev/round7/host-resource-monitor-v1/samples.jsonl`, latest.json.
It never stops workloads or mutates containers/libraries. First launch preceded
SCP completion and exited before sampling; original stderr remains in
`/tmp/round7-host-resource-observer-v1.log`, corrected start2 stderr in
`/tmp/round7-host-resource-observer-v1-start2.log`. Verify fresh samples before use.
Source SHA `92ab38ceafe86ff494b0e1d3c9af246f1db69b8377f58e70704d757e3e6b3e45`
matches local committed code. See `00_PROVENANCE/HOST_PERFORMANCE_OBSERVER.md`.
Future performance tools accept `--host-observer` and reject formal eligibility
on incomplete/stale/contended measurement windows. Current v6 is not modified;
its later performance can be analyzed against independent host samples. This
is a host contention proxy, not direct DRAM bandwidth measurement.
V1 correctly observed but stays WAIT on262144bytes historical host swap. Actual
retained serving admission is already V2: cgroup zero swap and no active host
swap I/O. Observer is now aligned to that policy, not the stale old strict-zero
lease. Owned cgroup=0, sampled owned VmSwap=0, repeated pswpin0/pswpout1 unchanged;
RAM available~912GiB. V2 rejects any swap counter change from its initial snapshot
and records raw memory/swap values continuously. Source SHA
`86066a50bc0bc02dba7eb0fa2bf8b0687fc62091333ac00c11ef954a99e815c1`.
Actual observer is `/tmp/round7-host-resource-observer-v2.py`, future samples path
`/workspace/c64-dev/round7/host-resource-monitor-v2/samples.jsonl`; use V2 for
future performance windows, not V1. Obsolete owned V1 PID283141 was identity
verified and SIGTERM stopped; all logs/samples preserved, no model/business task
stopped. V2 launch hash verified; confirm latest.json fresh and60S quiet before
claiming EXCLUSIVE. Current34 local tests pass.
Before frequency fitting, actual deployed `_init_kt_gpu_experts_masks` was
inspected: it counts 26 real MoE layers before computing ratio. Thus0.125 means
208 routed slots, not216 from counting dense layer0. Both loader and freezer
must continue to agree on that budget.

## Follow-up policy

Release-v9 staged (not a restarted driver) at
`/workspace/c64-dev/round7/tools-v9/migration_bridge/round7/tools`, source commit
`bb3c59eced61670fd23dcbd6a6e722030d548bdb`, archive SHA
`724534ee6705dd7b271bf15a61e45896f34402cb31a818657d6992b60a214e9e`.
SCP exit0 and host/container hashes verified before extraction. Active v6 and
native remain unchanged. On A3, read-only partial-window adapter smoke inspected
completed first4/36 cells, all four independent host intervals EXCLUSIVE,15samples
each, original-output serialization slack0.2748..0.2837s. This is only tool/
partial-window validation, not complete performance or scheduler selection.
Use v9 for full post-audit only after all36 cells complete and before archiving
original file mtimes; preserve raw samples and resulting audit artifacts.

14:30 heartbeat: current Static performance remains healthy,3/36 completed at
first inspection, hostV2 freshEXCLUSIVE. Do not extrapolate partial performance
or disrupt it. Prepared `audit_performance_windows.py` for unchanged v6: input
wallclock through original measured-output file mtime, raw-summary correlation,
closed/reaped lifecycle, nonoverlapping ordered36cells, frozen-source independent
host samples. Run on original A3 files only after complete campaign, before tar
changes mtimes. Missing/stale/contended evidence remains pending. Full local
suite54PASS. New adapter is not in v8 or running on A3; deploy separately after
review. See `00_PROVENANCE/HOST_PERFORMANCE_OBSERVER.md` for invocation.

Latest complete gate: Frequency CMMLU256 PASS173/256, invalid0, paired IDs256
verified (bothcorrect170/bothwrong79/regressed4/improved3). Archive SHA
`22793c6a95c543a10664fc265769d77aaad4539f069512ee2c968ce0e3590038` fully verified.
Healthy Static driver has entered observer-OFF performance automatically.
Preserve36-request3ABBA. Independent host V2 observation must cover actual
measurement windows before formal-isolated claims; a transient WAIT following
owned-process teardown is not itself a measured-window failure. No Dynamic,
Deferred, backing-MRE or production repair launched yet.

Release-v8 is now staged, not running:
`/workspace/c64-dev/round7/tools-v8/migration_bridge/round7/tools`.
Archive SHA256 `7f9bc2c6db659b523acebfa37c2536399bd61bb218b23921f975d146875a4b7c`,
built from source commit `cce07f5374e9932443ae156e88d146e7593e062d`.
SCP finished and both host/container archive hashes matched before extraction.
Includes CPU backing MRE, independent source identity, decode-only Dynamic HD,
actual quality migration telemetry, host isolation and full ABBA decision audit.
Active static v6 and native binary were not modified. After Static finishes and
reaps, use separately versioned v8 tools for original-binary backing MRE and
later gates; no Dynamic/Deferred probe has been launched yet.

Future Dynamic performance evaluation now has a pure-Python full36-cell ABBA
audit (`tools/evaluate_performance.py`): nine tests for missing/malformed cells,
coalesced decode measurement, median regressions and host-window completeness.
Full local suite43 tests pass. No A3 claim or deployment; active v6 unchanged.
See `02_DYNAMIC/PERFORMANCE_DECISION_PROTOCOL.md` before later default selection.

2026-09-17 checkpoint: Frequency HOLDOUT128 PASS and lossless archive verified;
see `01_STATIC_FREQUENCY/FREQUENCY_HOLDOUT_CHECKPOINT.md`. NPU-hit rate26.2652%
vs Uniform12.4405%, CPU assignments -15.7888% relative. Attribution raw2176,
included2048, unused tails128, unknown0; completed owned-process reaping.
Static driver is now Frequency CMMLU (last sampled163/256, correct112;
partial counts are not a quality conclusion). Preserve the healthy driver,
which proceeds automatically to observer-OFF ABBA if quality passes. No static
baseline selection yet. Fresh host observer V2 was EXCLUSIVE at
1789654100.9233055 with blockers[], outside CPU~0.719cores, owned swap0 and
unchanged pswpin0/pswpout1. Later performance needs its actual window audit,
not this single sample. This checkpoint does not change production semantics,
mask, quality threshold, or historical numerical status.

Continue useful preparation while a healthy long stage runs, without disturbing
it. Notify only meaningful stage transitions, failures/fixes or final decisions;
unchanged polling stays quiet. Stop only owned workers on resource/device errors,
never business workloads. Never claim LOCAL_VERIFIED as A3_VERIFIED_READY.
On truthful Round7 completion, pause the continuation heartbeat and report all
remaining limitations. Do not silently skip expensive required gates.
