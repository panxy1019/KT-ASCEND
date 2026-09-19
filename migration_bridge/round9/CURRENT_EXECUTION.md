# Round9 continuation checkpoint — 2026-09-19

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
