# Round9 continuation checkpoint — 2026-09-19

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
