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
