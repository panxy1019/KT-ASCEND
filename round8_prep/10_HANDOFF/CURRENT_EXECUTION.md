# Current execution — 2026-09-19 01:28 UTC

## UPDATE 18:31 UTC — both Dynamic gates audited; formal performance dispatched

Heldout-v3128/128PASS,8320updates,40layers x208consistent generations;
96triggeredrequests. CMMLU-v2 re-audit224/256PASS. Release and raw hashes:
11_DYNAMIC/HELDOUT_V3_RELEASE_AUDIT.md. Combined3.1MiBarchive copied locally
under raw/qwen-dynamic-qualified-v2-v3.tar.gz, SHA verified equal to .22 copy.
Candidate qualification-release.json published after independent review.

Owned systemd unit kt-round8-dynamic-perf-v2 launched at18:30:26UTC using
qwen_dynamic_retry_release.py --performance. Output qwen-dynamic-perf-v2;
first Static cell passed admission and is loading. Verify live before action;
do NOT duplicate. Three ABBAblocks12cells24requests,short64/32 medium256/64,
no hooks or long workload,NPU2CPU128–191. Formal result pending final audit.
Round9 PAUSE_DOWNLOAD is set and child group confirmed stopped. KEEP marker
throughout ALL cells; remove only after campaign ends/no timed work remains.
No Round8 completion or Dynamic speedup claim yet.

18:36UTC follow-up: firstStatic14/14shardsloaded, healthHTTP200,warmupPOST200,
timed client active. Actual protocol input/output lengths confirmed64/32 and
256/64; do not infer input length from padded server log rows. Healthy launch
confirmed; retain controller and performance reservation until campaign exits.

## UPDATE 17:41 UTC — Dynamic CMMLU complete and independently audited

CMMLU-v2:224/256,invalid0,delta+0.78125pp vs Q64 reference222. Frozen scorer,
ordered inputs/payloads, candidate and resource evidence independently PASS.
217/256 requests triggered actual updates. See11_DYNAMIC/CMMLU_V2_INDEPENDENT_AUDIT.md
for raw hashes and scope. Raw remains on persistent .22 host; second archive
pending, local desktop only~70MiB free. Heldout-v3 remains healthy101/128;
preserve it. No combined release/performance launch until full held-out audit.
Round9 download remains healthy18complete shards/54,539,657,864bytes,OOM0.
Do not interpret collected shard count as verified model identity.

## Download preparation authorized concurrently — performance interlock

Latest user authorizes Round9 source download/check/isolated preparation before
Round8 ends. Owned .22 systemd unit `kt-round9-model-download` performs bounded
download/verification, no NPU. Before formal performance create
`/home/admin/kt-round9-qwen38/PAUSE_DOWNLOAD`, confirm `download-state.json`
PAUSED and child group stopped, then allow disk activity to settle and recheck
isolation. Keep marker for ENTIRE timed campaign; remove afterward. Do not
stop current Round8 quality/held-out for downloads. Details and live paths in
`migration_bridge/round9/CURRENT_EXECUTION.md`. Round9 NPU model runs and full
F32 conversion remain gated; source download is not runtime admission.

## Latest route — Round8 closure, then Round9 (2026-09-19)

User authorized automatic continuation after truthful Round8 closure into
Qwen/Qwen3.8-Flash-Next. Read `migration_bridge/round9/EXECUTION_PLAN.md` and
`migration_bridge/round9/CURRENT_EXECUTION.md` before transition. No Qwen3.6.
Only lightweight Round9 preparation before Round8 final audit/local commit;
preserve current healthy jobs and finish authorized gates. Positive Dynamic
or Deferred speedup is not a prerequisite. No heavy Round9 tasks started.
Heartbeat should remain active after Round8 and proceed on the current
authorized .22 environment subject to fresh resource admission.

## UPDATE 15:46 UTC — original failing request replay PASS; release adapter ready

Heldout-v3 holdout_032 completedHTTP200,880input/16output,80layer updates,
elapsed482.895s. Original300s watchdog failure is eliminated on this request;
the complete128 gate remains RUNNING. CMMLU-v2 healthy (>84/256). Preserve both.
No offset warning, watchdog, traceback, OOM or cgroup swap in current samples.

Use NEW explicit attempt adapter after both complete:
`python3 tools/qwen_dynamic_retry_release.py` on .22 host. It verifies frozen
candidate/retry/ack/config hashes, requires old v2 stillFAILED, and runs the
original complete-input/score/generation/resource audit against heldout-v3 and
cmmlu-v2. It only prints a release proposal, never auto-writes approval.
Independently review it, archive complete raw evidence/hashes, then write
`qwen-dynamic-candidate-v2/qualification-release.json` using that proposal
(includes heldout_attempt and retry_sha256, both mandatory).

For formal performance use
`python3 tools/qwen_dynamic_retry_release.py --performance`.
It re-audits both stages before validating release and invokes frozen performance
with only the explicit heldout summary path changed. Timed configs/timeout/
workloads/isolation unchanged. Old direct audit/performance entrypoints still
point toFAILEDv2 and must NOT be used. Do not overwrite/symlink old evidence.
No performance release or new timed jobs created. Local29testsPASS.

## UPDATE 15:27 UTC — heldout v2 watchdog failure; v3 RUNNING

v2 heldout FAILED on holdout_032: scheduler watchdog300 expired during cold
audited layerwise prefill; migration continued until SIGQUIT. No OOM/offset
warning/weight assertion. Raw failure archived with SHA in
11_DYNAMIC/HELDOUT_WATCHDOG_RETRY_V3.md. CMMLU v2 healthy (>44/256), preserve it.

New versioned controller `tools/run_qwen_dynamic_heldout_retry.py --mode heldout`
uses qwen-dynamic-heldout-v3 and retry manifest qwen-dynamic-heldout-retry-v3.
Only watchdog1800 and attempt paths differ; original candidate immutable.
Retry metadata committed BEFORE acknowledgement/upload/launch:
`a39977ffa083ed8988286346429c43c7f69398bd`. Retry SHA
`9aba22dfd777053db321b148face7bda631fade57e526afc17be79b15178ad33`.
v3 admitted and RUNNING,2/128 returnedHTTP200; controller log
`dynamic-heldout-v3-controller.log`. Offset warning0; original failure at
holdout_032 has NOT yet been replayed successfully. Do not call fix closed.
Preserve this healthy new attempt. Full128 replay, no sample deletion or
threshold refit. Lane-a remains NPU4/CPU0–63 only. CMMLU60/256 healthy.

Final audit/performance currently hardcode heldout-v2: MUST add explicit v3
attempt-path adapters before release. Never replace old failed evidence with
v3 or claim old candidate passed unchanged orchestration. Formal performance
remains gated; CMMLU/tools/model implementation stay untouched.

## UPDATE 14:49 UTC — candidate committed; two evaluation lanes healthy

This entry supersedes older launch instructions below. Candidate commit
`9a183f309e1db794859646686431d9a1fbeb8819` precedes every held-out/CMMLU
request. Candidate SHA256
`69892f31dcbb450105a4adf153622c7bf6f5823292bfb8038ac9ae60160e55b9`;
threshold256/chunk512. Exact commit acknowledgement is installed remotely.
DEV1024 remains skipped by user; its original failed trigger evidence retained.

Fresh hook-free restart is independently audited 5/5 token arrays exact,
no offset warning. Do not rerun it. The transposed NZ migration fix, red/green
operator regression and actual model evidence are already locally committed.

On .22, healthy new controllers (verify live state, never duplicate):
- heldout: PID1462515, lane-a/NPU4/CPU0–63/NUMA0,1,
  `qwen-dynamic-heldout-v2`, log `dynamic-heldout-v2-controller.log`.
- CMMLU: PID1463082, lane-b/NPU6/CPU64–127/NUMA2,3,
  `qwen-dynamic-cmmlu-v2`, log `dynamic-cmmlu-v2-controller.log`.
Paths relative to `/home/admin/kt-round8-qwen35`. Both use independently
admitted384GiB containers and frozen candidate tool/config hashes.
Both are RUNNING. At14:49UTC heldout25/128 (initial short no-trigger rows),
CMMLU1/256 completed with40 real layer updates; second request actively updating.
All completedHTTP responses200;12resource samples/lane, cgroup swap0, OOM0,
tracebacks0, offset warnings0. HostMemAvailable630GiB; container usage131.2/
224GiB versus384GiB limits. This is healthy startup, NOT completed qualification.
Later heldout medium/long rows still must establish complete trigger coverage.

When both finish, independently audit raw128/256 input IDs, manifest/payloads,
generation/weight evidence, scores and resources via
`tools/audit_qwen_dynamic_results.py /home/admin/kt-round8-qwen35`.
Only actual PASS permits `qualification-release.json` in candidate directory
with heldout/cmmlu PASS and exact respective summary SHA256 values.
Then `tools/run_qwen_dynamic_perf.py` reruns fresh Static/Dynamic on prep NPU2,
CPU128–191, three ABBA blocks, short64/32 and medium256/64; no hooks or long
requests. No timed performance while either parallel lane is active.
Do not interpret parallel quality latency as formal performance.

Heartbeat `round7` remains ACTIVE every15minutes. No .16, push/main/tag.

## UPDATE — user authorizes repaired Dynamic evaluation; skip DEV1024

Active .22 `q32-dynamic-restart-v2` on lane-a is the fresh hook-free DEV replay.
Preserve it. Its controller waits for fixed `q32-dynamic-transfer-v2`, now
PASS120updates/40layers/3generations, no offset warning. Pre/post-fix five token
arrays exact. Root cause localized to copying transposed NZ expert slices;
independent regression old FAIL warning / new PASS exact. Fix is separate
`sglang-python-dynamic-v2`, local runtime source commit7096344e9; Static original
untouched. Original actual-path trace `q32-dynamic-warning-v1` preserved.

Do NOT repeat DEV1024. After replay independent audit, run
`prepare_qwen_dynamic_candidate.py` on host. Retrieve candidate bytes, local
commit before held-out, create freeze-ack with exact commit/candidate SHA.
Then `run_qwen_dynamic_evaluation.py --mode heldout` (lane-a NPU4/CPU0-63)
and `--mode cmmlu` (lane-b NPU6/CPU64-127) can run concurrently. Each enforces
frozen inputs and admission. Do not deploy .16, push, merge or tag.

After BOTH collected, use `audit_qwen_dynamic_results.py` and independently
review outputs/resources. Write qualification-release.json only on PASS,
then run `run_qwen_dynamic_perf.py` on prep NPU2/CPU128-191. It refuses active
parallel lanes and requires both stage releases; three short/medium ABBA blocks,
no long request, no hooks. No formal performance while quality lanes are active.

Q32 stability A+B now2304tokens, within/cross-process exact; cgroup and HBM
plateaus documented in11_DYNAMIC/STABILITY_AND_PERFORMANCE_AUDIT.md. Do not
call cgroup usage process RSS or claim zero leak. Staticv6 archived but strict
host-transfer isolation remains limited; Dynamic comparison reruns freshStatic.

## UPDATE 13:40 UTC — independent Dynamic entry audit

See `11_DYNAMIC/DEV_ENTRY_AUDIT.md`. Raw256/512 independently checked:
120 updates each,40 layers,3 generations; actual update rows512 for both.
DEV1024 FAILED trigger coverage with frozen chunk512; preserve failedv1.
NZ slice-copy/equality/host-oracle probe passed but did not reproduce/clear
the model storage_offset warning. Next: trace actual update/compute operands,
independent oracle, restart and observer parity, then commit DEV selection.
Do not launch held-out/CMMLU yet. Formal performance additionally requires
hook-free isolated window. Stability v3-resume B remains healthy; preserve it.
No production patch, quality/identity change, push, merge or tag performed.

## UPDATE 13:25 UTC — DEV256/512 done; stability B admission retry

Both Dynamic DEV summaries report PASS,120 update rows each (40layers x3).
Independent audit and storage_offset warning investigation remain pending;
no held-out authorized. DEV1024 dispatched on lane-a using its frozen candidate.
Check chunk512 vs threshold1024 trigger behavior explicitly before acceptance.
StabilityA completed18 requests/1152tokens. B launch was correctly rejected by
resource guard:894 swap-in pages/31.7s, zero swap-out, zero owned cgroup swap.
v2 remains FAILED at admission, not a token/determinism failure. Resume tooling
preserves A and starts only B under fresh v3-resume identity with the same
resource guard and comparison to A reference tokens. No limit was relaxed.

## UPDATE 13:08 UTC — three lanes started; foreground monitoring may end

User requested monitoring until all three tests had stable starts, while keeping
heartbeat active. Stability campaignA has3 completed64-token responses. Dynamic
256 and512 each returned the no-trigger request and entered real layerwise
prefill/update, with9 and8 migration audit rows respectively and continuing
layer progress. Early health timeouts resolved after cold initialization.
Available host memory558GiB. No test completion or Dynamic qualification yet.
Dynamic256 logged an Ascend storage_offset warning during first full-resident
compute. Preserve and investigate it before qualification: exact copied-weight
audits do not prove grouped-kernel output correctness. Continue monitoring
through heartbeat; do not restart healthy lanes merely for this note.

## UPDATE — user-authorized three-container .22 execution

Static v6 collection completed12/12; raw evidence retrieved locally, isolation
audit pending. User authorized two additional .22 containers. Idle owned
Round7 a/b containers (only sleep infinity) were stopped, preserving files.
New dynamic-a uses NPU4/CPU0-63/NUMA0,1; dynamic-b uses NPU6/CPU64-127/NUMA2,3.
Both passed real device enumeration and tensor copy/synchronize. Original prep
container remains NPU2/CPU128-191/NUMA4,5 for Q32 stability-22-v2. All C64=2x32.
The stability controller was dispatched. Dynamic256/512 DEV candidates use
the two new lanes;1024 follows on lane-a. Scheduler DEV256/HOLDOUT128 frozen
before DEV. No held-out use until candidate selection is committed. Concurrent
results are untimed system/quality evidence; formal performance remains isolated.
All lanes share original model/GGUF inodes. Observe host memory before loading
the third model; stop only owned failing workers. .16 remains cancelled.

## USER OVERRIDE — .16 migration cancelled

User requested removal of this project's .16 migration and all further testing
on .22. Do not reconnect/deploy .16 in future heartbeats. Failed stability raw
logs were retrieved into `round8_prep/raw/secondary-host-failure` locally.
Owned .16 containers, imported runtime image, project artifact/source copies
and four uploaded temporary tools were removed under explicit user
authorization; existing business containers and host driver remain untouched.
Post-cleanup inspection confirmed both project directories absent, only the
original four containers remain, and filesystem availability is 2.9TiB.
After .22 Static v6 audit, adapt admission/container orchestration to .22 before
long stability and Dynamic. Do not run the .16-specific admission unmodified.

## UPDATE — .16 audit: blocked by existing driver namespace ownership

The dispatched stability server never reached readiness: torch_npu reported
zero devices. No stability tokens were generated. Kernel logs report
`uda_occupy_dev_by_ns: Conflict open udevid (udevid=2)`. The authoritative
`/proc/uda/namespace_node` has ns_id=0, root_tgid=370374, devices0-7; Docker
identifies that PID as the unrelated `mzy-mindspeed` container. Do not stop or
alter that container. Empty npu-smi compute-process output was insufficient
admission. Removing ASCEND_RT_VISIBLE_DEVICES and matching SYS_NICE capability
did not restore visibility. Owned failed container v1 is retained under
`kt-round8-qwen35-stability-failed-device-v1`; replacement is stopped after
diagnosis. .16 stability/Dynamic must not launch until namespace allocation is
available and an actual in-container tensor probe passes. .22 work continues.

## UPDATE 12:09 UTC — secondary host ready; stability dispatched

.22 Static v6 completed 6/12 cells and is running block1 position2. Preserve
the healthy controller. .16 transfer completed; the frozen image, model
revision, CPU manifest, Q32 profile and launch-command hashes were checked by
`prepare_qwen_secondary_container.py`. The isolated container
`kt-round8-qwen35-stability` uses NPU2, CPU128-191, NUMA4/5 and 384 GiB.
The two-campaign Q32 stability controller was dispatched; inspect
`/home/admin/kt-round8-qwen35/qwen-q32-stability-v1.log` and its output directory
before any next launch. Container creation is not stability qualification.
After trajectory completion, audit resources and raw outputs before Dynamic.

## UPDATE — independent .16 stability/Dynamic lane authorized

User authorized 110.129.0.16 for the remaining Q32 stability and Dynamic
qualification. Read-only preflight at 10:56 UTC found the same Kunpeng-920
256-core/8-NUMA and eight Ascend 910B3 topology, 975 GiB MemAvailable, zero
NPU processes and no competing project container. Access is through the
110.123.0.3 jump host. Existing business containers remain untouched.

The frozen runtime image, `/home/admin/kt-round8-qwen35` artifacts and frozen
KT source are transferring directly from .22 to .16 under host PID 1038129;
completion requires artifact/hash audit before any model process starts. The
.16 lane will use physical NPU2, CPU128-191, NUMA4/5, C64=2x32, 384 GiB, Q32,
Dynamic/Deferred/Graph OFF for stability. It then uses a separate candidate for
Dynamic DEV selection and qualification. No .16 evidence is merged into .22
timing evidence. The heartbeat now tracks both machines.

## UPDATE — complete ABBA orchestration launched after first-block audit

The original Q64-only block completed at 09:59 UTC: short/medium/long
44.590/88.887/177.805 seconds. Preserve qwen-static-perf-v1; it lacks continuous
resource-window evidence and is preliminary only. It did not automatically
advance because the first launcher only scheduled one cell.

New host controller `tools/run_qwen_abba.py` executes twelve distinct cells:
three Q64/Q32/Q32/Q64 blocks. Output `qwen-static-perf-v2`, host PID1001975,
log `qwen-abba-v2.log`. Each cell has a separate position, fresh serving process,
launch admission, client/server logs and sampled host/NPU/container resources.
Final collection status is pending resource audit, never automatic formal PASS.
Preserve healthy controller. Failure stops advancement; do not overwrite its
state or outputs. Current launch uses the frozen hook-free v2 launch candidates.

After performance, Q32 remains the already system/quality-qualified stability
target; performance can inform deployment selection but does not substitute for
stability. Dynamic requires its own DEV selection, pre-heldout freeze and system,
quality/performance gates. Neither stability nor Dynamic has started yet.

## UPDATE 09:31 UTC — Q32 CMMLU audited and quality screen passed

Q32 completed all 256 frozen Qwen CMMLU items: 225/256 correct, invalid=0,
delta +1.171875pp versus the Q64 engineering reference 222/256. Independent
raw audit passed for ordered item IDs, unique request IDs, exact input token IDs,
candidate IDs, stop and sampling parameters, one finite score position and
frozen scorer replay. Archive `qwen-cmmlu-q32-v1.tar.gz` SHA256 is
`a4ddedc07dd962821124bb4da717da60d8cb9712c9fa4ac017fae2d4142a6e2b`.
The Q32 quality screen therefore passes the frozen minimum 210/256 with zero
invalid outputs. See `Q32_CMMLU_FINAL_AUDIT.md`.

Round8 static Q32 system and product quality screens are now complete. Formal
performance, long-run stability, Dynamic and Deferred remain unqualified.

## UPDATE 09:03 UTC — Q32 system frozen; quality running

Q32 same-process 32 requests, clean-restart four 32-token trajectories and
32 distinct prompts × 2 repeats all pass. Independent raw archive audit also
confirmed finite output logprobs, nonempty output and 40/40 actual NPU slots=32
with the shared-expert oracle. The immutable system archive is
`q32-system-evidence-v1.tar.gz`, SHA256
`6c7ae7a527bc94ab02411fb9db7aab5381ebdc7cac8f3a07711f1901ec53e21f`.
Pre-quality local freeze commit is
`7b214c9312b897c8258e2e680497f45c670ca7f6`.

Q32 CMMLU 256 is now running in the existing owned `.22` container on NPU2.
Raw output is `/home/admin/kt-round8-qwen35/qwen-cmmlu-q32-v1/raw.jsonl`,
progress is alongside it, and the log is
`/home/admin/kt-round8-qwen35/qwen-cmmlu-q32-v1.log`.
Entry record asserts the frozen Qwen manifest SHA
`365f557c0746db1e02ddee4ec66c7afb5e72eeb17e8dd8e0a47d1ed2b7af1b9d`,
Q32 profile SHA and Q64 reference 222/256. The original observed −5pp rule
requires at least 210/256 and zero invalid outputs. The first request completed
successfully. Keep the healthy run intact. After 256 rows, independently audit
ordered IDs, input IDs, stop/sampling/candidate fields, finite scores, scorer,
request IDs and manifest hash before deciding the quality gate. Diagnostic hooks
remain enabled, so these elapsed times are not formal performance measurements.

## UPDATE 02:33 UTC — Q64 reference frozen; Q32 entry active

Q64 quality completed222/256,invalid0; independent raw auditPASS (256orderedIDs,
inputIDs,params,candidateIDs,requestIDs,finitecandidates,re-score). Actual1689
source/nativehashes rechecked0mismatches beforechangingdiagnostictools.
ArchiveSHA8942a99c199e69ac3c06731092081365f0c5f8566466490536bf3e2f053e6ec0.
Reference freezecommitddab2f60d334c93040475d1aef92dd5b91eddfb3; see
Q64_REFERENCE_AUDIT_AND_Q32_ENTRY.md. Q64 is engineeringreference,notAllNPU.

**Q32 bring-up is current owned work.** Samecontainer/NPU2/CPU128–191/NUMA4/5,
model/GGUF/tokenizer,C64 unchanged. OldQ64server stoppedaftercomplete evidence;
Docker exitedwithOOMKilled=false. Newhostdriver3597607, log`q32-bringup-v1.log`.
FrozenoriginalprofileSHA b358399e6405d60b81d8231529b1ec6024ff7138cddab2e0cbac3ccf0b25f486.
Candidate`q32-launch-v1`, mask32/256eachof40layers. AdmissionPASS. Diagnostic
hooks adaptedonlyto expectedresidentcount; no productionarithmetic change.

02:37UTC: serverready,40/40structure/sharedoracle rows confirmed; first32token
Chinese request andfirst16tokenprefix/repeat PASS. Same-processmatrix healthy
at2/32requests. No OOMevents orcgroupswap. Keepthisrun intact.

Hostdriver3597608 waitsactualQ32health+40layer32slot/sharedoracle, thenruns
`q64_compact.py --profile q32 --phase same`. Log`q32-compact-same-v1.log`, output
`q32-compact-v1`. Do notduplicate/restarthealthy run. This chain STOPS after
sameprocessmatrix: cleanrestart is next, NOTquality. Newseparate restartcandidate
directory/log mustbe used to avoid mixingstructure/rawprocesses.

Next: auditQ32same32requests→cleanrestartfour32tokenprompts→32distinctprompt
screen→samefrozenCMMLUmanifest (Q64reference222;Q32minimum210/256 underold−5pp).
Do NOT run oldQ64qualitytool unmodified forQ32: its predecessor/output paths
are Q64-specific. Generalizewith explicitprofile and separate rawoutput, preserve
exactmanifest/scorer. No samplechanges,thresholdrefit ormaskreselection.
No Q32quality,formalperformance orRound8completion claimed.

## UPDATE 01:55 UTC — supersedes running32prompt screen below

32distinctprompt/64response screen completedPASS; rawaudit complete. Additional
productionNPU grouped vsLLAMAFILE vsCPUcanonical probesL0/20/39 completedPASS,
qlen1/8/64/512 and8onehotexpertcontrols, actualNZformat29 asserted. See
Q64_SYSTEM_TO_QUALITY_FREEZE.md. Originalv1 ND-layout probe preserved separately.

**Healthy freshQ64 CMMLU256 now running** on same server/container. Do not
restart/duplicate or launch competingNPU/CPU-heavytests. Hostdriver3444412,
log`/home/admin/kt-round8-qwen35/qwen-cmmlu-reference-v1.log`, outputdirectory
`qwen-cmmlu-reference-v1`. First3requests completed validly atlastinspection.
FrozenmanifestSHA365f557c0746db1e02ddee4ec66c7afb5e72eeb17e8dd8e0a47d1ed2b7af1b9d,
pre-scorefreezecommit5b828fe3c884c0e502337bd998dfdd678b4fd089 andfreezeack deployed.
Identical256CMMLUquestions/fewshots tooriginal; Qwen tokenizer/chatwrapper,
thinkingOFF,first-output-positionA/B/C/D argmax, noDSV2tokenIDs/scores reused.
This is untimedquality withdiagnostic hooks, NOTperformance.

Afterfinish, audit256orderedIDs,rawrequestinputIDs,4finitecandidate scores,
argmax/tie scoring,invalid0,manifest/ack andcurrentruntimeprovenance. Final
summaryREFERENCE_SCORED_PENDING_AUDIT is notautomaticQ64promotion. Review score
sanity andcomplete remainingentryevidence beforeQ32. Do notresumeRound7.
Localtests22PASS. Userallowshandoffhealthy longquality/performance runs.

Latest user authorized continuous operator migration, full40-layer conversion,
Q64-C64 static bring-up; hand off healthy long generation/quality/performance
checks. No push/main merge/tag. No Q32 promotion before Q64 predecessor audit.

## Completed this turn

- L0/L20/L39 real256-expert CPU operator tests: fixedtop8, qlen1/8/64,
  tenrepeats exact, finite, relative-L2≤0.01.
- Full40-layer F32 GGUF conversion:120tensors,30,720 expert-projections exact
  against pinned BF16 source,120GiB payload. Manifest35ca0ac63a3646ba7b89675053bbead19724325fa9d73f99910ef3c156a84405.
- Q64 loaded fullmodel and generated8tokens;40layers mapping/resident/dtype
  finite checks PASS. See Q64_INITIAL_BRINGUP_RESULT.md.
- Process A four-prompt32request matrix PASS: six16token repeats,32token
  baseline,8token prefix. Do NOT call this32distinctprompts.
- Process B clean restart4/4 32token trajectories exact to A.
- Shared oracle40/40 PASS: first-forward shared/gate/routed captured once,
  routed+sigmoid(gate)*shared equals finalMLP output exactly. First forward
  may be server health generation; it is not scored quality evidence.
- New diagnostic observer did not alter the four compared restart trajectories.
  This is bounded observer evidence, not proof for all possible prompts.
- Local regressions19PASS. Three launch/config failures resolved without
  core MoE/GDN/model arithmetic modifications; logs preserved.

## Healthy background run — DO NOT duplicate or restart

Host110.129.0.22 (through110.123.0.3 jump), ownedcontainer
`kt-round8-qwen35-prep`; physicalNPU2/logical0, CPU128–191,NUMA4/5, C64=2×32,
memory384GiB. OldownedRound7NPU2container remains stopped. Unrelated workloads
untouched. `docker inspect`/fresh resource inventory remains required.

Server hostdriver3295331; serverlog
`/home/admin/kt-round8-qwen35/q64-bringup-v8.log`, launchcandidate`q64-launch-v6`.
Generation chain hostdriver3302539, log`q64-restart-screen-v1.log`.
Current32distinctprompt×2repeat×8token screen at2/32 prompt pairs PASS when
inspected. Outputdirectory`q64-generation-screen-v1` under same ownedroot.
No OOM events and cgroup swap0. Not a CMMLU or performance run.

Inspectsummary.json only afterprocessends, audit64rawresponses and exactID
coverage againstprotocol. Any ordinary failure: preserve, reproduce, repair
minimally, regression, new run identity; never overwrite failedrawfiles.
Do not use timed numbers with these sync-heavy diagnostic hooks as performance.

Restart raw archive`q64-restart-evidence-v1.tar.gz` SHA256
700527f84f094e3412f11fbe1b15592e6651f264e54810623e0e4588659ff48a.
Includes rejectedresourcewindows unchanged; newresourceguard allows≤1MiB
historicalreadback/≥30s but zero swapout/ownedcgroupswap, fullRAMreserve.
See07_BUILD/FULL_MODEL_MEMORY_ADMISSION.md for rationale and exact scope.

## Not established

FullQ64 quality, Q32, all-NPU equivalence, production throughput and long-run
memory stability are NOT_QUALIFIED. Q64 is engineeringreference64/256resident,
notAllNPU. Afterdistinctpromptscreen, audit remainingoperator/systemgates and
freezeQwen prompt/thinking/scoring protocol before fresh CMMLU reference.
No automatic reuse of DSV2 scores or quality claims.
## UPDATE 2026-09-19 — formal performance scope reduced before v5

Per the user's explicit scope decision, formal Static performance now keeps
only short and medium batch-one requests. Short is reduced from 128 input / 64
output tokens to 64 / 32; medium is reduced from 512 / 128 to 256 / 64; the
1024 / 256 long request is removed. Three ABBA blocks remain, so the formal v5
campaign is 12 clean-process cells and 24 measured requests. The active v4
controller was interrupted before its first cell completed, its state/logs are
preserved, and only the project-owned container is stopped. v1-v4 must not be
merged with v5 because the workload contract changed.

Q32 remains the frozen target for long-run stability. Qwen Dynamic system/DEV
qualification may proceed independently on the isolated .16 host, but final
Dynamic performance must compare against the audited reduced Static v6
baseline under the same workload contract.

The initial reduced v5 controller was then stopped before completing any cell:
its server log path still referenced the historical v2 directory. That is an
evidence-provenance orchestration defect, not a model/runtime failure. v5 is
preserved as FAILED. The corrected fresh identity is v6; its server logs and
measurement outputs are both written below `qwen-static-perf-v6`.
