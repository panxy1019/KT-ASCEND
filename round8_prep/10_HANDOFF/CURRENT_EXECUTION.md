# Current execution — 2026-09-19 01:28 UTC

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
