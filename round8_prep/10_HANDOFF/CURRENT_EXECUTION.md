# Current execution — 2026-09-19 01:28 UTC

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
