# Current execution — 2026-09-19 01:28 UTC

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
