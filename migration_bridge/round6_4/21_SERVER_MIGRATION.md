# Round6.4 server migration — execution in progress

User authorizes continuation on110.129.0.22,2026-09-16. Credentials are not
stored in repository or artifacts. Source110.123.0.3 remains test-paused.

## Inspected destination

- SSH from local host closes before banner; ProxyJump through source works.
- Host gpu-server-01, Ubuntu22.04.5, kernel6.8.0-107-generic aarch64.
- Kunpeng920,256cores/4sockets,1TiB RAM,~3.1TiB disk available.
- Device0 Ascend910B3, firmware7.8.0.6.201, driver25.5.1.
- NPU0 no process at inspection. Existing `verl` running, untouched;
  `verl_train` stopped, untouched. No exclusivity claim for future use.
- Source driver26.0.rc1/firmware9.0.0.0.205 differs materially.

CANN9.0 release-note mapping includesHDK25.5.X:
https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/releasenote/release-notes.md
This permits a compatibility probe, not an assertion of runtime qualification.
910B-specific operator availability and actual device execution must pass.

## Execution boundaries

1. Snapshot stopped project container, transfer only project code/runtime,
   frozen model/GGUF/corpora/tool dependencies; verify hashes.
2. Create dedicated destination container, NPU0only, CPU0–31, TP1/batch1.
   Host driver read-only mounted, no host library installs/upgrades.
3. Test runtime import/device init/transfers, same-operands expert and compact
   OFF/ON determinism. Stop long campaign if these fail.
4. New-host descriptive OFF/ON-forced/ON-native must form a new complete
   cohort. Never append to source partialON or combine sourceOFF with targetON.
5. Final report separates source quality/performance/stability from destination
   bridge/descriptive evidence. Destination performance is not inferred from
   source timing. A new-platform READY claim requires corresponding evidence.

Frozen thresholds, corpora, model/placement/GGUF and defaultDeferredOFF remain.
Original sourceON cancellation is USER_REQUESTED_PAUSE, not a numerical FAIL.
No modification to business containers, NPU1–7 or host libraries is authorized.

Status: DESTINATION_CPU_BINARY_COMPATIBILITY_REPAIR; not complete.

## Transfer and first runtime probe

Destination container `kt-r64-910b-migration`, ID
`6d997f15ef460a16ac88354b405988a99bf9adcb5e6900a748c4f12c43ec216a`.
Actual cpuset0–31 verified. Only device0 plus manager/control devices mounted.
Source stopped-container snapshot imageID601e76d17e9fb4b1c5a72d491cf086c533a309c5c651315e73fe783c90460c8a;
destination Docker imageID307349929be39103b07679ec3bdda2e19efa789322b542e7fa3446ffbf4e81be.
All14 orderedRootFS layer hashes match. Configuration-ID serialization difference
not yet fully attributed; do not call the two imageIDs identical.

All9 model/tokenizer files match sourceSHA256; frozenGGUF andD1placement also match.
Initial torch2.9.0+cpu/torch_npu2.9.0.post2 import/device detection succeeds,
but ones() fails561103/ParseDynamicKernelConfig, including after set_env.
Inspected910B kernel directory contains onlyops_oam; OnesLike config exists
only forascend910_93. This is a missing target-chip operator package, not a
Deferred-model failure. No model test was started before addressing it.

Container-only installation completed from officialAscend Dockerfile URL:
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-910b-ops_9.0.0_linux-aarch64.run
PackageSHA25684621a13bfbd7d319412f4278d382dde804c24ffc3756c5a561da2f6da36944e.
Command`bash <package> --quiet --install --install-for-all` inside dedicated container.
Installer may update associated container runtime components; preserve resulting
version/hash evidence. Hostdriver/firmware remain read-only and unchanged.

## User CPU expansion

User explicitly requests more CPU during migration. Destination container
allocation becomes32physical cores0–31, all on NUMA0. Initial compatibility
bridge retains old16worker/taskset0–15 execution to isolate hardware/runtime
differences. A subsequent32worker execution must be an explicit separate profile
and pass same-path checks before using it for continued testing. Allocation
alone does not imply CPUInfer uses32threads. No other workload affinity changes.

## Target CPU binary repair

New device probe and framework transfer preflight passed19/19 after910B ops
installation. First full-model compact16 failed before requests with SIGILL in
CPUInfer initialization; failure root r64-910b-compact16-v1 is preserved.
Copied extension compile flags explicitly included unsupported+sve+bf16+i8mm.
Target CPU reports none of these features. NUMA membind warnings alone are not
evidence of the illegal-instruction cause.

A separate build from the same sources uses armv8.2-a+fp16+dotprod, disables
SVE/BF16/I8MM, CUDA/KML/AMX, retains Ascend backend. Build completed; isolated
CPUInfer16 constructor/destructor now passes. No model numerical claim yet.
Original extensionSHA3fbb95c00c5906b905e8fd689346f4e78f86714fddb6372fa0e481e0da7ce43f.
Target extensionSHA0f3aac9009931b18853c9b8fd6f6bba8b02d6ea396e79b2029bf9ce092049895.
Both retained under /workspace/results/migration-cpu-build-v1.

CPU regressionv1 stopped during collection because the legacy python directory
symlink was omitted from transfer. Restored the original symlink and launched
new root r64-910b-cpu-regression-v2; no test expectation or exclusions changed.
New binary acceptance is recorded by an explicit migration manifest only after
CPU regression passes, never by rewriting the original preflight provenance.
Local tooling76unit tests pass. Destination full-model requalification pending.

CPU regressionv2 now **18passed/3historicaldeselected**, including1000forwards.
RSS1565622272→1565896704 bytes (delta274432), ten sampled values plateau.
Not a zero-leak or long-duration serving proof. GuardEXIT_ZERO.
Evidence archived under `evidence/migration-910b-v1/`, including both failures,
passing preflight/regression and build manifests/logs. Rebuilt extension is
explicitly allowed only by its old/new SHA pair after regressionPASS; all other
source hash checks remain exact.

Compact16-v2 model readiness succeeded after this repair. The subsequent
queue is `r64-910b-continuation-v1`: compact16PASS ->16realexpert cases ->
compact32 -> independent full3mode descriptive32 -> analysis. It is a migration
and descriptive evidence sequence, not a replacement for destination quality
or performance qualification. Final ADR must retain this scope boundary.

## Completed destination16worker bridge

`r64-910b-compact16-v2`:32/32requests PASS, each mode4prompts ×3repeats,
then4prompts after clean restart.64tokens/request; finite and same-path exact.
This compact bridge is not a fresh full12prompt/10repeat qualification.

`r64-910b-expert-safety-v1`:16/16cases PASS,8selected experts ×tokens1/4,
10CPU repeats each, maxrelativeL2=0.003913063420715329<=0.01.
Crosshost diagnostic:16/16CPU output-hash lists match archived source cases;
0/16entire CPU/NPU metric dictionaries are identical. Do not claim full platform
bitwise equivalence. Same input/model/helper preserved; this is representative
expert evidence, not whole-model quality equivalence.

Build compiler:UbuntuGCC11.4.0. Target flags and compiler-version retained.
Queue advanced to compact32 after these passes; destination32 qualification is
still pending. No acceptance threshold or placement changes.

## Completed destination32worker compact bridge

`r64-910b-compact32-v1`:32/32requests PASS, same4prompt/3repeat/cleanrestart
protocol. All finite; each mode internally exact. Resolved command and CPUInfer
logs identify32threads. Crossworker diagnostic additionally finds32/32paired
output trajectories identical to compact16-v2. Not a formal speedup experiment.

At08:24UTC queue advanced to `r64-910b-descriptive32-v1`; complete new3mode
cohort required. Read-only host monitor3391118 records exact guard descendants,
CPU0–31 andNPU0; bounded5hours and ends on capture terminal progress. No signal
operations in this telemetry tool. Final descriptive and resource audit pending.

User subsequently requests background automatic3mode execution and stopping
monitoring. Host monitor3391118 stopped explicitly; existing partial samples
retained, not a full memory campaign. Test queue remains active, latestOFF20/136.
After all3integrity gates it automatically performs offline analysis. Final ADR
and evidence review remain separate from background execution completion.
