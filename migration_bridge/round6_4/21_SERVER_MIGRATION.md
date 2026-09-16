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

Status: MIGRATION_PREPARATION, not destination-tested or complete.

## User CPU expansion

User explicitly requests more CPU during migration. Destination container
allocation becomes32physical cores0–31, all on NUMA0. Initial compatibility
bridge retains old16worker/taskset0–15 execution to isolate hardware/runtime
differences. A subsequent32worker execution must be an explicit separate profile
and pass same-path checks before using it for continued testing. Allocation
alone does not imply CPUInfer uses32threads. No other workload affinity changes.
