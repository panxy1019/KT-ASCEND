# Round7 autonomous continuation

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
