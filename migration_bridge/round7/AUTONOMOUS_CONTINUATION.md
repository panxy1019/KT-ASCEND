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

DEV statistics collection attempt2 is running, 17/256 at 11:01 UTC on
2026-09-17. Progress will change; this is not a permanent status claim.
Use read-only checks of:

```
docker exec kt-r67-c64-development-v1 sh -c 'cat /workspace/c64-dev/round7/runtime-v2/progress.json; tail -5 /workspace/c64-dev/round7/runtime-v2/uniform-dev-v1/guard/process.log'
```

Collector log: `collect-dev-v2.log`; run directory `runtime-v2`.
Actual server stdout is `guard/process.log`, not empty `launcher.log`.
Driver is backgrounded through `flock` and an internal exclusive lock.
Do not launch a duplicate or run additional NPU tests while this stage owns NPU.
Do not replace active release-v5 tool files with later local edits: deploy future
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
5. Dynamic deployed-code audit and DEV-only threshold256/512/1024 sweep; freeze
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

## Follow-up policy

Continue useful preparation while a healthy long stage runs, without disturbing
it. Notify only meaningful stage transitions, failures/fixes or final decisions;
unchanged polling stays quiet. Stop only owned workers on resource/device errors,
never business workloads. Never claim LOCAL_VERIFIED as A3_VERIFIED_READY.
On truthful Round7 completion, pause the continuation heartbeat and report all
remaining limitations. Do not silently skip expensive required gates.
