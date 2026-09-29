# Round16 validate current execution

Updated: 2026-09-28T22:49:53Z.

## Resume 01 after externally changed resource state

The user reported physical device 12 free. This is the explicit external-state change required by the historical six-cycle terminal receipt; the old `ROUND16_PARTIAL_WITH_GAPS_EXTERNAL_RESOURCE` artifacts remain immutable historical evidence and do not block the resumed audit.

At 2026-09-28T22:42Z, a fresh read-only precheck found no external container claim, no device process and no Round16-owned process on physical12/card6/chip0. Two independent 60-second admission rounds then completed 12 of 12 ten-second windows. The minimum selected-core mean idle was 0.9950386622530794. Minimum NUMA free memory was 213.781 GiB on node2, 216.142 GiB on node3, 64.977 GiB on node4 and 222.169 GiB on node7. All frozen process, HBM, training-container and identity gates passed.

No container, NPU context, model process, generation attempt or device-component process was created. Round16 usage remains zero. The active next step is to finish and statically validate the fresh A1 six-request driver/controller, because the previously frozen `execute_controller` deliberately accepted only an injected callable and the copied v295 launcher still carries historical v295 container/evidence/budget identities. After that preflight, physical12 ownership is checked again immediately before block A starts. A and B device work remain serialized.

Authoritative resumed evidence: `RESUME01_REMOTE_RESOURCE_RAW.txt`, `RESUME01_DEVICE_CLAIMS_RAW.txt`, `RESUME01_ADMISSION_POLICY.json`, `RESUME01_ADMISSION_PRECHECK.json`, `RESUME01_ADMISSION_ROUND1.json`, `RESUME01_ADMISSION_ROUND2.json`, `RESUME01_ADMISSION_RESULT.json`, and `RESUME01_STATUS.json`.

Active status: `ROUND16_ACTIVE_RESUMED_A1_ADMISSION_PASS`. The earlier
`ROUND16_PARTIAL_WITH_GAPS_EXTERNAL_RESOURCE` disposition remains preserved
below as the terminal result of the prior unchanged resource state.

Round15 is terminal as `ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION`; cleanup and owned lease release remain verified. The selected S15 is the retained Round14a S0. Round15 was not reopened and no Round15 candidate was silently promoted.

A1 now has a strict fresh-evidence analyzer and immutable two-block controller. A2 can be selected only when both fresh blocks show direct CPU-result exposed wait as the first actionable region, at least 10% of direct Decode wall, plus an unused same-layer NPU window. A3 can be selected only when both blocks show direct Host launch/metadata cost at least 10%, first actionable, and covering a region beyond the rejected resident-only graph. GDN or NPU kernel duration cannot stand in for either gate. No fresh device evidence exists yet, so `selected_candidate=null` and status remains `DEVICE_EVIDENCE_PENDING`.

B now has the exact eleven-source/config preflight, atomic observed-digest token, source-bound physical-Q32 allocation checks, live 512-entry manager-map acknowledgement, BF16 physical32 w13/w2 checks, cumulative context capacity >=256, and a two-phase KT contract ordered as CPU submit -> preselected native branch -> CPU sync/drain -> merge. Because the frozen KT ABI has no work ID, the offline adapter permits one outstanding native task and stays fail-closed after ambiguous submit/sync failures. The real Qwen `self.experts` runner, preselected-Q32 native operation, local512 metadata replacement, S15 weighted aggregate parity, real stream cancellation/drain, resident receipt, complete layer, TP1 eager model and requests remain device pending.

The earlier integration milestone passed 209/209 tests. The newer B source-bound Qwen runner milestone and its combined 220/220 result are recorded below. The one merge-only test edit preserves the stricter shared `RouteBatch` rejection point; no runtime validation was relaxed.

Fresh admission cycles 1 through 5 created no container and consumed no device/model budget. Their last evidence showed external container claims covering physical devices 0 through 15. No external process or container was modified. The next device action is the sixth and final independent read-only admission cycle. A target still requires an exact identity match followed by two fresh 60-second CPU/SMT, NUMA, HBM, process and training-cgroup rounds before any NPU initialization.

Round16 usage remains zero of 14 model groups, 128 generation attempts, 6 abort controls, 16 device component processes and 32 NPU device-hours.

## Admission cycle 05

At 2026-09-28T18:50:52Z, a new read-only snapshot again found sixteen external `qwen38-vllm-*` containers with exact claims covering physical devices 0 through 15. Every explicit candidate failed `EXTERNAL_CONTAINER_CLAIM`; CPU/NUMA sampling was short-circuited and no device process or container was created. There was no Round16-owned remote process. This is failed admission cycle 5 of the allowed 6; all device/model/request usage remains zero. A sixth cycle will occur only after another bounded offline B runner-integration milestone.

## B source-bound Qwen runner milestone

At 2026-09-28T19:19:04Z, the bounded offline milestone required before the final admission cycle passed. The exact-source audit found a preselected Ascend seam at `MoECommMethod.fused_experts`, restricted the first possible device candidate to TP1/EP1 eager ALLGATHER BF16 physical Q32, and rejected the old per-occurrence native-result ABI for the live path.

The integrated CPU-only Qwen-shaped test now enters the bridge through `self.experts.forward`, acknowledges one identical placement plan across all 48 owner/runner pairs, runs the upstream selector exactly once on the prepared router tensor, propagates prepared hidden and valid-row state, orders CPU submit before native weighted aggregate and CPU sync afterwards, and leaves routed scaling/shared/output transform/reduction/residual with their original owners. The old occurrence mapping callback is optional and confined to the legacy arithmetic mock. `live_eligible` remains false because the exact image, physical32 device adapter, one-layer numerical parity, real stream drain and TP1 model have not run.

The combined integration tree passes 220/220 tests plus compileall, all 49 then-current JSON artifacts parse, the preflight overlay is byte-identical and `git diff --check` passes. The milestone receipt is `ROUND16_OFFLINE_INTEGRATION_RECEIPT.json`; B's specific receipt is `../round16-b/QWEN_SELF_EXPERTS_WIRING_RECEIPT.json`. No NPU, C128 bank, model, generation, abort or device component budget was consumed. This milestone opened the sixth and final independent read-only admission cycle recorded below.

## Admission cycle 06 and terminal disposition

At 2026-09-28T19:21:44Z, the sixth independent read-only admission snapshot found one external `qwen38-vllm-*` container claim for every physical device 0 through 15. Several even-device containers had restarted within roughly one minute and were loading; an instantaneous low HBM reading therefore did not supersede their ownership declarations. All explicit Round16 candidates failed `EXTERNAL_CONTAINER_CLAIM` before CPU/NUMA sampling, no Round16 process or container was created, and no external workload was modified.

The frozen failed-admission limit is now 6 of 6. There will be no seventh unchanged retry. Device/model/request usage remains zero. A fresh critical-path result, A2/A3 selection, B REAL_MOE, complete layer/model, A/B request comparison and all performance conclusions remain `NOT_RUN` or `NOT_MEASURED`, never `NO_GAIN`.

The terminal state is `ROUND16_PARTIAL_WITH_GAPS_EXTERNAL_RESOURCE`. S0 remains the default. Resume only after an externally changed resource state leaves one explicit candidate without an external container claim and that target then passes two fresh 60-second full admission rounds. The authoritative terminal files are `ROUND16_FINAL_STATUS.json`, `ROUND16_FINAL_REPORT_ZH.md`, `ROUND16_RESOURCE_RETIREMENT_RECEIPT.json`, `ROUND16_VALIDATE_BUDGET.json` and `P0_ADMISSION_CYCLE_06.json`.
