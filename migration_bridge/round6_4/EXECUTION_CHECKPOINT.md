# Round6.4 execution checkpoint

## Approved authority

User approved complete Round6.4 plan and 2pp absolute per-benchmark quality margin.
Read approved plan and `00_BASELINE_AND_SCOPE.md`. No repeated approval necessary.
Old Round6.3 stays REJECTED; do not run its candidate-order evaluator as this round's gate.

## Workspace

- `/home/admin/Documents/ChatGPT/KT/kt-round6-deferred`
- branch `codex/round6-4-official-deferred-approximation`
- start `ba3494d9aedcc0e5623157832134e1e32a8ea44a`
- approval commit `404274e17`; compact tooling commit `c8a08e5f0`
- child `f02f304a25364d1e8fe9c52515a22f20f5089a52`, unchanged
- inherited llama.cpp dirty: preserve, never stage/discard.
- No push/main merge/new READY tag yet.

## A3 current

SSH admin@110.123.0.3; root PTY was available in session 92973.
Only container `kt-r6-deferred-ascend`, NPU0, CPU0–15.
Never stop other users' tasks or mutate host libraries.
Actual checkout+overlay remains distinct from local HEAD (see preflight provenance).
Local SCP staging `/home/admin/r64-stage-4a6dFz` on A3.

Preflight `/workspace/results/r64-preflight-v1`: 19 passed, EXIT_ZERO; archived locally.
v2 accidental duplicate metadata startup cancelled before tests, not evidence.

**Healthy current run: do not restart**

`/workspace/results/r64-compact-v1`
driver `/workspace/results/run_r64_compact_controls.py`
driver log `/workspace/results/r64-compact-v1-driver.log`
Sequential P1_B0/P2_B0/P3_B0/D1_NODEFER/D1, 20 requests per profile.
Each profile root has guard/result.json, report.json, requests.jsonl, readiness.json.
Driver stops on first failure, guard cleans only owned server.
Top-level progress.json currently written on driver completion/failure, not per request;
inspect driver log and active owned PID before considering missing file a failure.

## Immediate continuation

1. Check healthy run; archive each final report/log/hash. Do not launch competing model.
2. Audit production diff/reference and finish WP0 freeze work concurrently offline.
3. Complete missing R2A/expert safety, nondefer compact and D1 system/no-hit prerequisites.
4. Adapt retained D1 runtime runner to new approved protocol (old r63 runner wrongly
   requires rejected candidate). S6 12×10×64 +8/16/32, restart12×64, observer OFF.
5. Explicit oracle four preselected engineering requests + actual 1000-forward lifecycle.
6. Batch1 runtime PASS then early batch2/4/pilot. No batch>4 or D2/D3 expansion.
7. Freeze GSM8K test full + CMMLU stratified2048, source/revision/templates/parser,
   paired score/exact interval implementation, synthetic tests and power scenarios.
   Contract currently explicitly forbids primary quality until this preparation complete.
   Old context512 insufficient for GSM input plus512 output; explicit equal OFF/ON
   capacity adaptation and validation required, never truncate prompts silently.
8. HD3/FD3 unobserved-input audit; use diagnostics not old logit exact gates;
   proceed quality/performance/stability/ADR per approved plan.

Same-path/system failures halt affected configuration. Finite ON/OFF numerical drift alone
does not block quality. 2pp is approved and immutable; confidence-crossing is INCONCLUSIVE.
No quality/acceleration claim until actual evidence. Default Deferred OFF.

## Scheduling

Reuse existing heartbeat id `round6-3` with Round6.4 prompt, no duplicate monitor.
Quiet on unchanged health; notify on milestones/failure/completion. Continue useful local
preparation while A3 runs. Pause after scope completion or terminal unresolvable blocker.
