# Round6.3 execution checkpoint

Maintainer authorized reviewed execution and autonomous in-scope repair and
continuation. Do not request another routine approval. Failed frozen gates stay
failed, and no downstream qualification skips a prerequisite.

## Completed

- New parent branch from `5e708dc63`; inherited llama.cpp dirty state untouched.
- Route-only QD2 operand audit: 8192/8192 forward union partitions PASS on A3.
  34871 row-low CPU candidates, 34431 cross-row protected, 440 deferred routes.
- Default-OFF teacher-force helper and append-only full-capture index.
- A3 CPU tensor/helper tests: 13/13 PASS; local torchless subset 7 PASS.
- Actual-decode small baseline: 4 requests x 8 targets, 36 forwards, final
  index/physical audit PASS; stopped only its owned server guard.
- Fresh defer0 self-history forcing: 36 full-logit forwards byte-identical;
  additional32 uniform synthetic target injections all differed from natural
  selection; actual next-input checks and final72-forward audit PASS.
- Defer1 forcing: 8 requests x8 targets including synthetic controls, all
  transaction/history checks PASS; final72-forward audit PASS. No cross-mode
  numerical threshold inference from these engineering controls.
- Independent32-prompt pilot source and planning rule preregistered before
  pilot inference; text dedupe against historical corpora/fixtures PASS.

## Running / next

### CURRENT AUTHORITATIVE UPDATE — QD3 IS ACTIVE

Observer correction PASS; fresh pilot complete with1814/2048 exposed positions,
32/32 prompts exposed. Registered planning selected N128. Formal QD3/HD3/FD3
inputs and statistical method frozen in commit8191a8edc before model execution.
Preregistration canonical SHA:
`19685bff9c72337873c466067ba76c579e4e562b25170d84d46e3d5185f83eb0`.

Active A3 pipeline: `/workspace/results/r63-qd3-v1-pipeline`.
Natural mode `r63-qd3-v1-natural-server/session` completed128/128 requests:
8320 physical/indexed,8192 targets,128 lookahead,unknown0,incomplete0.
Natural session ended Unix1789490639.611315. Frozen qd3-force.json canonical
SHA is `35d45fdd9fa8cc453f1b61ee464e3f2261710ea0ebecf2ce7fbca7ed4e79beae`.
Pipeline is advancing to force0. Do not restart a healthy run. It will run
force0 with self-history byte parity, then force1; audit all files, export
causal positions, apply500 exposure gate, and only then derive numerical risks.
The pipeline stops after candidate derivation and NEVER starts HD3/quality.
Expected total~79minutes from start, three8320-forward model runs.

Input/protocol root: `/workspace/results/round6_3-formal-inputs`.
Expected subsequent roots: `r63-qd3-v1-force0-*`, `r63-qd3-v1-force1-*`,
`r63-qd3-v1-defer0-positions.json`, `r63-qd3-v1-defer1-positions.json`,
`r63-qd3-v1-derivation`. Those are not yet completed evidence.
Host read-only telemetry: `r63-qd3-v1-resource-samples.jsonl`.

A3 synthetic/tool tests27/27 PASS; identity-fixture zero reserves are NOT model
qualification. Natural observer controls all PASS. New runtime adapter is
local-only, NOT_RUN, accepts only R6.3 candidate and frozen S6 SHA.
See07 budget and08 runtime notes before proceeding. Candidate must be committed
and all runtime subgates must close before HD3. No production changes after
formal freeze; new runtime tooling is preparation only.

If cancellation becomes necessary, signal the verified owned session controller
and let its model guard clean up before terminating orchestration. Do not kill
an outer short-grace guardian and orphan its nested model process.

### Historical progress (preserved below; superseded by update above)

Long full baseline and compact forced defer0 both PASS: 20x64 targets,
1300 final audited forwards each, 34045 retained tensor comparisons byte exact.
Raw capture shrank from1380241180 to592963662 bytes. See04 report.

Long defer1 full/compact checks also PASS:1300 forwards per mode,35345 retained
tensor comparisons byte exact; full1380686660 bytes vs compact593335350 bytes.
Full/compact final integrity audits PASS. Engineering exposure1120/1280,
150 current route events; repeated4-family smoke is NOT a formal exposure gate.

Protocol omission discovered: full/compact comparisons are NOT true observer
OFF/ON comparisons. Historical pipeline CANCELLED and disabled pending the
missing natural OFF/minimal-logits-reference gate. Pilot baseline had completed;
defer1 had206 indexed forwards when stopped. Preserve both as superseded,
NOT_FOR_CALIBRATION. No size plan or numerical reserves were produced.
See06_OBSERVER_OFF_GATE_CORRECTION.md. Repeat pilot in fresh directories only
after OFF/reference qualification. Frozen pilot inputs/planning rules unchanged.

Natural defer0 OFF control is being launched under
`/workspace/results/r63-natural-off0-retry1-control`, server
`r63-natural-off0-retry1-server`. Initial OFF startup was rejected before any
generation by the legacy missing-capture-directory check; fixed in tooling.
Finite follow-up `r63-observer-correction-retry1-pipeline` only checks natural
OFF/reference/full parity and stops on first failure; it does not start pilot.
Then reference0/full0, natural full1/reference1/OFF1 must be compared without
forcing history. Existing forced defer1 captures cannot substitute natural
defer1 free-generation parity after trajectory divergence.

Defer0 true OFF retry PASS:20/20 natural64-token trajectories exact against
full baseline; HTTP logprobs finite; physical captures0. Evidence archived at
`evidence/natural-off0/`. Minimal-reference0 also PASS:20/20 trajectories and
1300/1300 final-logit byte comparisons exact; archived at
`evidence/natural-reference0/`. Correction pipeline is now running natural
full defer1, followed by natural OFF1/reference1 checks.

Natural full defer1 has completed:1300 physical/indexed forwards,
1280 targets+20 lookahead, unknown0/incomplete0, PASS. Natural OFF1 is active.
A finite `run_corrected_pilot.py --wait-observer` process waits for correction
pipeline PASS (up to2400seconds), checks all six parity files, then uses fresh
`r63-pilot-retry1-*` outputs. It never reuses premature pilot results and stops
after exposure-only sample-size planning. QD3/HD3 are not automatically launched.

UPDATE: observer correction is now PASS (Unix1789487915.5951705).
Defer1 OFF20/20 trajectories exact; reference1/full1 1300/1300 logits byte exact.
All natural OFF/reference and full/compact prerequisites closed.
Fresh pilot pipeline has frozen gate hashes and started baseline under
`r63-pilot-retry1-baseline-server/session`; then `r63-pilot-retry1-defer1-*`.
Resource sampler writes `r63-pilot-retry1-resource-samples.jsonl` on the host
artifact mount, project-container process RSS/CPU and NPU0 aggregate only.
Never reuse premature `r63-pilot-baseline-*` / `r63-pilot-defer1-*` outputs.

After pipeline PASS review exposure-only size plan, freeze new QD3/HD3/FD3 and
statistical preregistration, then formal capture. New evaluator/causal export
code is being prepared; evaluator is NOT_RUN and requires further tests before
formal use. Neither old QD2 nor old HD2 are substituted for new formal corpora.

## Artifact roots and identities

Container `kt-r6-deferred-ascend`, NPU0, CPU0–15 production overlap.
Actual A3 base remains parent `e3084d309d6f39b8a9f7185d36ac27e4b77526d7`,
child `eaa58ccea4ee372b8b207dd16f398eafe3da37ff` plus documented source overlays.
Every server root includes `provenance.json`, raw captures, append-only index,
readiness report, guard command/PID/result. Local commits are not claimed as
unmodified A3 checkout SHAs.

Tools: `/workspace/results/round6_3-prequalification/tools`.
Legacy audited tools still used from `round6_2-prequalification/tools`.
Admin staging: `/home/admin/r63-stage-MjjDFs` (not qualification evidence).
New pilot token input SHA:
`0e309656587683068e515cd34024a360c6c8ca36924e3ee8a6a64d5f5b42e6ef`.
Long engineering smoke input SHA:
`bdeb64ec7cc853503cde7a274308e3c1a0c8be39353bab46a5162efc32461fa1`.

QD3 = RUNNING; candidate / runtime / HD3 / quality / batch2/4 = NOT_RUN.
Round6.3 = IN_PROGRESS, not qualified. Round6.2 remains blocked.
No new READY tag, push or main merge in this execution segment.

Current-thread heartbeat `round6-3` is ACTIVE (10-minute interval), created
after the natural QD3 phase completed. It resumes gated work from this file,
stays quiet on unchanged healthy state, and never duplicates an active run.
Pause it when the approved scope completes or a terminal in-scope blocker is
established. This is continuation scheduling, not evidence of gate completion.
