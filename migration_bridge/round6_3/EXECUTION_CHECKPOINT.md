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

Long full baseline and compact forced defer0 both PASS: 20x64 targets,
1300 final audited forwards each, 34045 retained tensor comparisons byte exact.
Raw capture shrank from1380241180 to592963662 bytes. See04 report.

Now running `/workspace/results/r63-long-force1-full-control`, with model root
`/workspace/results/r63-long-force1-full-server`, session
`/workspace/results/r63-long-force1-full-session`. The controller owns this
server, shuts it down after capture, and performs final index/physical audit.
Do not restart a healthy run.

After PASS run compact defer1 against the same long-force.json history and
compare observers. All are engineering
20-request prequalification, not QD3. Only then run independent pilot and use
the preregistered size rule; freeze new QD3/HD3/FD3 before formal results.

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

QD3 / candidate / runtime / HD3 / quality / batch2/4 = NOT_RUN.
Round6.3 = IN_PROGRESS, not qualified. Round6.2 remains blocked.
No new READY tag, push or main merge in this execution segment.
