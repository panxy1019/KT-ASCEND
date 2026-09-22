# HD6 one-shot review and quality entry

Evidence state: target-executed Ascend 910B3, not a new run on historical A3.
Production arithmetic unchanged. Candidate was frozen before HD6 at SHA
`43a0bd582a4b126feb7ed851b85e1e94c1de98131700547eead4615d3e21967a`.

## One-shot results

12 independent heldout prompts ×64 matched-history positions =768 positions.
Both captured paths and complete-start/end capture audits finished; owned-process
guards exited zero and reaped their descendants.

| Metric | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| Full-logit max absolute |0.375|1.0|2.0|7.25|
| Full-logit mean absolute |0.0573502|0.164640|0.358671|1.369234|
| Full-logit relative L2 |0.012936|0.0395801|0.0734774|0.185876|
| Candidate pairwise distortion |0.1875|0.5|1.29125|2.65625|

765/768 teacher-forced top1 choices agree. The historical B_pair2.1875 diagnostic
has **3 overflowing positions**, no historical stable flip and no ambiguity-set
escape. This is **not** a pass of the old Round6.5 numerical contract. The old
Round6.5 failure remains blocked. No new bound was fitted from these outputs.

Observed system controls:3120 selected-layer forwards, exact D2H/H2D/FP32 merge,
128/128 CPU expert instances exercised,12682 CPU route events,52 exact CPU-not-hit
rows. All26 physical expert structures match the frozen ownership profile.
Router-set changes occur at526/768 positions somewhere in the model; paths use
the same token history but their hidden states need not agree.

## Tail investigation, separately identified after HD6

All three old-bound overflow positions and the largest full-logit error position
were reviewed, without rerunning or replacing HD6. The protocol in07 selected
**every naturally routed CPU expert** at those positions:52 expert/input cases.

All52 passed original local0.01 safety and10-repeat determinism. Maximum rel-L2:

- LLAMAFILE vs independent FP32 CPU:0.004549721297493019.
- NPU vs independent FP32 CPU:0.004549762290540459.
- LLAMAFILE vs NPU:0.00038648697038840555.

This does not establish full-model equality. It provides no evidence for changing
LLAMAFILE arithmetic to rescue an observed failure. Together with unchanged
runtime bytes, transfer/merge checks and router traces it supports proceeding to
the task-quality evidence required by the pre-frozen diagnostic-only model layer.
Larger heldout tails remain an explicit limitation, not a new accepted global max.

## Decision: allow evidence collection, not final promotion

`quality_execution_approval.json` SHA:
`3f58c84aba86b1f3325272a183ab0b904e0ff387919ba4e0180fe35260632f6e`.

The approval binds candidate,HD outputs,tail review,91 runtime files,model/tokenizer
assets,profiles,parsers and execution tools. It authorizes serial observer-OFF HD
free generation followed by All-NPU and four-R32 on the same frozen32GSM8K+
256CMMLU set. It does not authorize changing samples,thresholds or production
semantics. Free-generation common-history observer parity is checked first.

The quality rule remains zero observed net accuracy loss per endpoint and no
invalid-count increase, with paired uncertainty and baseline-floor review. This
small screen cannot establish population noninferiority. No quality PASS is
claimed until actual results and limitations have been reviewed.

## Evidence

- `evidence/hd6_review_v1/hd6_logits.json` SHA
  `36af0eeef0ab9117f17b2d4b8fdaea0648fc253a2b1c40dce49366033738c903`.
- `hd6_routers.json` SHA
  `902ec37ac56287b47aba9dd1602dbe1e5c376d3e93d6b844e9f38e1252567832`.
- `hd6_transfers.json` SHA
  `2dc605e7797e28b8da20f2ba10e1496e2cb5bed480b410653f5023b9e49a6691`.
- `tail_operators.json`, `heldout_guard/`, `tail_guard/` preserve bounded followup.

Round7,all-MoE promotion,R16/R8 and Deferred remain outside this execution entry.
