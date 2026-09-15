# Round6.3 — blocked at frozen QD3 candidate-pair gate

## Decision

`ROUND6_3_A = BLOCKED_QD3_STABLE_CANDIDATE_PAIR_FLIP`.
No candidate contract was emitted. Runtime requalification,HD3,FD3,quality,
ADR approval and batch2/4 qualification remain NOT_RUN. Previous Round6.2
rejection is unchanged. This is not a READY freeze and no READY tag is created.

## Completed engineering work

Stateful teacher-forcing and complete-forward indexing were implemented and
validated on A3. True observer OFF/reference/full controls and full/compact
byte comparisons passed. The independent pilot measured1814/2048 exposed
positions and selected128 prompts under a preregistered sample-size rule.
QD3/HD3/FD3 inputs and derivation protocol were committed before inference
at8191a8edc. All three QD3 modes completed8320 physical forwards each, with
8192 targets and128 excluded lookahead,unknown0,incomplete0. Self-history
comparison passed216565 retained tensor comparisons. Deferred exposure7390
exceeded the unchanged500 minimum.

## Frozen numerical result

QD3 REJECTED on2 positions /3 non-anchor candidate pairs. Stable top1 exact
4647/4647,all three statistical tail checks passed; those results cannot
override the frozen all-candidate ordering gate. Exact pairs, hashes and
reserves are in09_QD3_REJECTION_INVESTIGATION.md. The original33MB report
and all captures remain preserved on A3, with a compact hashed local extract.

## Post-failure investigation

Original-tensor verification reloaded202 captures through the two failing
targets and reproduced all reported margins. Recorded input/history/hash,
terminal and pending transaction checks passed.

Fresh diagnostics used the same two baseline histories,64 tokens each, not
newly selected easy inputs. Both production and private-buffer explicit oracle
completed130 forwards (128 targets+2 lookahead), each with complete audits.
11 oracle private pending forwards passed producer/consumer identity checks.

| A3 comparison | Common tensor byte equality | Full-logit forward equality |
|---|---:|---:|
| Original QD3 vs fresh production |3510/3510|130/130|
| Fresh production vs explicit oracle |7800/7800|130/130|

No differing common tensors were observed. Diagnostic-only unpaired keys are
listed, not silently claimed validated. Private oracle removes the shared
incremental-buffer execution mechanism while preserving next-layer injection.
These two-history results support INTENDED_DEFERRED_APPROXIMATION and do not
identify a production defect to repair. They do not prove universal absence
of races or qualify downstream quality.

## Contract scope issue (ANALYTIC_DERIVED)

The registered reserve measures only contraction of anchor-to-candidate
margins. Arbitrary i/j candidate-margin contraction equals the difference of
their signed anchor distortions. Its magnitude is not bounded by the negative
component alone. Consequently the reserve is not a proof of stability for
all candidate pairs. The explicit frozen all-pair rule nevertheless applies:
the current run remains correctly REJECTED, including positive-to-zero ties.

Removing the all-pair gate, changing its bound, doubling the reserve or
recalibrating another metric would change the frozen numerical contract.
None was done. No upstream runtime patch found in the inspected source
establishes a remedy for this contract-level incompatibility.

## Boundary and next decision

Autonomous in-scope failure investigation is complete for these two cases.
Further progress toward qualification requires an independently justified
contract/protocol revision or an independently evidenced production defect.
Do not claim that a particular revised rule passes without a fresh freeze
and untouched held-out validation. HD3 has not been used to fit any rule.
No approved implementation repair remains demonstrated by current evidence.
The current-thread heartbeat round6-3 was PAUSED to avoid polling a terminal
rejection. No user/business process was stopped. Only owned diagnostic model
groups were cleaned up by their guards after completion.

## Provenance

Local branch: codex/round6-3-decode-exposure-qualification.
Base:5e708dc63d07afe4d79d26f0bdbb97d95ad3aaba.
A3 parent:e3084d309d6f39b8a9f7185d36ac27e4b77526d7 plus hashed diagnostic overlays.
A3 child:eaa58ccea4ee372b8b207dd16f398eafe3da37ff plus hashed diagnostic overlays.
Model:604d5664dddd88a0433dbae533b7fe9472482de0.
Container:kt-r6-deferred-ascend,NPU0,CPU0–15,TP1,batch1,production overlap.
Runtime/kernel/weights/placement unchanged during QD3 and rejected-case replay.
No host-library or business-container mutation. Inherited llama.cpp changes
preserved. Local commits are not yet pushed; no main merge or READY tag.

Evidence: evidence/qd3-v1-rejected-pairs.json,
evidence/qd3-v1-rejected-original-verification.json,
evidence/qd3-v1-explicit-replay-comparison.json,
evidence/qd3-v1-replay-session.json,evidence/qd3-v1-oracle-audit.json.
