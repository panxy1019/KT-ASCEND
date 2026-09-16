# Round6.6B final summary

## Outcome

`SCREEN_FAILED_NO_RESEED`.
Implementation/reference-lane investigation is complete;heavy-static promotion
is still blocked. No passing tag is created. Historical Round6.5 and V1 results
are unchanged. This is not a claim that the requested overall Round6 success
condition has been achieved.

## Evidence chain

1. Round6.6 three-way decomposition:608cases,128CPU expert instances,original0.01
   local safety gate passes;actual native-stage capture parity608/608 exact.
2. QD6/HD6 and independent exact controls complete. Old global B_pair is not
   repaired or enlarged. New model-level distributions remain diagnostic.
3. V1 quality screen failed:3→2/32 GSM8K and145→142/256 CMMLU.
4. Round6.6A freezes every V1 correctness discordance,including gains;all32
   full mixed-partial replays and10repeats/backend are exact to production.
5. One independent official-style base-model protocol study was preregistered,
   with new32+256test IDs and fixed train/dev examples. No second reseed.
6. New quality:14→13/32 GSM8K,174→173/256 CMMLU. Raw288-pair audit passes;
   frozen quality screen fails. Population-quality conclusion remains inconclusive.

See02_FINAL_ADR.md for paired intervals,real-answer failure review and the precise
limits of the no-new-implementation-defect conclusion.

## Provenance

- Evidence commit before final bookkeeping:
  `19ab602e8722fe25d4b1d859afa807b9c27f1703`.
- Branch:`codex/round6-6b-base-model-protocol`.
- Actual target parent:`e3084d309d6f39b8a9f7185d36ac27e4b77526d7`.
- Actual target child:`eaa58ccea4ee372b8b207dd16f398eafe3da37ff`,with frozen overlays.
- Model:`604d5664dddd88a0433dbae533b7fe9472482de0`.
- Tokenizer:`41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669`.
- New quality input:`885ccf76386dcfed2867142417300ac69c4401754360f92e6b6e4f179adffd97`.
- Raw paired audit:`51b0f7133a553e0e464b507b45e909689c460ffca9a9224e82d604409ab8dff6`.
- Full57,579,411,648-byte GGUF rehashed after the run:
  `20114dba7790a9b6be40a17ffb3881c516b81ea86b8f2fcfd3ea61d052e96fe2`,unchanged.
- Three deployed collector/controller/audit hashes match01_EXECUTION_FREEZE.md
  after completion. Runtime and model/tokenizer asset checks passed before/after.

## Safety and verification

Target110.129.0.22,Ascend910B3,NPU0 only;32workers,one pool,CPU0–31;
container`kt-r65-static-offload`. No host library or business-container mutation.
Both serving lifecycles COMPLETE;outer guardEXIT_ZERO,reaping complete.
Final`docker top` shows only the container's idle`sleep` process. Test/model
processes are gone;the container/artifacts are retained for audit,not deleted.

Local tests:Round6.6 16/16,Round6.6A 2/2,Round6.6B 5/5;diff whitespace check passes.
Pre-existing`third_party/llama.cpp` dirty state is preserved;do not claim a fully
clean user worktree. New work is committed locally;no GitHub push or main merge
was performed in this execution.

## Final gates

| Gate | State |
|---|---|
|Backend decomposition|PASS within measured coverage|
|System/exact-path evidence|PASS within documented coverage|
|Production arithmetic defect found|NO demonstrated new defect|
|V1 quality|FAILED|
|Independent B quality|FAILED|
|Population noninferiority|NOT_ESTABLISHED|
|Performance|REVIEW_REQUIRED,no speedup claim|
|All-MoE heavy qualification|NOT_RUN|
|Round7 Dynamic|NOT_STARTED|
|Heavy Deferred probe|NOT_RUN|
|Default Deferred|OFF|
|Overall heavy-static promotion|BLOCKED|

Repeating fresh quality samples until one passes is not authorized by this
preregistered study and would invalidate its evidential purpose. The remaining
problem is not resolved by inventing an arithmetic patch or declaring an
uncertain small-sample comparison statistically equivalent.
