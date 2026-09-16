# Round6.6B — independent base-model quality protocol study

This is a single, bounded follow-up, not a repair of V1's failed result.
Entry requires Round6.6A mixed-partial replay to complete without an unexplained
implementation difference. V1 and Round6.5 remain BLOCKED. Production arithmetic,
four-R32 masks, checkpoint, tokenizer, workers and generation budget do not change.

## Independent hypothesis

The frozen checkpoint is a pretrained base model. V1's zero-example GSM8K prompt
produced only3/32 reference successes and18 truncations. The official
[DeepSeek-V2 evaluation](https://arxiv.org/html/2405.04434v5) uses8-shot GSM8K and
5-shot CMMLU with likelihood-based multiple-choice scoring. This motivates a
protocol study; it does not prove that V1's losses were a benchmark defect.

## Protocol, fixed before new inference

- GSM8K32 fresh test IDs, excluding all V1 and older screen/dev evaluation IDs.
  Eight worked examples from the pinned train split. Choose by SHA256 rank with
  seed `660602`, among examples of at most128 tokenizer tokens, solely to keep
  the unchanged2048 context capacity. Answers remain512-token greedy, original
  numeric parser, truncation is wrong. No outcome-based exemplar selection.
- CMMLU256 fresh test IDs, one per67 subjects plus189 globally hash-ranked.
  Exactly five existing dev examples per subject in source order. Score A/B/C/D
  single-token conditional log-probabilities at the answer boundary. This is
  letter-likelihood scoring, not a claim of exact official harness reproduction.
- Both modes use identical frozen input IDs, stop lists and evaluator. BOS as
  before. Fail preflight if any input plus budget exceeds2048; do not silently
  delete long test questions or shorten their prompts.
- Freeze source revisions, exact bytes, exclusion IDs, exemplar IDs, tool hashes
  and final input SHA before either model mode. Run All-NPU then four-R32.
- Keep the previous conservative engineering screen: zero observed net accuracy
  loss on either benchmark and no invalid-count increase. Report paired counts,
  bootstrap intervals, reference floor and truncations; do not claim population
  noninferiority. This rule is not changed to2pp after seeing V1.
- No second reseed/re-prompt attempt after these outputs. A failure closes this
  protocol hypothesis as insufficient. A pass only qualifies this new limited
  quality screen; it does not erase V1, qualify all-MoE, or prove universal quality.

The new data are independent evaluation inputs for this hypothesis, not a fresh
heldout for numerical-envelope fitting. No B_pair is estimated or changed.
No new production backend, Dynamic, Deferred or all-MoE promotion is authorized
by this protocol study alone.
