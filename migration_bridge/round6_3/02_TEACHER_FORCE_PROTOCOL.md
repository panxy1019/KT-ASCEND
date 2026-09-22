# Stateful teacher forcing v1 — engineering prequalification

## Execution boundary

`ModelRunner.sample` invokes a default-OFF helper only when
`KT_R63_FORCE_MANIFEST` is set. Before arming, original sampling is unchanged.
After a non-generating `/model_info` readiness check and empty capture boundary,
the controller exclusively creates the arm file containing the exact manifest
SHA. The adapter loads and validates this SHA once, retains immutable process
state, and exclusively creates a new trace. Unknown requests after arming fail.

Batch1, TP1, greedy, Graph OFF, radix reuse OFF, no bias or penalties in request
protocol. Batch>1 fails explicitly; not yet a batch qualification implementation.

Each RID: one complete prefill, then one consumed frozen token per decode.
Sequence length and absolute position are checked against the original prompt
and monotonically advancing step. An extra production-overlap lookahead is
allowed exactly once and is excluded from target counts and exposure gates.
More forwards, duplicate prefill, missing steps, wrong IDs or consumed tokens
fail. Scheduler owns KV and advances it through its ordinary path; this helper
neither reads nor shares KV tensors across mode processes.

Natural logits are captured before sampling. The adapter verifies finite logits
and natural max selection, logs a full-logit byte hash, then replaces only the
returned token ID. The sampler logprob fields still describe natural sampling,
not an evaluated forced token; qualification must use captured raw logits, not
these HTTP output-logprob values. Capture and sampling records must join on
PID, sequence, RID, step and identical logit hash.

## Index and final integrity

Atomic complete `.pt` commits precede append-only index records. Client reads
the index incrementally. Full tensor observer remains in use at this stage.
Once this project's serving process is stopped, an independent final audit
reconciles every physical capture with index and result, rejects uncommitted or
extra files, and checks one PID / contiguous sequence. Startup and lookahead
are never silently removed.

## Engineering smoke, not statistical qualification

Four new engineering prompts, 8 target tokens each, first establish natural
defer0 baseline. A fresh process forces those histories. Full-logit bytes must
match. Four additional controls force token101 uniformly at all positions:
at least one non-greedy injection must occur and the next actual forward must
consume it. These artificial controls are not QD3 or pilot exposure evidence.

Larger 20-request/64-step observer parity and independent 32-prompt pilot remain
required before any formal QD3/HD3 freeze. A small smoke PASS does not close them.
