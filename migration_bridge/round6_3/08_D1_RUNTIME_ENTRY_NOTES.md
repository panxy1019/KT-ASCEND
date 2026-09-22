# Candidate-bound D1 runtime — preparation only, NOT_RUN

Entry requires successful QD3 derivation, candidate JSON committed locally,
full candidate SHA and exact A3 deployed source/artifact hashes recorded.
QD3 capture completion alone is insufficient. HD3 remains prohibited until all
runtime subgates close; a determinism-only report cannot stand for the whole gate.

## Reuse and minimal adapter

`round6_2/tools/run_runtime_determinism.py` already implements frozen S6:
12 prompts x10 repeated64-token requests plus8/16/32 prefixes, then fresh restart.
Its current candidate version check is explicitly round6.2-v2-candidate.
Do not falsify a Round6.3 candidate version to reuse it. Port a small tool-only
adapter accepting round6.3-v2-candidate while reusing its response validator.
Retain raw HTTP responses, exact token/logprob ID matching and finite logprobs.
Use true observer OFF for this same-process/restart gate, production overlap,
same route recorder/configuration as the qualified engineering controls.

## Additional required subgates

- Full transaction evidence: routing once, terminal layer8 never defers,
  producer7/consumer8 order, pending fully drained, input/ID/weight buffers intact.
- Whole-request CPU-not-hit: frozen historical input `[1000]`; inspect EVERY
  physical forward, including any lookahead. Target-row-only zero is not enough.
  If actual whole-request no-hit does not hold, report it rather than changing
  the definition or selecting a convenient QD3/HD3 input after numerical results.
- Explicit replay: frozen input from `round6_2/fixtures/runtime_controls.json`,
  corresponding to historical hd_en_02:43. Use its recorded input bytes, not a
  new HD3 example. Existing source uses KT_R61_EXPLICIT_REPLAY test-only mode.
- >=1000 representative CPU-expert forwards/lifecycle: retain actual counts and
  hashes, RSS/HBM samples, exceptions and pending checks. Do not infer CPU work
  merely from total generated tokens, and do not count padding as real work.
  Observer-backed transaction/lifecycle evidence is separate from observer-OFF
  same-path determinism.

Legacy analyses are references, not newly verified results:
`round6_1/tools/analyze_runtime_controls.py` embeds a54-row legacy target and
stage-file naming; it cannot be applied blindly to the new complete-forward
schema. `verify_full_request_cpu_not_hit.py` correctly distinguishes target-row
misses from whole-request zero routes, but expects the old preflight/session
format. Preserve the semantics, adapt only the input schema, and unit-test
negative cases before any new PASS report.

## Runtime closure

Same-process, fresh restart, exact-once/terminal/pending, whole-request no-hit,
explicit replay, and lifecycle reports must all identify the same candidate and
configuration. Only their combined A3 evidence can authorize HD3 one-shot.
No runtime tests have been executed for the Round6.3 candidate at this point.
