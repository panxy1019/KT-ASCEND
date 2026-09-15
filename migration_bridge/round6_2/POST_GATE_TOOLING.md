# Prepared downstream tooling — not downstream qualification results

These tools were prepared while QD2 defer0 was collecting. None authorizes
skipping QD2, candidate freeze, D1 runtime, or HD2. No HD2/quality/FD model
request was issued during this preparation.

## D1 runtime

`tools/run_runtime_determinism.py` requires a frozen candidate and original S6
12-prompt manifest. The main campaign uses ten 64-token repeats and 8/16/32
prefixes per prompt (156 requests). A fresh-process campaign checks 12 outputs
against the first campaign. Raw API responses are retained; ID/count/finite
logprob checks and the first mismatch fail immediately. This is only the
determinism component; exactly-once, terminal drain, no-hit, explicit replay
and instrumentation parity remain separate required runtime evidence.

## Quality

`tools/run_quality_checked.py` invokes the original Round4A4 evaluator, whose
source SHA is pinned to
`3d1884744e828668c60704f836ce4265ecf137d1e1da8770f98a69118c3de926`.
No prompts, conditional choice scoring, tie policy, generation protocol or
manifest changes. It requires an HD2 PASS report and saves every raw response.
Missing/nonfinite probabilities and invalid token IDs fail closed.

`tools/compare_quality_checked.py` implements the preregistered paired
20,000-replicate bootstrap, seed6202, 95% interpolated percentile interval.
A wholly negative interval is regression; otherwise the claim is only
NO_DETECTED_SIGNIFICANT_REGRESSION, not equivalence/noninferiority. Counts,
reference labels, missing/duplicate pairs and invalid outputs are checked.

## Actual-forward free generation

`tools/run_free_generation_captured.py` keeps logits from the actual generating
prefill/decode forwards and validates input IDs, absolute positions, sequence
length, request ID, source provenance, transaction hashes, actual serving
maxima and producer/consumer transitions. It does not substitute full-prefix
prefill logits for original decode logits. All 64 output positions and the
serving lookahead must be present. The established overlap implementation
expects 65 complete forwards for each 64-token request; a mismatch is an
instrumentation incident, not an implicit numerical PASS.

`tools/compare_free_generation_captured.py` compares only the first divergence
under identical preceding history, using frozen V2 decision rules. Deferred
exposure includes earlier cached generation steps plus the current prefill or
decode step. No threshold refitting occurs.

## Tests and evidence limits

Local suite: 56 tests, 49 passed and 7 torch-dependent tests skipped because
local torch is absent. This is not an A3 model qualification result.

The actual-forward parser was CPU-replayed on already saved A3 smoke captures:
40/40 forwards per mode PASS; no new model requests. Evidence:
`evidence/smoke/fd-parser-defer0.json` and `fd-parser-defer1.json`.
The first two parser attempts exposed undeployed Python helper dependencies
(`verify_reserved_inputs`, then `freeze_corpora`). Both failed during import,
before any capture parsing or model call. Helpers were deployed; subsequent
CPU replay passed. The ongoing QD2 process/source was not changed or restarted.

The real FD campaign, quality A/B and D1 runtime gates are still NOT_RUN at
this tooling commit. Future summaries must replace those states only with
the corresponding new A3 evidence.
