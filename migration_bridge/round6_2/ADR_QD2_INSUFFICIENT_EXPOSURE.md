# ADR — Stop at QD2 exposure sufficiency

Date:2026-09-15. Decision: **STOP_INSUFFICIENT_EXPOSURE**.
Evidence: A3_VERIFIED route-only gate; LOCAL_VERIFIED independent recount.
This is not a numerical rejection of Deferred or a qualified V2 contract.

## Requirement and observation

CONTRACT_V2_SPEC.md, committed before model outputs, requires at least500
positions with actual Deferred exposure in the current row or request prefix.
Insufficient exposure is explicitly STOP, not permission to add favorable prompts.

Fresh QD2 defer1 replay completed8192 positions and16384 complete forwards.
Built-in full audit and independent matrix/transaction audits PASS. Original
startup-health contaminated run remains INVALID, neither filtered nor used as
qualification evidence. A documented fresh full retry closed that capture issue.

The unchanged evaluator measured343 exposed positions against minimum500,
returned INSUFFICIENT_EXPOSURE, and exited1 BEFORE loading cross-mode logits.
Its evidence explicitly states `numerical_metrics_read: false`.
No fitted reserve, candidate.json or HD2 inference was produced.

## Independent route-only recount

Manifest raw SHA256:
`637f7d730cccdb037bd395d1fc0489598ef9eb31e186f6cb9fff83719d9e5e54`.

| Measure | Observed |
| --- | ---: |
| Evaluated positions |8192|
| Exposed positions, union |343|
| Required |500|
| Current-row exposed positions |70|
| Prefix-exposed positions |292|
| Both current and prefix |19|
| Prompts with exposure |62/128|
| Current Deferred route events |70|
| Prefix Deferred route events |370|
| Exposure label inconsistencies |0|

Exposed positions by family:English89, Chinese169, code41, JSON35, math9.
Route-event records:E16=293, E50=137, E41=10, E38=0. These440 route records
are not440 distinct exposed positions. Replayed prefixes are not independent
sample units. Ordinary CPU routes, extra lookaheads or repeated identical
requests cannot be used to inflate the343 union count.

Independent helper reads route metadata only, with no torch or tensor-logit
access. Its first diagnostic output used a wrong optional expert-field name
and reported UNSPECIFIED for that breakdown;343 was unchanged.
`independent-count-v2.json` uses required `expert_logical_id` and supersedes
only that breakdown. Both diagnostic files and the original A3 gate are retained.

## Decision

- QD2_CAPTURE_INTEGRITY = A3_VERIFIED_READY, fresh retry only.
- QD2_EXPOSURE_SUFFICIENCY = FAIL,343<500.
- NUMERICAL_RESERVE_DERIVATION / CANDIDATE_FREEZE = NOT_RUN.
- D1_RUNTIME_REQUALIFICATION / HD2 = NOT_RUN.
- QUALITY / FREE_GENERATION / PERFORMANCE = NOT_RUN.
- ROUND6_2_DEFERRED_CONTRACT_V2 = NOT_QUALIFIED.
- ROUND6_2 = BLOCKED_INSUFFICIENT_EXPOSURE.

No acceptance relaxation, promotion tag, merge or production code change.
Earlier verified rounds retain their historical status.

## Reopen boundary

Do not repeat the same validated dataset to manufacture exposure. No corpus
additions, expert reselection, layer expansion, token-budget changes or lowering500
within this frozen qualification attempt. A separately approved future
sampling/exposure-design investigation must preregister a new experiment and
provenance, not retroactively approve this one. HD2 remains unobserved and must
not be used to design the replacement. This ADR does not authorize expansion.

## Evidence

- `evidence/qd2-defer1-retry1/`:original manifests and complete audits.
- `evidence/qd2-exposure-gate/exposure_gate.json`:original A3 failure.
- `evidence/qd2-exposure-gate/evaluator-result.json`:exact command and exit.
- `evidence/qd2-exposure-gate/independent-count-v2.json`:independent recount.
- `QD2_CAPTURE_INCIDENT.md`:separate immutable failed-run incident.
