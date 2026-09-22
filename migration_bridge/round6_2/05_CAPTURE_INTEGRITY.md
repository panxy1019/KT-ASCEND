# Round 6.2 capture integrity — preliminary finding

Status: `A3_VERIFIED` for the smoke scope below. QD2/HD2 outputs are not
established by this document.

## 2026-09-15 smoke results

| Mode | ON requests | Complete forwards audited | OFF/reference parity |
| --- | ---: | ---: | --- |
| defer0 | 20 | 40 | 20/20 token and full-logit byte exact |
| defer1 | 20 | 40 | 20/20 token and full-logit byte exact |

Each request produced one target prefill and one serving lookahead. Every
lookahead was tied to that request's returned token and next absolute
position. Only the target prefill supplied the evaluated logit vector.
Sequences are contiguous, no duplicate/uncommitted capture exists, and target
requests have exactly one matching full-input record. The defer1 smoke has
four actual Deferred routes; defer0 has zero. Five separately frozen prompts
were repeated four times; these are not QD2/HD2 qualification data.

Evidence: `evidence/smoke/defer{0,1}-parity.json` and
`evidence/smoke/defer{0,1}-forward-audit.json`, plus full ON request records.
Reference mode disables full transaction/route capture but retains one final
raw-logit observer. This necessary common measurement is not a claim that
there is literally no observer in OFF mode. Uninstrumented production
determinism remains a separate retained runtime gate.

The first A3 attempt exposed direct `model.forward` calls bypassing module
hooks. Its failed directory is preserved; see `TELEMETRY_SMOKE_INCIDENTS.md`.
The corrected opt-in wrapper instruments the actual instance forward.
Local suite: 29 tests passed, including direct-forward and reference paths.

Defer1 preliminary smoke transaction provenance is empty, with explicit run
directories and source-overlay deployment recorded separately. Do not reuse
those records as formal qualification provenance. Defer0 smoke includes
actual checkout/source-overlay metadata. Qualification requires matching
nonempty run metadata in every captured target record.

## Failure mechanism

`TensorDumper._dump_hook` flushes its dictionary when the inner `model`
returns. DeepSeekV2ForCausalLM then invokes `logits_processor`. Consequently,
the legacy file can contain current model input metadata and previous-forward
logits. The last logits remain buffered until another forward. This is a
capture association defect, not evidence of an inference arithmetic defect.

The same ordering exists in official SGLang snapshot
`541ddc37cbc92c60dc748db5ff1a2aad0b069a80`:
[source](https://github.com/kvcache-ai/sglang/blob/541ddc37cbc92c60dc748db5ff1a2aad0b069a80/python/sglang/srt/debug_utils/tensor_dump_forward_hook.py).
No upstream rebase or merge was performed.

## Minimal opt-in instrumentation

`KT_R62_COMPLETE_FORWARD_DUMP=1` suppresses the inner flush, snapshots input
metadata at outer forward entry and flushes after the outer logits result.
`capture.next_token_logits` is explicitly bound to that forward. A stale
dictionary or non-logits outer output fails fast. Default behavior is unchanged.
This hook is diagnostic-only and may synchronize transfers; it does not
qualify uninstrumented production determinism or performance.

Dependency-free tests execute the actual dumper class with mock tensors:
legacy one-pass lag, two complete aligned calls, cloned input metadata,
stale-state rejection, and unexpected-output rejection. This does NOT establish
NPU stream or tensor-lifetime safety; A3 smoke remains mandatory.

## Historical evidence boundary

Do not reinterpret V1 rejection as PASS. Do not assert that all old sweeps are
invalid: their extraction may compensate for the lag. Each extractor must be
audited. In particular, selecting a dump because its input token equals a
response token is not a sufficient identity proof. The Round6.1 interpretation
of an "auxiliary scoring pass" is unproven and is superseded by this concrete
hook-order finding for new capture design.

## Required A3 smoke before qualification

Use only non-qualification prompts. Match full input/history hashes, outer
forward IDs, returned greedy token and full-vocabulary logits; require the
returned token to be a true maximum while preserving deterministic tie policy.
Check two consecutive distinct requests plus clean restart, trace completeness,
deferred-only weights, prefix/current exposure and pending drain. Missing
associations must fail, never be inferred from pass filenames alone.
