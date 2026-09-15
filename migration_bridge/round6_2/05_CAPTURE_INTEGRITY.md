# Round 6.2 capture integrity — preliminary finding

Status: `LOCAL_VERIFIED`; A3 dry-run `NOT_RUN`. QD2/HD2 outputs `NOT_RUN`.

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
