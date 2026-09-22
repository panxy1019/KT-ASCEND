# F3 bounded prefill telemetry persistence

Date: 2026-09-21

## Scope

The layerwise prefill runtime now sends one complete logical-route summary to
its existing event callback when a prompt finishes. The callback used by the
real binding appends this event to the layerwise JSONL evidence stream. No
per-token event stream, new background writer, or second statistics collector
is introduced.

This is host-framework implementation evidence. It does not grant a device,
full-model, quality, performance, or deployment qualification.

## Completion event

The terminal event retains `status: PREFILL_COMPLETE` and adds the following
bounded fields:

- `schema`: event schema version 1.
- `request_id`: the exact request identity used by the collector and transport.
- `generation`: the placement generation held across all prompt chunks.
- `token_range`: `[0, token_count]`, the complete half-open prompt span.
- `layers`, `experts`, and `topk`: the route geometry.
- `completed_phases`: exactly 1.
- `logical_expert_counts`: integer route counts for every logical expert in
  every layer.
- `logical_expert_weight_mass`: finite, nonnegative route-weight mass for every
  logical expert in every layer.

Both matrices have fixed `layers × experts` size. Their storage is independent
of prompt length, and every count row must conserve `token_count × topk` routes.
The event builder copies the matrices so an event callback cannot mutate the
statistics returned to the prefill-end F4 barrier.

The return value from `PrefillRuntime.after()` remains the existing route-stat
document with `counts`, `weight_mass`, `pending`, `tokens`, and geometry fields,
plus `request_id` and `generation`. F4 therefore consumes the same statistics
shape as before.

## Lifecycle and failure behavior

Intermediate chunks continue to emit only `LAYERWISE_BATCH_COMPLETE`; they do
not publish a complete aggregate. A normal final chunk finishes the transport,
closes the collector phase, validates the complete aggregate, clears the
runtime request, and sends the single completion event.

An abort discards the pending collector phase and emits `PREFILL_ABORT`. It
never constructs or sends `PREFILL_COMPLETE`, including cancellation between
chunks. Partial statistics cannot escape as completed evidence.

The event callback is part of the fail-closed runtime transaction. If completion
event construction or persistence raises, the runtime enters
`BROKEN_RELOAD_REQUIRED`, clears its collector and request ownership, and does
not admit further work in that process. The underlying transport has already
finished at this point, so cleanup does not attempt a second abort transaction.

## Local verification

The focused tests cover:

- multi-chunk request identity, placement generation, and complete token span;
- exact logical expert counts and weight mass;
- conservation and malformed/partial-stat rejection;
- no complete event on in-flight or between-chunk abort;
- completion-event sink failure poisoning the runtime and clearing ownership;
- compatibility of the returned statistics with the prefill-end barrier.

Real scheduler persistence still depends on the existing JSONL event callback
installed by `prefill_binding.py`. Its device and full-model evidence must be
generated under the separately admitted fresh overlay and gates.
