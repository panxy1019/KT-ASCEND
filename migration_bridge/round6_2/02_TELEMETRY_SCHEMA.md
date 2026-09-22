# Round6.2 telemetry identity contract

The required executable field contract is `tools/validate_position.py`.
Status: adapter implemented; A3 smoke correlation/parity verified within the
scope of `05_CAPTURE_INTEGRITY.md`. Hash the executable with the eventual
candidate and capture manifests; changes before capture must be versioned.

One row represents a full-vocabulary logits vector (102400 values) for a known
request/history/forward/target position, in either defer0 or defer1 mode.
Token index is the generated-history target index; route token_position is
the absolute input position within that request, including original prompt.

Prefix excludes the current input row. Exposure includes every actual Deferred
CPU route throughout the request, not only the target row or total CPU routes.
Each event must carry request and token identity, actual D1 producer/consumer,
logical CPU expert ID, zero-based original top-k slot, pre-normalization route
score and actual normalized execution weight. Neither quantity may be inferred
from all-CPU weight aggregates. Stage norms and amplification may be supplied
as explicit additional diagnostic fields; unavailable values must be labeled.

Repeated prefix records may be stored separately and referenced by hash in raw
artifacts, but must be fully resolved before validation. Missing metadata never
defaults to zero exposure. Trace truncation and undrained pending work fail.

The validator checks counts, identities and frozen D1 ownership, not correctness
of the collector itself. The smoke separately checks absolute position
alignment, normalization semantics, trace count and pending drain.
The two modes' request_input_hash and baseline_history_sha256 must match in the
comparison adapter. This check is not replaced by identical response tokens.
