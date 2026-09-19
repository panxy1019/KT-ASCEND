# Dynamic CMMLU v2 independent audit

Audit date: 2026-09-19 17:41 UTC. Host: authorized 110.129.0.22.

The frozen auditor was hash-checked against candidate.json before executing
its CMMLU-only audit. No held-out release or performance approval was written.

## Result

- 256/256 frozen ordered unique item IDs, exact input IDs and complete request
  payloads, HTTP200 and matching response request IDs verified.
- Frozen first-position A/B/C/D finite-logprob scorer independently replayed:
  224/256 correct, invalid=0, accuracy87.5%.
- Q64 engineering reference222/256; observed delta +0.78125pp. The frozen
  observed -5pp product screen passes (minimum210/256).
- Q32 Static225/256; Dynamic differs by -1 item (-0.390625pp). This is not
  evidence of a change in model intelligence or statistical superiority.
- 217/256 requests recorded actual layer updates:84.765625% trigger rate.
- Resource samples contain zero owned swap, OOM and OOM-kill; server log has
  neither Traceback nor the transposed-NZ storage_offset warning.
- This parallel diagnostic run is NOT formal performance evidence.

## Frozen identity

- Candidate commit:9a183f309e1db794859646686431d9a1fbeb8819
- Candidate SHA256:69892f31dcbb450105a4adf153622c7bf6f5823292bfb8038ac9ae60160e55b9
- CMMLU manifest SHA256:365f557c0746db1e02ddee4ec66c7afb5e72eeb17e8dd8e0a47d1ed2b7af1b9d

Raw files remain in persistent host directory
`/home/admin/kt-round8-qwen35/qwen-dynamic-cmmlu-v2/` on .22:

| Artifact | SHA256 |
|---|---|
| summary.json | d75499ff11c9dce51a23b9b8b7fdc8239b977bc5ed1bc6122648f9958a31ef79 |
| raw.jsonl | c31a9861f9efc31e64c772dbe7a4a50dfa610af1987af8380e9d06acde7f7e60 |
| resources.jsonl | 0a898519f7ebcc472b11b46b5f99dfefadfeb0fd8df36702ad6bc39079c07615 |

Do not delete these files. A second recoverable archive and combined stage
release remain pending; local desktop has only approximately70MiB free, so no
large raw copy was attempted here. The remote collection state intentionally
remains COLLECTED_PENDING_INDEPENDENT_AUDIT; this document records the separate
audit without rewriting original collection evidence.

## Remaining predecessor

Heldout-v3 was healthy101/128 at this check. Its full generation/consumption
audit is still required. No claim of complete Dynamic qualification, speedup,
Round8 completion or Round9 model readiness follows from this quality result.
