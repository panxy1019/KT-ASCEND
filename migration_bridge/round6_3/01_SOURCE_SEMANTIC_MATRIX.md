# Round6.3 source mechanism and operand audit

## Frozen semantics (CODE_INSPECTED)

Local selector and previously inspected official snapshot
`0dce4c9b097710951d1b3e232f50b973ff0a5505` use the union of row-wise top5
protected expert IDs over the entire forward. This round does not change it.
CPU input/output transfers, BF16 boundaries, FP32 partial merge, L7 producer
and L8 terminal policy remain unchanged.

## Real captured operand audit (A3_VERIFIED)

Source: successful Round6.2 defer1 retry, 8192 target forwards / 373440 rows.
No model-logit comparison or numerical reserve fitting was performed.
All selector input weights matched the captured CPU-copy weights exactly.
All partitions and terminal immediate routes passed. No top5 boundary ties
occurred, so all 8192 union reconstructions are strict, not tie-order estimates.

| Route diagnostic | Count |
| --- | ---: |
| L7 CPU routes in repeated prefixes | 282299 |
| Unique row-low CPU candidates | 34871 |
| Row-low CPU candidates protected by other rows | 34431 |
| Actually deferred CPU routes | 440 |

98.738% of these row-local candidates were protected by other rows. This is
fixed-operands causal evidence for cross-row suppression, not a prediction of
the new decode exposure rate. Decode arithmetic and routing must be measured.

Evidence: `evidence/selector-audit.json`; tool `tools/audit_selector_operands.py`.
The first deployment attempt could not write a root-owned old tools directory;
its guard exited before reading a capture. Retained original guard:
`/workspace/results/r63-selector-audit-guard`. Fresh retry used the new Round6.3
tools directory; no old tools or input artifacts were overwritten.

## Adapter boundary

SGLang `ModelRunner.sample` returns next IDs to the existing overlap scheduler.
The default-OFF test hook runs after ordinary greedy sampling, before this
return. It checks actual input IDs, absolute positions, sequence length and
per-RID monotonic steps, then returns a frozen token through the normal scheduler
path. It never assigns KV buffers or changes logits, routing or expert selection.

Appending a capture index replaces repeated filesystem scans in new tooling.
Full tensor capture remains enabled for initial parity. Compact capture and
batch>1 are not yet qualified. The adapter explicitly rejects batch>1.

Local protocol tests: 7 passed. A3 scheduler/forcing parity: pending.
