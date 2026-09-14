# P1 Before-Fix Reproduction

The original 45-case failure was reproduced in the A3 disposable container
using the frozen all-NPU JSON and the frozen P1 placement. The service was
single-request, greedy, BF16, TP1 and bound to CPU0-15/NPU0.

## Result

| Measurement | Frozen P1 |
|---|---:|
| completed requests | 45/45 |
| finite | yes |
| prefix deterministic | yes |
| exact token requests | 30/45 |
| maximum absolute logprob delta | 2.4378519617021084 |

The earliest inspected failure was `v_en_01`. All-NPU generated `[185,185]`
for its first two tokens, while the original Hybrid generated `[185,549]`.
The exact Layer 17 input, TopK IDs and TopK weights were captured for prefill
and the first two decode passes.

System-correctness checks excluded the earlier stream/buffer class of bugs:

- repeated Hybrid requests were deterministic;
- shorter generations were prefixes of the 64-token generation;
- the same Layer 17 input and routes reached both backends before divergence;
- no crash, deadlock, stale result or non-finite value occurred;
- validation exercised all four selected CPU experts.

Therefore the entry blocker was classified as numerical/logits divergence,
not lifecycle or routing corruption.
