# P1 Same-Layer Multi-Expert

Placement: Layer 17 CPU experts `{6,8,25,36}`, physical NPU expert count 60.

## Controlled real-weight contract

The real Layer 17 fixture covered:

- C0: zero CPU-owned routes;
- C1: one CPU-owned route;
- C2: two CPU-owned routes in one token;
- C3: CPU routes in different TopK positions;
- C4: four CPU-owned routes and a different multi-token combination.

KTEP routed output was byte-identical to the explicit sum of CPU and NPU
contributions. The CPU-not-hit row was byte-identical to all-NPU. Applying the
routed scaling and shared expert exactly once also produced zero error. The
logical-to-local mapping was contiguous int32 and CPU entries remained `-1`
only at the internal mapping boundary; NPU routing received sanitized IDs and
zero weights.

Sequential and overlapped execution produced identical hashes:

```text
routed 940ee97d2f4dceb8f0bb34498882a21ef6d84c1ee87de65b5b09b457c5bc4d2f
final  1ca4322ca4dcf4188dd171eb079949cf875d35f3bec81ba58e8d8b8efdc386a7
```

The 1,000-forward lifecycle run produced exactly one routed hash and no crash,
deadlock, traceback, stale result or mapping change.

## Full-model blocker

The held-out P1 matrix completed all 45 requests with finite logprobs and stable
prefixes, but only 30/45 had exact all-NPU token IDs. Maximum observed
`|delta logprob|` was `2.4378519617021084`, above the `0.50` hard-block line.

The same Hybrid request is deterministic and every shorter generation equals
the corresponding 64-token prefix, so this is not evidence of the Round 3
stream/buffer race. Validation alone exercised all four CPU experts for 1,233
CPU routes (`E6=456`, `E8=58`, `E25=464`, `E36=255`), so the result also cannot
be accepted by claiming the selected experts were not used.

```text
MULTI_EXPERT_SINGLE_LAYER = BLOCKED
classification = TOKEN_DIVERGENCE / LOGITS
```
