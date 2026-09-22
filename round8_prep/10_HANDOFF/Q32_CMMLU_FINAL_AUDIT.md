# Q32 CMMLU final audit — 2026-09-19

Q32-C64 Static was evaluated with the unchanged frozen Qwen CMMLU manifest
`365f557c0746db1e02ddee4ec66c7afb5e72eeb17e8dd8e0a47d1ed2b7af1b9d`.
The evaluator used the Qwen tokenizer and chat template, thinking disabled,
greedy temperature 0, one generated token, the same A/B/C/D candidate token
logprob scoring and the same ordered 256 items as the Q64 engineering
reference.

The raw archive is `qwen-cmmlu-q32-v1.tar.gz` with SHA256
`a4ddedc07dd962821124bb4da717da60d8cb9712c9fa4ac017fae2d4142a6e2b`.
Its raw JSONL SHA256 is
`ced9027d9e685659e624dd7163269e7d097e315e2830b2d0974a5f2494119610`.

The independent audit verified 256 rows, 256 unique item IDs, 256 unique
request IDs, HTTP 200 for every response, exact frozen input IDs, candidate IDs,
stop sequences and sampling parameters, one output score position per item,
finite candidate scores, and replayability through the frozen scorer.

Result:

| profile | correct | count | invalid | accuracy | delta vs Q64 222/256 |
|---|---:|---:|---:|---:|---:|
| Q64-C64 reference | 222 | 256 | 0 | 86.71875% | 0pp |
| Q32-C64 Static | 225 | 256 | 0 | 87.890625% | +1.171875pp |

The frozen observed quality rule requires at least 210/256 correct and zero
invalid outputs. Q32 therefore passes the quality screen. This is a product
screen result under the frozen manifest; it does not establish token trajectory
equivalence, formal throughput, long-run memory stability, Dynamic qualification
or Deferred qualification.

Answer distribution for Q32 predictions was A=62, B=61, C=64, D=69.
