# P3 Same-Process Determinism

Evidence state: `A3_VERIFIED_READY`

The first P3 hard gate ran on the default production-overlap path:

```text
9 prompts × 10 repeated 64-token greedy generations
+ one 8/16/32-token prefix request per prompt
= 117 requests
```

Results:

| Requirement | Result |
|---|---|
| one 64-token trajectory per prompt | 9/9 PASS |
| repeats exact | `all_repeat_exact=true` |
| 8/16/32 prefixes exact against 64 | `all_prefix_exact=true` |
| request failures | 0 |
| traceback/crash/deadlock | 0 |
| non-finite model output observed | 0 |

Artifact: `evidence/g1-same-process-validation9.json`; file SHA
`909abbdb85837207eb8ad24cfe7d7f557d88fe12d2dbfe1ad566f89430afc85f`;
canonical internal SHA
`72f802d2e6386826bc912b0e341a5f176041661ac14cf41ca8b7fe71790c67a8`.

`P3_SAME_PROCESS_DETERMINISM = A3_VERIFIED_READY`.
