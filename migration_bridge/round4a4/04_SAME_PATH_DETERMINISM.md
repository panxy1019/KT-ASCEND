# Same-Path Determinism

Evidence state: **A3_VERIFIED**.

Disposable container: `kt-r4a4-pairwise`, CPU 0-15, NPU0 only.

Four representative Q2 prompts cover English, Chinese, math and structured
JSON. Within one process, each prompt was generated ten times for 64 tokens.
All repeated token sequences were exact. Independent 8/16/32-token requests
were exact prefixes of the 64-token sequence for all four prompts.

The server was then terminated and cleanly reloaded with the same frozen
runtime and seed. All four new 64-token sequences were byte-for-byte identical
to their pre-restart references.

| Gate | Result |
|---|---:|
| prompts | 4 |
| repetitions per prompt | 10 |
| same-process exact | 4/4 |
| prefix exact | 4/4 |
| clean-restart exact | 4/4 |
| all finite | PASS |

Evidence:

- `evidence/wp0-same-process.json`
- `evidence/wp0-clean-restart.json`
- `evidence/wp0-restart-compare.json`

`SAME_PATH_DETERMINISM = A3_VERIFIED_READY`.
