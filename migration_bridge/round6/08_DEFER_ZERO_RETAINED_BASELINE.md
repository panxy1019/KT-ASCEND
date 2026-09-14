# Round 6 defer=0 Retained Hybrid Baseline

Evidence state: `A3_VERIFIED` for the compact P1/P2/P3 controls below. This is
not a replacement for the historical Round 5 full qualification.

## Why this control precedes final D1

The Round 6 selector, pending-state, and SGLang topology changes must not
change the validated non-Deferred Hybrid path. All three retained placements
were exercised with `--kt-max-deferred-experts-per-token 0` on parent
`b6815196d8bfd531f810ba34372999dc77911b9c`, SGLang child
`eaa58ccea4ee372b8b207dd16f398eafe3da37ff`, and the final-state installed
wheel recorded in `06_LOCAL_TESTS_AND_BUILD.md`.

The P1/P2 GGUF came from the read-only Round 5 reference mount; the P3 GGUF
came from a newly added read-only Round 4A.5 closure mount. The Round 6
disposable container was recreated to add that mount. Source, venv and results
remained in persistent bind mounts; no host libraries, business containers or
NPU1–7 were modified.

## Protocol and results

Each profile used frozen S6 prompt IDs `s6_en_01`, `s6_zh_01`, `s6_math_01`,
`s6_json_01`: two repeated 32-token greedy generations plus independent
8/16/32-token prefix requests. All four prompts per profile had exact repeat
and prefix checks.

| Profile | Placement file SHA256 | Result | A3 result JSON SHA256 |
|---|---|---|---|
| P1_B0 | `c12d2954e0188ab9d7d1567085d342fe4b3ed5b769fdb976f743c2bbb50fc509` | 4/4 PASS | `d5dfb77b0c77887cc66981ad03b04c01fe5b898130b6c816dbf19d1fe273bbbb` |
| P2_B0 | `25636c437750c56dac46efb711ab9b0876b5ef4e2519edc13ef8a2c3abca67b3` | 4/4 PASS | `2cd5425f0304314f496d6ae7fd09a59c3cb6bbc00b65639843ca2b88ca175896` |
| P3_B0 | `e5544faa22f210e28b02648316a139ca737aca81d9c034d757b7fd59b2e0a55b` | 4/4 PASS | `ae34b6c2838890483214d387bf2e7ce1bac7fe29a12cb5efe0ab517e20eb2404` |

Each A3 artifact directory contains `result.json`, `run.log`, `server.log`,
`RUN_PASS`, and `SHA256SUMS.txt`:

```text
/home/admin/kt-artifacts/round6-deferred/b0-p1-final/
/home/admin/kt-artifacts/round6-deferred/b0-p2-final/
/home/admin/kt-artifacts/round6-deferred/b0-p3-final/
```

## Gate

```text
ROUND6_DEFER_ZERO_P1_COMPACT = A3_VERIFIED
ROUND6_DEFER_ZERO_P2_COMPACT = A3_VERIFIED
ROUND6_DEFER_ZERO_P3_COMPACT = A3_VERIFIED
```
