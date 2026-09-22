# Round 6 D1 clean-restart determinism

Evidence state: `A3_VERIFIED` for the D1 clean-restart gate only. This does not
close D1 numerical, downstream-quality, stability, or D2/D3 gates.

The production-overlap D1 service was stopped after the e308 same-process
matrix. A new serving process loaded the same e308 wheel, model, D1 placement,
GGUF, NPU0-only configuration, and `--kt-max-deferred-experts-per-token 1`.
It generated 64 greedy tokens for each of the 12 frozen S6 prompts. No
sequential-control override was used.

The independent process results were compared with the earlier 12-prompt,
10-repeat, prefix-checked e308 run using
`round4a5/tools/compare_determinism_runs.py`. The comparison required the
same corpus, prompt order, prompt input-token hashes, and exact output-token
lists. Result: **12/12 exact; `all_exact=true`**.

| Artifact | SHA256 (file bytes) |
| --- | --- |
| `evidence/d1-final/clean-restart-e308/before.json` | `171b717bfec144fec3eaa37acd7baf090ab19d5d83271fc1bcde7b23c9639edc` |
| `evidence/d1-final/clean-restart-e308/after.json` | `0b551ec45820dfe3b43a40041c6b32a5989085872dea36705e56a463c9e82102` |
| `evidence/d1-final/clean-restart-e308/compare.json` | `0cf4d7cf11fc8d2d80a52b98f5860331e5a911c6235ba36c53eece7e7c7d7fe5` |

The compare artifact's canonical payload SHA is
`822237e2316da9a44e2cd4c70ae69c050b1b2f3d852d1e35a639bb8fb97b879f`.
The A3 raw result directories are
`/workspace/results/d1-determinism-e308-a/` and
`/workspace/results/d1-clean-restart-e308-b/` in disposable container
`kt-r6-deferred-ascend`.

Next gate: defer=0 versus defer=1 matched-history QD numerical sweep. HD is
reserved until a candidate is derived and frozen from QD.
