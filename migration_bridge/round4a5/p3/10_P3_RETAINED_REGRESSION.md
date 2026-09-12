# P3 Retained Regression

Evidence state: `A3_VERIFIED_READY`

Fresh retained regressions ran after P3 qualification and after all serving
processes had been stopped:

| Suite | Result |
|---|---|
| Round 2A CPU expert plane | 21 passed |
| Round 2B Ascend runtime plane | 23 passed, 1 expected CUDA-only skip |
| Round 2C synthetic Hybrid MoE | 44 passed |
| SGLang KT EP Ascend E2E | 4 passed |
| Round 3 real Layer 17 | PASS; mixed rel-L2 0.0032707 |
| Round 4A.4 P1 compact C0-C4 | PASS; max rel-L2 0.0035398 |
| P1 lifecycle | 1,000 overlap forwards, one output hash |
| P2 compact production overlap | 4 prompts × 3 repeats × 64, repeats and prefixes exact |

The Round 2B CUDA-only skip is expected on Ascend. No unexplained failures
remain. The retained FP32 partial/output-buffer test correction was used; no
stale BF16 test allocation was interpreted as a runtime failure.

All source logs and compact JSON outputs are stored under `evidence/g9-*` and
inventoried by `evidence/SHA256SUMS`.

`P3_RETAINED_REGRESSION = PASS`.
