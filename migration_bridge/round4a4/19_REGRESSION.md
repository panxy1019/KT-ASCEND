# Regression

Status: `PARTIAL_WITH_RECORDED_LEGACY_FAILURES`

## Current Round 4A.4 checks

- all Round 4A.4 Python tools compile with `py_compile`;
- all Round 4A.4 shell launchers pass `bash -n`;
- placement tooling completes successfully;
- current SGLang KT EP E2E plus the new numerical-dump test: 5 passed;
- the instrumentation is default-off and its child diff changes no arithmetic,
  routing, ownership, merge, or scheduling behavior;
- P1 controlled sequential/overlap and 1,000-forward lifecycle pass.

## Retained Round 2 suites on A3

| Suite | Result | Interpretation |
|---|---|---|
| Round 2A | 17 passed, 4 failed | all four failures are stale dtype expectations: retained tests require BF16 CPU partials while the preserved Round 4A.1 FP32-partial contract returns FP32; numerical E2E rel-L2 remained 0.003894 |
| Round 2B | 22 passed, 1 skipped, 1 failed | retained `test_d2h_cpuinfer_moe_h2d_pipeline` observed an invalid large output in the disposable-container run; recorded as an unresolved legacy regression failure |
| Round 2C | 44 passed | PASS |

The Round 2A failures must not be “fixed” by reverting FP32 Hybrid partial
accumulation. Their assertions need a separately reviewed update to the current
visible-boundary contract. The Round 2B pipeline failure is retained verbatim in
the evidence log and is not marked PASS.

The project remains blocked earlier by the independently reproduced P2
`SAME_PATH_NONDETERMINISM`; no regression result is used to override that gate.

Evidence logs:

- `evidence/regression-round2a.log`
- `evidence/regression-round2b.log`
- `evidence/regression-round2c.log`
- `evidence/regression-sglang-kt-ep.log`
- `evidence/regression-placement-tools.log`
