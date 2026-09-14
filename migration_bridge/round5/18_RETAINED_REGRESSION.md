# Round 5 — Final Retained Regression

Evidence state: `A3_VERIFIED_READY`

The final focused suite was rerun after P3 on the exact production source
qualified by the Round 5 tag:

```text
parent: be5b48621db32fb894402e3e886060dea263cdc9
SGLang: 412219b4b05a888808f0b88d2963a8b6164a0fda
```

## Final rerun

| Plane | Test | Result |
|---|---|---:|
| R2A | ARM detector, GGUF load, routed/mapping correctness, 1000-forward lifecycle, two-layer identity | 18 passed |
| R2B | framework-managed Ascend BF16/FP32/int64 transfer and fail-fast contracts | 5 passed |
| R2C/SGLang | KTEP ownership, int32 mapping, placeholder/zero-weight routes, active accelerator stream | 4 passed |

All final reruns used `/workspace/venv-aligned/bin/python`; unexplained failures
were zero.

An initial final-run command accidentally invoked container system Python.
The kt-kernel groups failed at collection because that interpreter did not
contain `kt_kernel`; the SGLang test imported its fallback path for the same
reason.  No model request or production test ran in that attempt.  The logs
were retained and the complete matrix was restarted using the exact venv.

The broader old R2A suite also has three tests that inspect local private
diagnostic attributes (`_gguf_loader_path` and `cpu_runtime_diagnostics`).
Those attributes are deliberately absent from upstream.  They remain
`STALE_TEST_EXPECTATION`, not unexplained failures and not APIs restored to
production solely for test compatibility.

## Retained higher-level gates on the same source

| Gate | Result |
|---|---:|
| Round 3 real Layer17/Expert8 replay | PASS |
| P1 C0-C4 and 1000-forward | PASS |
| P1 determinism/pairwise/quality | PASS |
| P2 determinism/pairwise/coverage/quality | PASS |
| P3 full qualification and stability | PASS |

There was no production source change between these profile runs and the
final focused rerun.  Parent commits after the installed runtime wheel are
test-only; all imported production Python and compiled kt-kernel code match
the tagged source semantics.

```text
R5_RETAINED_REGRESSION = PASS
R5_UNEXPLAINED_FAILURES = 0
```
