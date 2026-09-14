# Git Commits

- parent base: `ba508e4f920e99cd8cf1c0127d9aa6e5e0ac2559`
- SGLang base/final: `8a4bb4c850860377d9880b8c0fa570ca2d0d19e1`
- parent branch: `feature/kt-round4a2-cpu-gemm-numerical-closure`
- child branch: `feature/kt-ep-round4a2-cpu-gemm-numerical-closure`

Final commit groups:

1. `test(round4a2): add CPU GEMM backend numerical probes`
2. `docs(round4a2): record exhausted backend investigation`

There is no production backend commit because the candidate failed P1. The
exact parent final is the branch HEAD reported by `git rev-parse HEAD` after
the documentation commit; it is also reported in the final handoff.
