# Round6.4 same-graph engineering oracle

Evidence state: **A3_VERIFIED**. Gate: **ENGINEERING_ORACLE_PASS**.

This compares production cross-layer Deferred with the independent explicit
private-pending implementation of the **same approximate graph**. It does not
compare Deferred OFF versus ON, and is not a downstream quality qualification.

Four preselected engineering prompts (English, Chinese, math, JSON) use the
64-token histories from the successful D1 runtime run. Each mode captures
260 physical forwards: 256 requested positions plus four lookahead forwards.
The explicit path exercises 22 private deferred forwards.

| Check | Result |
|---|---:|
| Common tensor checks | 15,600 / 15,600 byte-exact |
| Full-logit comparisons | 260 / 260 byte-exact |
| Mismatches | 0 |

Evidence: `evidence/r64-engineering-oracle-v2/comparison.json`, `report.json`,
`force.json`, and the per-mode capture audit reports. Raw captures remain on A3
under `/workspace/results/r63-r64-engineering-oracle-v2-{production,explicit}-server`.
The source report SHA256 is
`7359ea61c5aa041aba22540f341fb1ad8d069b12544bb6cf52f9ac9078640c1d`.
The report binds actual deployed source hashes, D1 placement, GGUF, and runtime
evidence; the local documentation commit is not misrepresented as deployed code.

Attempt v1 remains preserved as a tooling failure: its forced-history manifest
omitted the retained reader's required wire version. The correction adds the
wire version and validates against the real reader before starting serving.
Three local manifest tests pass. No production code changed and no numerical
acceptance condition was relaxed.

Next gate is the actual routed CPU-task lifecycle audit. Generation token count
is not a substitute for measuring CPU task exposure. Quality, performance, and
final Round6.4 approval remain pending.
