# CMMLU serial product acceptance

Current state: NOT_YET_QUALIFIED. Fresh R64 is required before comparisons.

Frozen manifest SHA256:
`bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04`.

Exact256CMMLU rows are retained from the Round6.6B manifest with existing
five-shot dev examples, prompt strings, token IDs, labels and four candidate
letter IDs. No dataset download, resampling or quality-based expert selection.
Scoring uses the conditional log probability of A/B/C/D at one answer position,
with fixed A/B/C/D order for equal scores. Temperature0, batch1, max1token,
same base model/tokenizer/runtime. This is the project's frozen letter-
likelihood evaluator, not a claim of byte-identical official benchmark tooling.

Product gate: SYSTEM PASS,256valid results, observed accuracy loss <=5pp
relative to freshly collected R64. Integer resolution permits at most12fewer
correct answers. A reference174 requires>=162; actual fresh score controls.

Per-item raw response/log probabilities/latency and correctness are preserved.
The report includes improved/regressed correctness pairs and20,000 paired
bootstrap replicates(seed67). This observed finite-sample product policy is
distinct from a population non-inferiority test. Prior numerical/GSM8K results
remain historical evidence outside this newly approved CMMLU-only product gate.
