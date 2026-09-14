# Round 6 Deferred Matched-History Numerical Protocol

State: `PREDECLARED_BEFORE_QD_OUTPUT`. No QD or HD model logits had been
captured when this rule and the associated tools were committed.

## Comparison and independence

For the D1 placement, generate a 64-token greedy history on defer=0 for each
frozen QD prompt. Capture full logits from defer=0 and defer=1 under that same
history, one position per request. Repeat independently for HD. The two sweep
manifests must contain the same corpus SHA, baseline-history SHA, prompt IDs,
token indices and baseline tokens. Free-running output after a divergence is
never used to derive numerical bounds.

The candidate set per position is the union of defer=0 top-32 and defer=1
top-32. Let `a` be the defer=0 top-1 token. For each other candidate `i`,
record the absolute change in `z_a - z_i`, as well as full-logit max/mean
absolute error, relative L2, cosine and candidate-logit error.

## Route association

The opt-in Deferred JSONL trace must be captured in a shared run directory.
The sweep reads only records appended during its own sequential request. For
target-token route counts and route weights it uses the **last row** of the
last `submit` record per CPU-enabled layer, not the aggregate prefill row
count. The trace must show final pending drain after each request. If target
row fields are absent or the trace cannot be associated with the request,
the sweep fails rather than treating missing telemetry as zero CPU hits.

Layer, actual deferred-route count, CPU-hit count, prompt family, position
bucket and route-weight bucket are diagnostic strata. The same frozen D1
placement is used for QD and HD. No threshold or placement may be tuned on HD.

## Predeclared QD rule

Sensitivity is reported for QD pairwise p99, p99.5, p99.9 and max, each at
factors 1.0, 1.25 and 1.5. The candidate selected for HD is **1.25 × the QD
maximum absolute pairwise margin distortion**; the 25% reserve is declared
before QD output is observed. Full-logit max-absolute and relative-L2 bounds
use the same `1.25 × QD maximum` rule. The candidate is rejected already on
QD if it has any overflow, stable flip, ambiguity escape, non-finite value or
an empty stable region.

At HD, the frozen candidate must have zero pairwise and full-logit envelope
overflow, 100% stable top-1 exactness, 100% ambiguity membership, finite
logits and position-aligned route telemetry. HD cannot be refitted; any
revision of the mechanism or bound requires a new held-out corpus.
Ambiguity-set size is diagnostic only. Expert-level relative-L2 safety
`<=1e-2`, system semantics, same-path determinism and downstream quality are
separate hard gates, not waived by this numerical contract.

## Tool chain

```text
run_teacher_forced_sweep.py --deferred-trace-file <shared JSONL>
compare_deferred_matched_logits.py
qualify_deferred_pairwise_contract.py derive  # QD only
qualify_deferred_pairwise_contract.py validate --role HD  # frozen candidate
```

The Round 4A pairwise `B_pair=2.1875, K=32` remains frozen for defer=0
retained regression. It is **not** silently reused as the defer=1 envelope.
