# R63-PILOT preregistration — before pilot outputs

Purpose: measure natural stateful decode exposure and execution cost, not fit
numerical reserves. 32 new prompts, 64 targets each. All prompts retained,
including zero-exposure prompts. D1, batch1, existing expert placement and
defer budget. Baseline histories are generated under defer0 and frozen before
defer1 replay. Pilot outputs cannot enter QD3/HD3/FD3.

Pilot starts only after larger 20-request/64-position forcing/observer parity
and full index/transaction audits pass. Prompt/token duplication against frozen
historical corpora is checked before inference; exact deduplication does not by
itself establish statistical independence or eliminate semantic template reuse.

## Exposure and proposed formal size selection

For each of 32 prompt blocks, X is the number of 64 target positions whose
causal history (prefill, previous decode, current decode) includes actual CPU
Deferred work. Thus 0 <= X <= 64. Report current route events separately and
the count of prompts having any exposure. Synthetic forcing controls excluded.

Before observing pilot output, fix this conservative bounded-variable rule:

```
planning_alpha = 0.05 / 3
L = max(0, mean(X) - 64 * sqrt(log(1/planning_alpha) / (2*32)))
for N in [128, 256, 512]:
    conservative_total = N*L - 64*sqrt(N*log(1/planning_alpha)/2)
    select first N with conservative_total >= 500
```

This combines a pilot mean lower bound and two fresh-sample deviation bounds
(one each for QD3 and HD3), using three equal .05/3 union-error budgets under independent, bounded,
exchangeable prompt-block assumptions. It is a planning calculation, not a
guarantee for authored prompts nor a replacement for the actual >=500 gate.
If no N qualifies, do not append prompts until the gate passes; record a new
workload-design hypothesis/ADR and separate data rather than fit the holdout.

Once selected, freeze full QD3 and HD3 independently before their outputs.
Numerical tail analysis remains 95% coverage, alpha .05/3 and 5% tail budget;
recompute and preregister the corresponding order rank for chosen N. The pilot
size rule neither changes these numerical rules nor supplies numerical reserves.

## Cost estimate

Report forward target/lookahead counts, prefill rows versus decode rows, HTTP
elapsed time, final audit time and capture bytes. P50/P95 request costs provide
an estimate for the fixed full run, not a promised speedup. Model startup and
artifact hashing are recorded separately. No worker-count, NUMA or NPU changes.
