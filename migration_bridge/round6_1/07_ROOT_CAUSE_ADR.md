# ADR: Round 6.1 Deferred numerical root cause

Status: `ACCEPTED` for root-cause classification.

Decision date: 2026-09-15.

## Context

Round 6 V1 froze a pairwise contract at `B_pair=0.46875`, K=32, then correctly
rejected D1 on the independent HD corpus because `hd_en_02:43` had pairwise
distortion 0.75. The contract and rejection are immutable. Round 6.1 was opened
to determine whether that residual was an implementation defect, an intended
Deferred approximation, a semantic specification gap, or inconclusive.

## Considered decisions

### A. IMPLEMENTATION_DEFECT — rejected

This would require production Deferred to disagree with an independently
implemented explicit replay, or a routing, buffer, ordering, pending-state, or
CPU-not-hit invariant failure. None was observed.

### B. INTENDED_DEFERRED_APPROXIMATION — selected

Production B equals explicit replay C across 436/436 aligned stage/call tensors,
all three semantic deferred/incremental aliases, and all 102,400 final logits.
Both differ from no-defer A first at the exact tensor where the Deferred
schedule intentionally hides the selected L7 CPU contribution until L8.

### C. SEMANTIC_SPECIFICATION_GAP — rejected

Local and audited upstream semantics agree on the essential boundary: a
selected contribution produced for L is carried as pending CPU state and
merged at L+1. Upstream race fixes address buffer initialization/lifetime but do
not redefine this mathematical visibility rule.

### D. INCONCLUSIVE — rejected

The reproduction, stagewise localization, replay, runtime controls, and
whole-request CPU-not-hit control provide a complete forward attribution for
the frozen case.

## Decision

```text
ROUND6_1_ROOT_CAUSE = INTENDED_DEFERRED_APPROXIMATION
```

The primary first-divergence class is:

```text
DEFERRED_VISIBILITY_DIFFERENCE
```

The subsequent mechanism is:

```text
DOWNSTREAM_NONLINEAR_AMPLIFICATION
```

The decision is based on these independent facts:

1. the exact historical residual reproduced without environment drift;
2. A/B are exact through L7 routing and NPU work;
3. the first difference is the intentionally delayed L7 CPU contribution;
4. explicit replay without production pending machinery is exact to B;
5. live pinned buffers remain exact before/after sync;
6. completion precedes consumption and pending state clears once;
7. a whole-request CPU-not-hit case is exact between defer=0 and defer=1.

## Consequences

There is no Round 6.1 production fix. The diagnostic instrumentation remains
opt-in/test-only and cannot alter normal execution when its environment
variables are unset.

The result does not approve Deferred numerically. The old held-out rejection
still blocks D1:

```text
ROUND6_V1_CONTRACT = HISTORICAL_HELDOUT_REJECTED
ROUND6_D1          = BLOCKED_PENDING_NEW_INDEPENDENT_CONTRACT
ROUND6_D2          = NOT_RUN
ROUND6_D3          = NOT_RUN
```

A future Round 6.2 may research a signed, decision-critical contract only after
precommitting a new mechanism, QD2, and independent HD2. It may not raise the
old threshold, refit HD, or reinterpret Round 6 V1 as passing.

## Evidence map

| Question | Evidence |
| --- | --- |
| Historical reproduction | `02_HISTORICAL_CASE_REPRODUCTION.md` |
| First semantic difference | `02_STAGEWISE_LOCALIZATION.md` |
| B equals explicit C | `03_EXPLICIT_DEFERRED_REPLAY_ORACLE.md` |
| No observed lifetime defect | `04_RUNTIME_LIFETIME_CONTROLS.md` |
| No-CPU mechanism control | `05_WHOLE_REQUEST_CPU_NOT_HIT_CONTROL.md` |
| Descriptive risk only | `06_NUMERICAL_RISK_ANALYSIS.md` |
