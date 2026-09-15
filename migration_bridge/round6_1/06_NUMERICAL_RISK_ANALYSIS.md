# Round 6.1 descriptive numerical-risk analysis

Evidence state: `ANALYTIC_DERIVED` from frozen A3 captures.

This analysis is mechanism research only. It does not derive a new threshold,
does not validate a contract, and does not restore held-out status to HD.
Round 6 V1 remains rejected.

## Signed definitions

For baseline anchor `a` and candidate `i`:

```text
m0 = z0[a] - z0[i]
m1 = z1[a] - z1[i]
d  = m1 - m0
e  = abs(d)
adverse contraction = max(0, -d)
beneficial expansion = max(0, d)
```

## Frozen corpus summary

| Corpus | Positions | Candidate pairs | abs p95 | abs p99 | abs max | adverse max | beneficial max | top-1 changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QD | 1,024 | 31,807 | 0.0625 | 0.125 | 0.375 | 0.375 | 0.25 | 1 |
| HD | 1,024 | 31,810 | 0 | 0.125 | 0.75 | 0.625 | 0.75 | 3 |

The existing global sanity metrics remain finite. QD full-logit max absolute
error is 0.8125 and relative-L2 max is 0.0444545; HD values are 0.8125 and
0.0319061. These are descriptive and are not new acceptance limits.

## Historical blocker sign

For `hd_en_02:43`, candidate 16520:

```text
m0 = 10.75
m1 = 11.5
d  = +0.75
e  = 0.75
adverse contraction = 0
beneficial expansion = 0.75
top-1 changed = false
```

The frozen V1 contract correctly rejected `e=0.75 > 0.46875`; Round 6.1 does
not retroactively change that outcome. Mechanistically, however, this specific
outlier increased the winning anchor's margin and did not threaten the greedy
decision.

## Stratification and correlations

The machine-readable reports include strata by baseline-margin bucket,
target-row Deferred hit, Deferred route count, prompt family, token-position
bucket, top-1 status, and target-row CPU-route-weight proxy.

Pearson correlations are weak:

| Correlation | QD | HD |
| --- | ---: | ---: |
| abs distortion vs baseline margin | -0.0731 | -0.0516 |
| adverse contraction vs baseline margin | -0.0580 | -0.0514 |
| abs distortion vs target-row CPU-weight-mass proxy | +0.0265 | -0.0217 |
| adverse contraction vs target-row CPU-weight-mass proxy | -0.00531 | -0.0248 |

The frozen QD/HD rows retain all target-row CPU route weights, but do not retain
which subset of those weights was actually deferred. The weight correlations
are therefore explicitly labeled proxies; no Deferred-only weight conclusion
is claimed.

The target outlier itself has no CPU/Deferred hit on row 43. Stagewise evidence
shows it is downstream of earlier prefix Deferred work. Therefore target-row
hit count or weight mass alone cannot characterize accumulated Deferred risk.

## Implications for a possible Round 6.2

The data supports studying, but does not approve:

- signed adverse contraction separately from beneficial expansion;
- decision-critical candidate margins rather than undirected global maxima;
- prefix-level Deferred exposure rather than target-row-only telemetry;
- a new, precommitted QD2/HD2 protocol with an entirely new held-out corpus.

No candidate rule is selected here. Opening Round 6.2 requires explicit
authorization.

## Evidence

- Absolute/global descriptive report: `evidence/numerical_risk/result.json`
- Signed QD report: `evidence/numerical_risk/qd_signed.json`
- Signed HD report: `evidence/numerical_risk/hd_signed.json`

```text
ROUND6_1_NUMERICAL_RISK_ANALYSIS = COMPLETE_DESCRIPTIVE_ONLY
NEW_NUMERICAL_CONTRACT           = NOT_CREATED
```
