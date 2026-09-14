# Round 6 Deferred Risk Register

Evidence state: `CODE_INSPECTED`

| ID | Risk | Severity | First gate | Required evidence |
|---|---|---:|---|---|
| R6-R01 | Sparse block terminal silently defers with no consumer | Critical | unit/topology | per-layer defer rule + fail-fast test |
| R6-R02 | Empty/non-CPU deferred set creates false pending | High | unit/mechanism | `A3_VERIFIED`: zero false/missed pending after `0fbbc572b` |
| R6-R03 | Deferred contribution dropped or merged twice | Critical | D1 mechanism | explicit partial-sum reference |
| R6-R04 | Shared two-slot buffer reused before consume | Critical | D1 1000-forward | slot generation trace, one hash |
| R6-R05 | Class-global pending leaks across requests/models | Critical | sequential requests | terminal zero + clean reset/drain |
| R6-R06 | Exception/shutdown leaves pending worker | High | fault injection | surfaced exception, graceful drain |
| R6-R07 | NPU host drain races deferred enqueue | Critical | A3 mechanism | submit/start/end ordering trace |
| R6-R08 | Raw ACL transfer shortcut reintroduced | Critical | code + behavior | framework copy and transfer regression |
| R6-R09 | Prefill batch-union selection hides coverage | Medium | route profiler | requested vs actual deferred counts |
| R6-R10 | Deferred numerical amplification grows by layer | High | QD/D2/D3 | per-layer matched-history distribution |
| R6-R11 | Held-out stable token flip or envelope overflow | Critical | HD | 100% stable exact, zero overflow |
| R6-R12 | Quality regression despite numerical pass | Critical | quality | paired vectors/bootstrap, invalid=0 |
| R6-R13 | Correct feature has no performance value | Medium | A/B | alternating cycles and confidence interval |
| R6-R14 | Existing defer-zero baseline regresses | Critical | A3 B0 | Round 5 compact/frozen contract |
| R6-R15 | Coverage is achieved by post-validation reselection | High | freeze | disjoint S6/QD/HD and immutable hashes |

Round 6 does not relax any risk by reducing CPU hits, changing held-out inputs,
using sequential control as production, or widening a contract after HD.
