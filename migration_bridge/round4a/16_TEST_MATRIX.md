# Round 4A Test Matrix

| Domain | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| Frozen base/tag | PASS | - | - | - |
| Placement deterministic/validated | PASS | PASS | PASS | PASS |
| Real expert identity | PASS | PASS | NOT RUN | NOT RUN |
| Physical NPU count | 63 | 60 | validated only | validated only |
| 0/1/2/multi CPU routes | - | PASS | NOT RUN | NOT RUN |
| Explicit CPU+NPU sum | - | PASS | NOT RUN | NOT RUN |
| Sequential vs overlap | - | PASS | NOT RUN | NOT RUN |
| 1,000-forward lifecycle | - | PASS | NOT RUN | NOT RUN |
| Full-model requests | 15/15 | 45/45 | NOT RUN | NOT RUN |
| Exact token IDs | PASS | FAIL (30/45) | NOT RUN | NOT RUN |
| Prefix determinism | PASS | PASS | NOT RUN | NOT RUN |
| Logprob budget | PASS | FAIL (2.43785) | NOT RUN | NOT RUN |
| CPU hit coverage | anchor hit | 4/4, 1,233 | NOT RUN | NOT RUN |

Ordered execution stopped at P1 as required.
