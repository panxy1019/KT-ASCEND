# D1 same-process and clean-restart runtime — A3 verified

Result: **RUNTIME_PASS**. Evidence root `evidence/r64-d1-runtime-v1`.

| Gate | Observed |
|---|---|
| Same process |12frozen S6 prompts×10×64 plus8/16/32 prefixes=156requests |
| Unique trajectory |1 perprompt; all repeatedtokens exact |
| Prefix |all36 shortrequests match own64token prefix |
| Clean restart |12/12×64 exact against firstprocess |
| Finiteness |all returned per-token logprobs finite; token/logprob IDs aligned |
| Observer |OFF, tensor directories remainempty |
| Execution |originalproductionoverlap, batch1,context512,TP1,NPU0 |
| Cleanup |bothownedserverguards plannedCANCELLED; no unrelatedprocess actions |

The same-process matrix generated8352tokens, restart768. These counts are not
substituted for CPU task/expert forward exposure. That requires a separate
instrumented lifecycle audit. This gate does not establish downstream quality
or speedup, nor compare the different OFF/ON computation graphs for equality.

Source/artifact binding recorded in `progress.json`:

- Approval file SHA `488739bfb34ac7faedbe20df60b124840a22a409059ceaf47b6b8bee98da2f8b`.
- Actual preflight provenance SHA `f5bcbda2a3b9e751c2d6a622af625ddbf0ab281d2ba88f3291339f9e7ea39f84`.
- Runtime runner SHA `0131526df0cf13dd564b3f29e309e0729979a5628e7ab25a2cec3fe77a1328e7`.
- Shared lifecycle runner SHA `0ccdd7c3132a8e6cac52c9ee8f1fb5f8d3126c0a264a75e1c108a5cc62ef849e`.
- Final progress SHA `658e51a68af7a37cee1d06934ec15d6eaad6e9ef5d0abfa134382bded40eea82`.

Every file listed in the actual preflight metadata was rehashed before startup;
the model_runner, placement and GGUF were additionally bound. Actual A3
checkout-plus-overlay identity remains explicit, not replaced by a local docs SHA.

## Subsequent oracle tooling failure, separate evidence

`r64-engineering-oracle-v1` failed before a completed request because the new
test orchestrator omitted `version=r63-forced-history-v1` from the retained
teacher-force wire manifest. Model-side HistoryLedger correctly raised
`R63_MANIFEST_VERSION`; the owned guard stopped only that experiment.

This is **TOOL_MANIFEST_SCHEMA**, not evidence of Deferred numerical or
same-path regression. Production code and the successful runtime evidence are
unchanged. Failure logs, command and source metadata are retained under
`evidence/r64-engineering-oracle-v1`.

The tool-only correction supplies the wire version and invokes the actual pure
HistoryLedger validator before model launch. Three local tests cover acceptance,
missingversion and wronghistorylength. New attempts use `r64-engineering-oracle-v2`;
the old failure is neither overwritten nor declared PASS. The wire schema name
does not reactivate the rejected Round6.3 numerical acceptance contract.
