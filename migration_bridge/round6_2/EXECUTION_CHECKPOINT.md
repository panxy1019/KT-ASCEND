# Complete-plan execution checkpoint

Latest task source:
`KT_ASCEND_Round6_2_Complete_Execution_Plan_After_Capture_Fix.md`.
The complete plan was read before executing its new Phase1 audit.

| Phase | State | Evidence / remaining work |
| --- | --- | --- |
| 0 prequalification checkpoint | RECORDED | parent start47618943b, childa903c004d; inherited llama.cpp unchanged |
| 1 legacy capture audit | PASS | independent436/436 stage comparisons rechecked; B/C compensated logits proven with predecessor/hash/API margins |
| 2 telemetry / complete forward | SMOKE_A3_VERIFIED | direct-forward wrapper; transaction/route/provenance adapter; see capture-integrity measurement boundaries |
| 3 statistical method preregistration | COMMITTED_BEFORE_OUTPUT | CONTRACT_V2_SPEC, prompt128, rank127, three alpha=.05/3 families |
| 4 QD2/HD2 input freeze | COMMITTED_BEFORE_OUTPUT | 128 each; actual tokenizer IDs; joint canonical hashes; separate five-prompt smoke |
| 5 A3 capture smoke | PASS_IN_RECORDED_SCOPE | defer0/defer1 each 20 requests, 40 audited forwards; 20/20 full-logit byte-exact reference/ON comparisons |
| 6 QD2 | BASELINE_HISTORY_RUNNING | frozen 128x64 defer0 history, then same-history dual sweeps; no candidate reserve yet |
| 7 candidate threshold freeze | NOT_RUN | no fitted reserve |
| 8 retained runtime | NOT_RUN | old evidence is not new requalification |
| 9 HD2 | NOT_RUN | held-out outputs remain absent |
| 10–13 quality/free generation/performance/ADR | NOT_RUN | predecessor gates closed first |

The new plan requires telemetry before model testing. Input preparation and
method registration were completed offline while the adapter is unfinished;
no model gate was skipped. The corpus freeze does not itself prove statistical
exchangeability or actual Deferred exposure. Those limitations remain explicit.

A3 prequalification now includes four smoke services in the project's
disposable container, sequentially using NPU0 alongside authorized existing
workloads. Own services were stopped through their individually recorded
process-group supervisors. No other workload was terminated. No A3-ready tag,
candidate approval, main merge or promotion was created.

Current baseline history server: `r62-guard-qd2-history`, run root
`r62-qd2-history-server`; client supervisor: `r62-client-qd2-history`;
history output: `r62-qd2-history`. All are under `/workspace/results`.
Both supervisors have a predeclared 14400-second timeout. HTTP deadline is
600 seconds; no automatic retry/filtering. Healthy ongoing runs must not be
restarted. On error, stop only these explicitly owned groups.

Next: finish and hash QD2 baseline history; start full-transaction defer0 and
defer1 replay with identical frozen histories and nonempty matching run
provenance. Then derive reserves, enforce all QD2 hard gates and exposure
minimum before committing any candidate. D1 runtime requalification and HD2
remain downstream. Synthetic decision tests: 6/6 CPU-only on A3; local suite
29 passed plus those six skipped because local torch is absent.
