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
| 6 QD2 | DEFER0_COMPLETE_DEFER1_RUNNING | defer0 8192 positions + 16384 complete forwards pass both built-in and independent audits; defer1 collecting successfully; no cross-mode numerical decision yet |
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

Baseline history finished with client EXIT_ZERO and 128/128 rows. Raw manifest
is retained on A3 and exactly backed up as `evidence/qd2-history/manifest.json.gz`.
The history server has been stopped through its own supervisor.

Completed defer0 server supervisor: `r62-guard-qd2-defer0`, run root
`r62-qd2-defer0-server`; client supervisor: `r62-client-qd2-defer0`;
sweep output: `r62-qd2-defer0`. Client EXIT_ZERO, 8192 rows, 16384 complete
forwards. Independent matrix and transition audits PASS. The service was
stopped deliberately through guardian281573 / child group281578 after success.

Active server supervisor: `r62-guard-qd2-defer1`; run root
`r62-qd2-defer1-server`; active client `r62-client-qd2-defer1`;
sweep output `r62-qd2-defer1`. All are under `/workspace/results`.
Both supervisors have a predeclared 21600-second timeout. HTTP deadline is
600 seconds; no automatic retry/filtering. Healthy ongoing runs must not be
restarted. On error, stop only these explicitly owned groups.
Metadata: defer0 `round6_2-prequalification/qd2-defer0-provenance-v2.json`;
defer1 `round6_2-prequalification/qd2-defer1-provenance.json`.
The older metadata file without `-v2` was not used for model capture.

Next: finish defer1 replay with identical frozen
histories and nonempty matching run provenance. Enforce the route-only exposure
gate BEFORE reading numerical comparisons. Then derive reserves and enforce all QD2 hard gates and exposure
minimum before committing any candidate. D1 runtime requalification and HD2
remain downstream. Synthetic decision tests: 7/7 CPU-only on A3; local suite
29 passed plus those seven skipped because local torch is absent. Additional
A3 real CPU tensor tests reject all four injected capture integrity failures.

Additional whole-matrix gate before numerical evaluation:
`tools/audit_complete_matrix.py --root <sweep-root> --output <sweep-root>/complete_matrix_audit.json`.
It rechecks both target and lookahead independently (including finite logits,
sequence lengths and terminal transitions); original aggregate totals alone
are not used as a substitute for one pair per request. Also retain the
`audit_deferred_transitions.py` report required by the numerical evaluator.

Downstream tooling is prepared, not yet qualified; see POST_GATE_TOOLING.md.
The latest A3 CPU-only tool tests passed22/22. No HD2 model output exists.
