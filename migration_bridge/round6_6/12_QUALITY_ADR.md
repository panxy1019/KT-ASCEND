# ADR: Round6.6 V1 does not pass its frozen quality screen

Decision: **REJECT V1 PROMOTION — FROZEN_QUALITY_SCREEN_FAILURE**.
No production algorithm patch,no sample/threshold/placement change.

| Endpoint | All-NPU | four-R32 | Gain / loss pairs | Invalid NPU / Hybrid | Truncated NPU / Hybrid |
|---|---:|---:|---:|---:|---:|
|GSM8K|3/32|2/32|0 /1|0 /0|18 /17|
|CMMLU|145/256|142/256|2 /5|8 /7|0 /0|

The pre-frozen criterion required zero observed net loss per endpoint and no
invalid-count increase. Both accuracy endpoints fail. The empirical paired
bootstrap95% intervals are[-9.375pp,0] and[-3.125pp,+0.78125pp]. Conservative
paired-marginal intervals are wider. These small samples do **not** establish a
population-level degradation of those exact magnitudes,or population equivalence.
Statistical uncertainty does not override the explicitly frozen screen rule.

Both288-request collections finished normally. All288 aligned inputs,stop lists,
budgets and raw-answer scores were independently re-audited. Runtime/asset/tool
hash drift is empty. The owned sequence correctly reported BLOCKED after scoring;
this is a quality gate failure,not a serving crash or resource failure.

GSM8K regression is one genuine answer change:80/8 was answered10 by All-NPU but
the Hybrid continuation later multiplied again and answered20. It is not a
truncation or parser artifact. CMMLU has7 correctness-discordant pairs:5 losses,
2 gains,all diverging at the first generated answer token. All must be included
in subsequent diagnosis,not only the losses.

The base-model zero-shot GSM8K protocol also has weak discrimination:only3
baseline successes and18truncations. The official few-shot protocol difference
in11 is a separate research hypothesis,not permission to rescore this experiment.

## Retained conclusions

- Three-way B0 and52-case tail safety PASS under original0.01.
- Same-path repetition/prefix/restart PASS.
- QD6/HD6 exact system/transfer/ownership controls PASS.
- HD6 old global bound has3 overflows; old contract remains failed.
- Free-generation common-history observer audit PASS.
- **Round6.6 V1 final qualification BLOCKED_QUALITY_SCREEN.**
- Round6.5 historical BLOCKED,Round7 NOT_STARTED,Deferred OFF remain unchanged.

## Autonomous next step, separate identity

Open Round6.6A quality-failure diagnosis. Freeze all8 correctness-discordant
items,recapture unchanged same-history first divergences,verify observer parity,
then inspect relevant router/operator/prefill boundaries. Do not patch arithmetic
without a violated invariant. If only expected backend perturbation remains,
record that limitation; any revised evaluation protocol needs a new independently
frozen study rather than turning this result into PASS.

Evidence: `evidence/quality_v1/`, including raw requests,paired_quality.json and
postprocess/all_quality_pairs.json. Raw paired audit SHA:
`d0b839b7cdaa162cea321242de330c06ed63a9ac9b9fb42f316c76714e8e01da`.
