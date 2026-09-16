# Round6.6A — quality-failure diagnosis (not a new passing contract)

Base:Round6.6 V1 failed-quality freeze `ad8795496`.
Branch:`codex/round6-6a-quality-failure-diagnosis`.

The V1 screen remains failed:3→2/32 GSM8K,145→142/256 CMMLU.
Its small-sample uncertainty does not waive the frozen zero-observed-loss rule.

1. Freeze **all8** correctness-discordant items (6losses,2gains),unchanged input
   bytes and original answer tokens. No result-driven subset deletion.
2. Fresh All-NPU and four-R32 capture with12 diagnostic tokens and one audited
   lookahead. For CMMLU,the original quality request stops after the answer;
   only its first-divergence boundary is a faithful quality replay. Extra tokens
   are diagnostic,not new quality evidence. GSM8K first divergence is index11.
3. Verify fresh sampler choices against original observer-OFF quality outputs
   through each first-divergence boundary. Force the same NPU history in Hybrid.
4. Audit all captured transfers/merges/ownership;measure candidate margins,
   first router changes and hidden amplification. No bound fitting.
5. Inspect actual implicated prefill operands,including complete mixed-MoE
   replay against production,not just isolated single-expert safety. Preserve
   all observed shape/route combinations and both gain/loss controls.
6. Only a violated semantic invariant justifies a production correction.
   Otherwise report expected arithmetic sensitivity and separately evaluate the
   base-model benchmark protocol hypothesis using a new study identity if needed.

No masks,workers,GGUF,model,old contract or V1 quality answers are changed.
No Deferred,Dynamic,all-MoE promotion or guessed NPU reduction emulation.
This diagnosis does not itself qualify quality or create new heldout evidence.
