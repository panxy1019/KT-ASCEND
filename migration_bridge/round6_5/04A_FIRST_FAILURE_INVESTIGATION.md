# First failure — four-layer R32 numerical envelope

Date2026-09-16. State:BOUNDED_INVESTIGATION_COMPLETE_PROTOCOL_STOP.
Not accepted/not repaired;no production defect has been established.

Qualification root:`/workspace/results/r65-scale-matrix-v2/four-R32`.
Same-process/prefix/restart28requests PASS. Complete8request capture integrity
PASS:520forwards=512target+8lookahead. Pairwise evaluation uses actual serving
sampler token before teacher forcing,top32union,B_pair2.1875 unchanged.

First and only failed position:

- rid:`r65-h2-h2_en_01`,zero-based token54;
- baseline top1=Hybrid top1=82;
- stable exact259/259,ambiguity membership253/253;
- maxpairwise9.59375,27candidate pairs exceed2.1875;
- full-logit maxabs20.703125,relativeL2.5690376 at that position;
- adjacent positions51–53,55–57 have relativeL2~.008–.018;
- both captured inputIDs[6],position[64],seq_len[65].

This is **STATIC_NUMERICAL_SCALE_BLOCKER**,not a successful token-only result.
Top1agreement does not override the frozen envelope. Single-layer success does
not authorize all-layer continuation. Scale queue FAILED and capacity queue
FAILED_PREDECESSOR;no fourR16/R8,all-layer,frequency or Deferred promotion.

Next bounded investigations,with all production semantics unchanged:

1. Independent fresh-process full8request replay,identical history/order.
2. Raw-logit comparison and same-path byte equality vs original fourR32.
3. Debug-only matched-history single-failing-prompt R64/fourR32 layer traces.
4. Locate first router-set/output amplification;verify physical mapping,
   controlled same-input expert safety before attributing to floating point.

Do not prelabel reduction order,buffer corruption or CPU shortage as cause.
No threshold fitting,expert reselection,worker-count change or corpus removal.

## Independent replay and layer localization

`r65-four-R32-failure-replay-v1`:all512matched-history logits exactly reproduce
the first run,including byte hashes. Same first overflow9.59375. This is forced
history evidence,not an additional free-generation repeat matrix.

`r65-four-R32-localization-v1`:one64-token failing prompt replayed with existing
debug-only model/CPU capture,all26layers. Both NPU and Hybrid logits exactly
match their respective original64positions. No observer-induced output change.

At token54 the first router expert-set change occurs at Layer17:
All-NPU `[40,41,25,26,54,62]`,Hybrid `[40,41,25,26,54,29]`.
The smallest baseline router6/7margin is not sufficient attribution:Layer12's
small margin did not flip;the observed first set change is Layer17.

| Layer | MLP input relative-L2 | MLP output relative-L2 |
|---|---:|---:|
|1|0|0|
|9|0.0176189|0.0199085|
|16|0.0385850|0.0525219|
|17|0.0414522|0.2177792|
|23|0.0855135|0.5808219|
|24|0.3619912|0.9168881|
|26|0.6387869|0.9173166|

These are cross-backend layer states with different accumulated hidden inputs,
**not** same-operand individual-expert safety measurements. They must not be
compared directly against the0.01same-operand expert gate.

Four CPU layers×65forwards=260exact transfer/merge checks:
hidden D2H,global IDs D2H,routing weights D2H,CPU output H2D,and FP32 merge all
PASS;Deferred IDs remain-1. Actual physical shapes/masks for26layers validated,
including32resident experts at1,9,17,26. Shared expert fusion disabled by model
configuration,consistent with the split-expert path.

Current classification is **reproducible numerical amplification with observed
router-set changes**,not proof that a particular CPU reduction tree alone is
the root cause. Five implicated same-operand expert probes subsequently passed:
relative-L2range0.00090679–0.00392756,10CPUrepeats each exact. See10_FINAL_ADR.md.

Resource diagnostic window has72samples/no sampler errors:sampled cgroup peak
90,998,427,648bytes,swap0,NPU-owned process peak28,948MiB. This is not a controlled
performance result or proof of DRAM bandwidth saturation.

## Official reference boundary

Re-read the fixed [Ascend tutorial](https://github.com/kvcache-ai/ktransformers/blob/0dce4c9b097710951d1b3e232f50b973ff0a5505/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md).
It demonstrates resident/CPU capacity tiering with DeepSeek-V4 W8A8/MXFP4 and
separate accuracy validation. It does not establish the DSV2BF16/F32 pairwise
bound or prescribe a fix for this isolated logit event. No upstream merge,
quantization switch or unsupported performance inference follows from it.
