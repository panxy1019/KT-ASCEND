# First failure — four-layer R32 numerical envelope

Date2026-09-16. State:INVESTIGATION_IN_PROGRESS,not accepted/not repaired.

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

## Official reference boundary

Re-read the fixed [Ascend tutorial](https://github.com/kvcache-ai/ktransformers/blob/0dce4c9b097710951d1b3e232f50b973ff0a5505/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md).
It demonstrates resident/CPU capacity tiering with DeepSeek-V4 W8A8/MXFP4 and
separate accuracy validation. It does not establish the DSV2BF16/F32 pairwise
bound or prescribe a fix for this isolated logit event. No upstream merge,
quantization switch or unsupported performance inference follows from it.
