# LLAMAFILE BF16 full-context staging compatibility

2026-09-17. Evidence vocabulary: CODE_INSPECTED / LOCAL_VERIFIED;
A3 staging and triggered migration are pending, not qualified by this document.

## Reproduction and root-cause boundary

`backing-requalification-v3` on private runtime-v4 closed operators100/100,
Uniform compact32+restart4, Frequency compact32+restart4 and directR64174/256,
invalid0. First threshold256 triggered DEV request then failed before migration
publication: `LlamafileMoEWrapper` lacks
`submit_write_weight_scale_to_buffer`. The initial short/no-trigger request
completed; guard reaped owned model processes. No512/1024 probe, threshold
selection, HOLDOUT or Dynamic quality gate ran.

The pinned official [LLAMAFILE wrapper](https://github.com/kvcache-ai/ktransformers/blob/0b954fe4ca2f9e79519da66156e6a3f92b686232/kt-kernel/python/utils/llamafile.py)
loads GGUF into CPUInfer but provides no weight-staging writer. The pinned
[BF16 full-context pipeline](https://github.com/kvcache-ai/sglang/blob/3424f35d0e60f03b9e75d9b90941f8f4cb944524/python/sglang/srt/layers/moe/kt_ep_wrapper.py)
calls that interface. The deployed source independently reproduced this seam;
AMX writer support cannot be assumed to cover LLAMAFILE.

## Lossless original failure

`raw/dynamic-llamafile-staging-failure-v1.tar.gz`, SHA256
`565a84a404e2dbffc24a678d1a80b429edce51d00ce68857418b5e38e588d9a0`.
Contains original failed requests/routes/generation-consumption, resolved command,
actual process stack, lifecycle/guard/reaping, driver progress, runtime-v4
provenance and exact imported wrapper source. SCP completedexit0, A3/local hash
match, full tar validationexit0. Original directory remains immutable on A3.

## Minimal production-compatible seam

Only the BF16 context method selects an F32-GGUF stager when its wrapper has no
native write interface. Real native writers retain the original branch. The
adapter accepts LLAMAFILE, TP1, CPU pinnedBF16 destinations and explicit Ascend
layout only. It uses the wrapper's existing frozen mmap loader and global
logical expert IDs, validates GGML F32 and complete source size, rejects lossy
BF16 conversion/nonfinite weights, and writes gate/up/down into the existing
double buffers with CPU Tensor.copy_. It never reads checkpoint weights for
production staging; checkpoints are an independent test oracle only.

Ascend destination orientation: w13[hidden,2*intermediate] contains gate.T then
up.T; w2[intermediate,hidden] contains down.T. No raw pointer transfer, arithmetic
change, ownership change, threshold suppression, duplicate coordinator or new
quantization. Host staging is synchronous, but original H2D stream/event double
buffer fences remain unchanged, including synchronization before slot reuse.
This is not a sequential-control override of expert execution. Its overhead
must be included in later real Dynamic performance.

Prepare a new private runtime and freeze the helper hash in provenance. Original
deployed source/native and runtime-v4 results must not be overwritten. Current
94 pure-Python tests pass locally; this is not A3 qualification. Next execute
independent early/middle/late exact GGUF→buffer→NPU→checkpoint comparisons and
repeat/slot-isolation/invalid-source controls, then ordered model predecessors
and the original DEV probes in a new result directory.

## A3 standalone verification — A3_VERIFIED

Tools-v16 source commit `a1c822f75ddd8b853b46520f6323f67bc2365d32`, archiveSHA
`6f87d6b0dfcc20ef4f5d77e6dd077273c7ddba3a17889f0c859110320e1155f6`.
Initial upload into a root-owned artifact directory was permission-denied;
corrected SCP to admin's explicit file completedexit0, host/container hashes
verified before extraction. No failed transfer was treated as deployment.
All94 pure-Python tests also passed on A3 Python3.11.

New private runtime-v5 wrapperSHA
`5762b3386e5bf8e9ee0e36ee922fd38fe073a1b75bb1a7ab3330dc01b0e814cf`;
stager helperSHA
`b623670d0043a4d55359e4497fe06521aaed059d433fcb76105b98dd82948f44`.
The private CPU native remains461821...; original imported source/native untouched.

Standalone real-weight staging regression PASS12 cases at layers1/13/26 and
logical experts0/1/31/63, four repeats each: both projections exact against
independent frozen checkpoint, H2D exact, all nine applicable other-slot
checks exact, lossyF32 source rejected. Original mmap remains read-only;
the negative control mutates a private test clone only. This certifies bounded
same-source conversion/layout and transfer, not the model's full pipeline,
hot-set selection, transactional publication or Dynamic performance.

Lossless `raw/llamafile-staging-regression-evidence-v1.tar.gz`, SHA256
`8e91dec9f052d1a0f39227b87fe62cb53b86cd4b2cc234a964ef82983b07aecb`,
contains result/log, runtime-v5 provenance and exact private source/helper,
tools-v16 release. SCPexit0, A3/local hash match, full tar validationexit0.

At21:43UTC, new driver `backing-requalification-v4` is RUNNING phaseoperators,
using tools-v16/runtime-v5 with no copied PASS summaries and no resume. Ordered
operators→Uniformcompact→Frequencycompact→directR64→DEVprobes256/512/1024.
Keep healthy driver intact. Nochosen threshold, noHD or Dynamic/Deferred product
promotion. New full-model failure, if any, must remain preserved separately.
