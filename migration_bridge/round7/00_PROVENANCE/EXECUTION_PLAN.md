# Round7 execution freeze

Date: 2026-09-17 UTC. Evidence labels: CODE_INSPECTED, LOCAL_VERIFIED,
A3_VERIFIED. A stage is not COMPLETE until its preserved results pass its gates.

Base: `bc2e5d53e05d97db83b40c09a6f80cddff0549ab`.
Branch: `codex/round7-static-frequency-dynamic-deferred`.
Local parent gitlink: `f02f304a25364d1e8fe9c52515a22f20f5089a52` (submodule not
materialized in the new audit worktree). This is distinct from the actually
deployed eaa58c… child plus overlays below; it must not be reported as the child
used for A3 execution. Round7 does not silently substitute local gitlink files
for the frozen, already-qualified imported runtime.
Official KT audit: `0b954fe4ca2f9e79519da66156e6a3f92b686232`.
Its SGLang gitlink: `3424f35d0e60f03b9e75d9b90941f8f4cb944524`.
Latest official SGLang audit: `541ddc37cbc92c60dc748db5ff1a2aad0b069a80`.
These are comparison snapshots, not an authorization to replace the deployed runtime.

## Ordered checkpoints

1. Preserve actual imported Python/native/container/model/corpus provenance.
2. Freeze independent DEV 256 and HOLDOUT 128, tokenizer-counted buckets and no
   overlap with CMMLU. Commit before activation collection.
3. Uniform R8-C64: exact official stat recorder, DEV activation collection.
   Freeze global frequency mask, routed resident budget 208, dense layer excluded.
4. Frequency system/restart/observer/operator gates; HOLDOUT route/performance
   comparison; fixed CMMLU 256. Choose static by routing/performance, not answers.
5. Audit actual full-prefill/update path; DEV threshold sweep; freeze threshold.
   No-trigger, trigger, copied BF16 identity, generation publication, workload
   shift, restart, HOLDOUT/performance/CMMLU.
6. Selected scheduler D0/D1 independent system/performance/quality evaluation.
7. Final evidence hashes, ADR, selected defaults, Qwen research-only handoff.

## Immutable acceptance

CMMLU manifest `bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04`;
R64 reference 174/256; product pass >=162/256 with invalid=0. Local operator
relative L2 <=0.01. Same-path/restart determinism, weight/mapping/ownership,
exactly-once weighting/scaling/shared, transfer lifetimes remain hard gates.
Generation publication must not expose mixed weights/mapping/masks.

CPU64 means two CPUInfer threadpools of 32 on NUMA4/5, cpuset128-191, not NPU TP2.
NPU TP1, batch1, Graph/MTP/speculative OFF. R16-C64 remains fallback.
Deferred remains OFF unless the independent D1 evaluation supports selection.
Dynamic/Deferred no benefit is a legitimate completed classification.
No push, merge, tag, weight edits, sample removal or post-result threshold changes.
Historical Round6.4 no-speedup and Round6.5 numerical rejection are not rewritten.

## Existing runtime

Owned container `kt-r67-c64-development-v1`, physical NPU2 (logical0 inside),
host110.129.0.22. Actual parent e3084d309d6f39b8a9f7185d36ac27e4b77526d7,
child eaa58ccea4ee372b8b207dd16f398eafe3da37ff plus preserved overlays.
New artifacts are confined to `/workspace/c64-dev/round7/`; prior frozen evidence
and business containers are not modified. All model artifacts are mounted RO.

Official primary sources:
[KT scheduling](https://github.com/kvcache-ai/ktransformers/blob/0b954fe4ca2f9e79519da66156e6a3f92b686232/doc/en/kt-kernel/experts-sched-Tutorial.md),
[mask implementation](https://github.com/kvcache-ai/ktransformers/blob/0b954fe4ca2f9e79519da66156e6a3f92b686232/kt-kernel/python/experts_base.py),
[pinned integration](https://github.com/kvcache-ai/sglang/blob/3424f35d0e60f03b9e75d9b90941f8f4cb944524/python/sglang/srt/layers/moe/kt_ep_wrapper.py).
