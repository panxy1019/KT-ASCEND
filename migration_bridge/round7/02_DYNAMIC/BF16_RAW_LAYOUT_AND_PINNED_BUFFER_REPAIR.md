# BF16 full-context raw-layout and pinned-buffer boundary

## Preserved failure

Runtime-v5 / tools-v16 / backing-requalification-v4 closed operators100/100,
Uniform32+restart4, Frequency32+restart4, and direct All-NPU174/256 invalid0.
The first threshold256 triggered probe then failed before generation publication
with `LLAMAFILE_BF16_STAGING_ASCEND_LAYOUT`. The failed process is reaped;
threshold512/1024, threshold selection, HOLDOUT and Dynamic quality have not run.

Lossless local failure archive: `raw/dynamic-bf16-raw-layout-failure-v1.tar.gz`;
SHA256 `979182fe2e5fa8700fb17e8080131245561e8e14fd04038683e37092acd94cc3`.
Complete SCP and local hash/tar inspection succeeded. It contains failed probe,
predecessor Frequency/All-NPU raw results and runtime-v5 source/provenance.

## Reproduction and root-cause boundary

CODE_INSPECTED: SharedFullContext creates buffers from raw weight shapes before
UnquantizedFusedMoEMethod's NPU post-loading transpose. Staging originally accepted
only inference layouts: w13[2,H,2I], w2[2,I,H]. It must also accept the paired raw
layouts w13[2,2I,H], w2[2,H,I]. Mixed/unknown layouts remain rejected.

A3_VERIFIED: `probe_bf16_context_buffers.py` called the actual runtime-v5 allocator
with reduced raw-shaped BF16 NPU parameters. Both actual CPU buffers were unpinned;
their collected TP1 pointers were correct and CUDA availability was false.
This establishes a separate allocator compatibility gap, not proof that unpinned
memory caused the layout exception. The probe used no model requests and released
only its own context buffers.

## Minimal repair

- Stage exact-expanded F32 GGUF in either explicitly paired layout. Raw buffers
  receive raw matrices; inference buffers receive transposed matrices. No changed
  expert arithmetic, routing, scaling, selected experts or acceptance criterion.
- BF16 NPU TP1 buffers use framework-owned pinned CPU tensors. CUDA/other
  quantization paths retain their original shared-memory allocator. TP>1 is
  explicitly rejected on the new NPU allocation branch.
- Original double-buffer device-event fences remain intact. Tensor references are
  owned by the context; no raw ACL pointer transfer or duplicate coordinator.
- Source guarded private deployment preserves original runtime-v5 and source.

LOCAL_VERIFIED:99 pure-Python tests PASS, including paired-layout validation,
mixed-layout rejection, pinned allocation, TP1 restriction and unchanged non-NPU
or non-BF16 branches. Actual original deployed source passed all port guards.

A3 validation pending: raw and inference staging exactness against frozen source
and checkpoint, H2D identity, negative lossy-source rejection; reduced actual
allocator pinning/pointers; ordered model predecessor and triggered migration
probes. This document does not claim Dynamic qualified.

## Official cross-check

[Pinned KT SGLang BF16 full context](https://github.com/kvcache-ai/sglang/blob/3424f35d0e60f03b9e75d9b90941f8f4cb944524/python/sglang/srt/layers/moe/kt_ep_wrapper.py)
allocates from raw shapes and uses CUDA-only host registration.
[Pinned unquantized MoE](https://github.com/kvcache-ai/sglang/blob/3424f35d0e60f03b9e75d9b90941f8f4cb944524/python/sglang/srt/layers/quantization/unquant.py)
defines raw projection shapes and performs NPU transpose after loading.
Reviewed2026-09-17; no wholesale upstream update.
