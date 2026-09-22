# Round8 Qwen port preparation — research only

Audit date2026-09-17. No Qwen production port or model-weight download is made.
Round7 completion does not qualify Qwen Dynamic or Deferred.

## Candidate freeze

Primary requested candidate: `Qwen/Qwen3.8-Flash-Next`.
HF revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, obtained with `hf models info`.
Downloaded only official config.json at that exact revision, SHA256
`889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b`.
Architecture `Qwen4ExpForConditionalGeneration`, model_type `qwen4_exp`, text
model_type `qwen4_exp_text`. This is not interchangeable with Qwen3-Next.
[Official config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json).

## Verified config structure

48 decoder layers, 512 routed experts, top10, shared-expert intermediate640,
routed intermediate640, hidden2560. The layer_types list has36 linear-attention
and12 full-attention layers. Linear output gate is sigmoid. Residual configuration
has hc_count4 and hc_lowrank320. PLE is at layer2, ngram_size3 and
ngram_vocab_size_base20,000,000. Attention indexer budget2048, compression4.
The multimodal config has vision tower; text-only serving must be explicitly
designed and audited, not assumed from the DeepSeek integration.

The config advertises BF16 dtype, not proof of all checkpoint tensor formats.
Exact shard/weight-format manifest, tokenizer hash, required RAM and conversion
identity are pending a Round8 artifact preflight. A new F32 GGUF would be much
larger than DeepSeek's; do not reuse its192GiB container admission assumptions.

12.5% residency maps to64/512 resident per layer (3072 routed slots for48 MoE
layers if every decoder layer is routed). A256-expert model would map to32/256;
that example does not describe this actual candidate.

## Official support boundary

Official KT snapshot0b954fe4… documents Qwen3-Next-80B-A3B, not qualification of
Qwen3.8's qwen4_exp path. The official support request
[#2179](https://github.com/kvcache-ai/ktransformers/issues/2179) was still OPEN
when inspected. It identifies GDN/QSA, four-branch gated residual and PLE as
integration differences. This issue is a request, not a passing Ascend test.

At audited kvcache-ai/sglang main541ddc37…, model-file inventory contains
`qwen3_next.py` and its MTP implementation, but no qwen4_exp model file. Existing
Qwen3-Next has NPU-related branches; they do not prove qwen4_exp compatibility.
[Official Next tutorial](https://github.com/kvcache-ai/ktransformers/blob/0b954fe4ca2f9e79519da66156e6a3f92b686232/doc/en/Qwen3-Next.md).

## Round8 entry/preflight

1. Freeze Round7 selected scheduler, runtime source/native/environment evidence.
2. Inspect upstream SGLang qwen4_exp implementation and its Ascend kernels:
   GDN, QSA/indexer, gated residual, PLE lookup/offload, text weight loader.
3. Freeze supported Python/Transformers/CANN/torch_npu/operator versions from
   actual kernel requirements. Current CANN9/torch_npu2.9 alone is not sufficient
   evidence; config's Transformers5.8dev is metadata, not a prescribed install.
4. Establish fresh All-NPU or canonical accelerator reference if capacity permits.
5. Static Heavy-Offload, TP1, batch1, Dynamic/Deferred/Graph/MTP/speculative OFF.
6. Local operators → one layer → four layers → all routed layers, independent
   system/determinism/quality/capacity evidence; no inherited DeepSeek readiness.
7. Only then separately requalify Frequency, Dynamic and Deferred.

Current preparation decision: **BLOCKED_WITH_REASON** for immediate production
port: missing audited qwen4_exp KT integration and Ascend operator/artifact
preflight. This does not block Round7. Qwen3-Next-80B-A3B-Instruct is a separately
documented fallback candidate, not a silent substitution for the requested model.
