# Qwen3.5 text execution boundary

Frozen config: 40 MoE layers; 30 GatedDeltaNet linear-attention blocks +10 full-attention blocks. Hidden2048, routed256, topK8, routed/shared intermediate512. Vision branch present in checkpoint; language-only must be verified. MTP1 exists but stays OFF.

Text tokens → embeddings → attention/GDN → residual/norm → router TopK8 → existing MoE implementation/KTEP boundary → disjoint CPU/NPU expert computation → routed merge; separately gated shared MLP added once → residual → LM head.

`Qwen2MoeSparseMoeBlock` is reused by KT pinned Qwen3.5. Router renormalization and shared gate semantics come from Qwen, not DeepSeek scaling constants. Runtime audit must trace exact chosen forward variant, fused/unfused paths, final dtype, borrowed streams, transfer lifetime, physical slot count and placeholder zero-weight behavior.

Uniform Q64 means64 resident of256 per layer, **not64 activated**. Each token selects8 routed experts. Actual CPU/NPU split depends on routing; 25%/12.5% residency does not prove25%/12.5% route hit rate.

Static JSON mask tests validate intended mapping only. Actual wrapper construction, weight loading, NPU GDN and shared-expert output have NOT been exercised.
