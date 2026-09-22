# KT / Qwen / Ascend: audited execution map

Evidence level: frozen-source audit. Physical provider execution coverage will be appended from this round. Full path, SHA256 and symbol lines are in `source-map.json`; binary identities are in `baseline-lock.json`. Public references are semantic context only, not an imported dependency update.

| Plane | Actual frozen path | Meaning / boundary |
|---|---|---|
| Serving | KT-integrated SGLang checkout `f02f304a25364d1e8fe9c52515a22f20f5089a52` plus recorded local port/overlay | HTTP/tokenizer → scheduler serial batch → model runner. Upstream SGLang is not substituted. |
| Model | `sglang/srt/models/qwen4_exp.py:Qwen4ExpForConditionalGeneration` | Local Qwen3.8 reference adaptation, 48 language layers, guarded single text request, GDN/QSA/GR/PLE state semantics. |
| Placement | `kt_ep_wrapper.py:KTEPWrapperMethod`, framework hooks | Static uniform32/512 per layer, physical32 BF16 weights. Model identities remain Qwen-specific. |
| Prefill routed experts | `prefill_binding.bind_expert_method` → `PrefillRuntime.apply` → `LayerwiseTransport` → private Ascend BF16 backend | When Layerwise active, bypass original CPU/NPU hybrid apply; stream real logical experts through bounded shadow. Source is original BF16, including routes backed by MXFP4 on CPU in decode. |
| Decode routed experts | original `KTEPWrapperMethod.apply` | Stage input, launch CPU branch, mask/remap resident IDs, run NPU resident experts, wait CPU contribution, F32 partial merge then BF16. |
| CPU engine | `kt_kernel.utils.llamafile.LlamafileMoEWrapper` / `experts_base` / native extension | Actual GGUF MXFP4 weights; Q8 activation packing in the qualified oracle; C128 four NUMA pools. Native SHA `614ae60f70ff70d1e47d7fd32e857f887f874b6248eb332172c8a637cb471a6d`. Python submit duration is not worker compute time. |
| NPU expert | `UnquantizedFusedMoEMethod` Ascend dispatch | Original BF16 resident/shadow weights, processed transposed NZ29; canonical tensors are ND. Same-bank CPU reference and original-BF16 NPU reference must be combined separately. |
| GDN | `Qwen4ReferenceGatedDeltaNet.forward` → `round9.gdn.eager_gdn` / `gated_delta_recurrence` | **Eager reference**, token recurrence and FP32 persistent recurrent state; projections use inherited linear layers. This is not a claim of an optimized fused GDN kernel. |
| QSA | `Qwen4ReferenceIndexer.forward` / `Qwen4ExpAttentionDecoderLayer.self_attention` → `round9.qsa.select_tokens/sparse_gqa` | **Eager reference** index scores/topk/gather and row-loop attention, QSA4 compression. Ascend server backend flag does not mean this custom path uses a fused Ascend sparse-attention kernel. Short prompts do not qualify the >512-block selection pressure branch. |
| GR | `round9.hyperconnection.GatedResidual.mix/combine` | Eager Torch projections, normalization, sigmoid and gated residual mixing. No new GR kernel this round. |
| PLE | `Qwen4ReferencePLE`, original BF16 shard reader, `PLEConsumerRuntime`, existing2slot async ring | v61 profile explicitly ASYNC; global default remains OFF. Current/known input only; events and consumer lifetime retained. |
| Context | request-local QSA KV/index pool, GDN pool, PLE history | Prefix OFF. Existing compact/reference context integration does not transfer allocator authority or establish HiCache. |

Three independent granularities: scheduler chunks32 tokens; CPU expert workspace/microbatch <=32; shadow prepare/release keyed by logical layer and routed expert set within the bounded transport lifetime. The old report's repeated-shadow heuristic used `(layer, logical_expert_list)`, not a full model/source/layout identity. We retain old files and correct their interpretation here.

Public source checks on 2026-09-22: [KT-Kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md) documents independent APIs and KT SGLang integration; [Native Precision](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/Native-Precision-Tutorial.md) supplies the hybrid/layerwise reference terminology; [Ascend support table](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/reference/support_features.mdx) does not replace this local mixed-state audit. These moving pages are `WEB_REFERENCE_ONLY`, not a pinned local binary qualification.

NPU resident/shadow operator lowering was further audited in frozen `unquant.py` (lines577–637): `torch.ops.npu.npu_moe_init_routing_v2` → `npu_grouped_matmul` → `npu_swiglu` → `npu_grouped_matmul` → `npu_moe_finalize_routing`, with the KT F32-contribution branch before final hybrid merge. This is source dispatch evidence; individual device kernel durations remain outside the sampled module spans.

The actual CPU bridge uses `get_current_device_stream_handle(hidden_states.device.type)` and borrows public `npu_stream`; it does not cast the NPU handle to a CUDA ABI. Native Ascend callback worker subscribes to ACL reports; submitted callbacks then reach CPUInfer. Internal queue/packing/GEMM timestamps are not exported by this round's Python seam and remain UNKNOWN.

Actual placement correction from full-model evidence: the frozen server's `uniform` flag produces logical IDs0–31 in every layer (all48 checked in v63-q32-physical.json). It is not the interval16 fixture used by the standalone Q32 qualification probe. The real replay uses the captured mapping, preserving the actual CPU/NPU precision split. Captured layer0 routes contain NPU-resident IDs20 and14, plus8 CPU-routed assignments.
