# Official compatibility matrix

Audit date: 2026-09-18. Pins are in `pins.json`. Presence in source is not NPU qualification.

| Boundary | Observed source evidence | Status / action |
|---|---|---|
| Qwen3.5 model | KT pinned `models/qwen3_5.py` exists; MoE text type constructs `Qwen2MoeSparseMoeBlock` | CODE_INSPECTED; no new coordinator |
| MoE integration | `qwen2_moe.py` uses `get_moe_impl_class`, TopK from config, separately gated shared expert | Trace actual backend dispatch in isolated construction before claiming KTEP active |
| GDN | KT pinned source has separate qkv/z/b/a projections; current upstream has fused qkvz/ba | CONFLICTING_VERSION_SEMANTICS; no blind PR copy |
| Shared expert | sigmoid gate times shared MLP, added to routed result | Preserve once; shared fusion disabled initially |
| Routing | TopK=8, renormalization from `norm_topk_prob`; no justification for copying DSV2 routed scaling | Derive from Qwen config/model, test once-only weighting |
| Text only | KT server args contain `--language-only` | Verify actual deployed candidate CLI and model path; absence of image requests alone does not prove vision weights skipped |
| BF16 CPU | KT tutorial explicitly x86/AVX512, CPU method BF16 dispatched through NativeMoEWrapper | UNKNOWN_NEEDS_ARM_PROBE |
| LLAMAFILE | Factory supports method; historical DSV2 ARM C64 evidence exists | New Qwen dimension512/256-expert/packed loader needs new operator probe |
| NPU | Upstream Qwen source contains NPU branches and sgl_kernel_npu imports | Need compatible GDN operator package; CUDA tutorial is not Ascend proof |
| Python deps | KT pinned pyproject: torch2.9.1, transformers-kt5.6.0.post3 | Conflicts with recorded Round7 torch2.9.0 stack; isolated resolution required |
| Quant loader audit | Guessed `hardware_backend/npu/quantization/loader.py` raw URL returned404 for both pins | NOT_AUDITED; locate real module before patch, do not interpret404 as missing support |

Primary sources: [KT tutorial](https://github.com/kvcache-ai/ktransformers/blob/105359b4bf70073af9fab992ad29920666acccc9/doc/en/Qwen3.5.md), [KT child model](https://github.com/kvcache-ai/sglang/blob/3424f35d0e60f03b9e75d9b90941f8f4cb944524/python/sglang/srt/models/qwen3_5.py), [upstream model](https://github.com/sgl-project/sglang/blob/9784d5f979c9eaa5a6d9b31e596a7a73cb76dab8/python/sglang/srt/models/qwen3_5.py).
