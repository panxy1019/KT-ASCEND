# Round 9 minimal SGLang/KTEP runtime integration map

Date: 2026-09-20
Evidence state: `SOURCE_MAPPED`; no import, server, remote, or NPU run is claimed here.

## Fixed bases

| Role | Exact source | Use |
|---|---|---|
| Executable runtime base | `/home/admin/Documents/ChatGPT/KT/kt-round6-deferred/third_party/sglang`, `f02f304a25364d1e8fe9c52515a22f20f5089a52` | Keep this tree and its KTEP/Ascend integration as the framework base. |
| Qwen4 semantic donor | `/tmp/sglang-qwen4-audit-20260914`, `4ccff141dbe992794f9da6c3aa23535b4f72000d` | Read with `git show`/`git grep`; the checkout is sparse. Port semantics, not the whole runtime generation. |
| Last serving overlay | `round8_prep/runtime/kt_ep_wrapper.py`, SHA256 `77dfbbab73431111e72b6ac2683e4f4ecfa617280436e9ce55a3d7f89177e46b` | Later KTEP fixes over the f02 file. |
| Round 9 eager references | `migration_bridge/round9/runtime/{hyperconnection,gdn,ple_hash,ple_rows,ple_sync,qsa}.py` | Canonical correctness implementations for the first Graph-OFF, MTP-OFF path. They are not SGLang adapters by themselves. |

The recovered f02 tree already contains the Round 3 Ascend fixes: guarded CUDA capability probing, non-CUDA router fallback, device-generic KTEP streams and stream handles, all-accelerator-layer bypass, `int32` CPU expert mapping, physical accelerator expert count, and `-1` route sanitization. Do not replay patches `migration_bridge/round3/sglang_patches/0001` through `0008` blindly. Verify their invariants in the resulting files instead.

KTEP remains the only CPU/NPU MoE coordinator. Qwen4 must construct the normal f02 `FusedMoE`/Qwen MoE layer and let its registered `KTEPWrapperMethod` wrap the accelerator method. There is no second model-level Hybrid coordinator and no separate CPU-expert dispatch loop.

## Why 4cc cannot be copied wholesale

The 4cc Qwen4 model is from a different SGLang runtime generation. Its `python/sglang/srt/models/qwen4_exp.py` directly imports interfaces absent in f02, including `sglang.srt.runtime_context`, `sglang.srt.model_executor.runner`, and `sglang.srt.layers.utils`. Its pool changes are split across `mem_cache/kv_cache_configurator.py` and `model_executor/pool_configurator.py`, neither of which exists in f02. Its QSA implementation imports Triton kernels and CUDA-oriented sparse-attention machinery. Replacing f02 with that dependency closure would also replace the already-qualified KTEP and Ascend seams.

The minimal path is therefore a Qwen4 compatibility port onto f02's Qwen3.5 hybrid runtime. This is viable structurally because 4cc's `Qwen4ExpTextConfig` derives from `Qwen3NextConfig`, while f02 already recognizes `Qwen3NextConfig` as a hybrid GDN configuration and already has `GDNAttnBackend`, `HybridLinearAttnBackend`, `HybridLinearKVPool`, `MambaPool`, and the Ascend-only hybrid-GDN backend selection.

## Required source set

### Preserve from f02

These are runtime owners, not donor replacement targets:

- `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- `python/sglang/srt/layers/moe/quant_method_registry.py`
- `python/sglang/srt/layers/quantization/unquant.py`
- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/models/qwen2_moe.py`
- `python/sglang/srt/layers/attention/linear/gdn_backend.py`
- `python/sglang/srt/layers/attention/attention_registry.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/srt/models/registry.py`

The registry discovers a model module's `EntryClass`; a hard-coded Qwen4 entry in the model registry is unnecessary.

### Reapply the later KTEP overlay as one closed unit

Copying only `round8_prep/runtime/kt_ep_wrapper.py` is incomplete. Relative imports added by that file require these sibling modules under `python/sglang/srt/layers/moe/`:

| Destination | Frozen source |
|---|---|
| `kt_ep_wrapper.py` | `round8_prep/runtime/kt_ep_wrapper.py` |
| `bf16_resident_layout.py` | `migration_bridge/round7/tools/bf16_resident_layout.py` |
| `llamafile_bf16_staging.py` | `migration_bridge/round7/tools/llamafile_bf16_staging.py` |
| `verified_bf16_cache.py` | `migration_bridge/round7/tools/verified_bf16_cache.py` |

This unit supplies the pinned BF16 host buffers, Ascend resident-NZ-to-raw normalization, LLAMAFILE F32-to-exact-BF16 staging, device-generic context-load streams, and full-tensor NPU expert copies. The overlay also imports `sglang.srt.debug_utils.round6_1_deferred`; that module is present in f02 and must either remain present or the diagnostic import must be made optional before packaging. Preserve f02's existing `unquant.py` route sanitizer and physical-weight expert count alongside it.

The process-local BF16 cache has an explicit immutable-source assumption. Its use therefore remains coupled to the existing checkpoint/GGUF identity guard; it must not turn an unverified mutable GGUF into a trusted cache source.

### Adapt from 4cc Git objects

The following object paths define the Qwen4 behavior. “Adapt” means translate them to f02 APIs while preserving equations and checkpoint names; it does not mean copy all their imports.

| Qwen4 concern | 4cc object paths | f02 destination/change |
|---|---|---|
| Config schema and layer classification | `python/sglang/srt/configs/qwen4_exp.py` | Add `configs/qwen4_exp.py`; export `Qwen4ExpConfig` and `Qwen4ExpTextConfig` from `configs/__init__.py`; register them in f02 `utils/hf_transformers_utils.py`. Preserve `Qwen3NextConfig` inheritance so f02's hybrid-GDN detection and cache sizing apply. |
| Text model, weight names, GR/PLE ordering, MoE boundary | `python/sglang/srt/models/qwen4_exp.py` | Add an f02-compatible `models/qwen4_exp.py` with `EntryClass = [Qwen4ExpForConditionalGeneration]`. Derive from f02 Qwen3.5 classes and keep f02 forward-batch, loader, PP/TP, and MoE calls. Port only Qwen4 layer deltas. |
| Hyperconnection equations | `python/sglang/srt/layers/hyperconnection.py` | Add `layers/hyperconnection.py` using the eager four-branch semantics validated in Round 9. Do not bring `hc_mix_triton` or CUDA-device construction into the first Ascend path. |
| PLE equations and weight loading | `python/sglang/srt/models/qwen4_exp.py`, `python/sglang/srt/models/qwen4_exp_ple_table.py` | Add a model-facing PLE layer using Round 9 `ple_hash.py`, `ple_rows.py`, and `ple_sync.py`. Preserve original shard names and bounded `pread`; do not materialize the full embedding table. |
| PLE request state | `python/sglang/srt/mem_cache/ple_state_pool.py`, plus the `register_slot_state`, allocation, copy/fork, host-offload, free/reset changes in 4cc `mem_cache/memory_pool.py` | Add `ple_state_pool.py` and surgically extend f02 `MambaPool`/`HybridReqToTokenPool`. The n-gram history and short-convolution window must follow the same slot lifecycle as GDN state. Patching allocation alone is insufficient because radix fork/copy and clear otherwise reuse stale PLE state. |
| QSA index and sparse attention semantics | `python/sglang/srt/layers/attention/qsa/{config,metadata,mqa,qsa_indexer}.py`, `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py` | Add an f02-compatible eager QSA backend which obtains each request's logical token rows from `HybridLinearKVPool`, applies the Round 9 canonical compression/selection/GQA operations, and writes/reads the ordinary f02 KV pool. Do not port `qsa/{kernel,sparse_attn,graph_metadata}.py` for the first NPU path: they are Triton/CUDA implementations. |
| Backend selection | 4cc `python/sglang/srt/layers/attention/attention_registry.py` | Extend f02's existing hybrid-GDN branch so Qwen4 full-attention layers select the eager QSA backend while linear layers still use f02 `GDNAttnBackend`. Preserve `--attention-backend ascend` for the underlying NPU full-attention/KV operations. |
| Cache construction | 4cc `mem_cache/kv_cache_configurator.py`, `model_executor/pool_configurator.py`, and `mem_cache/memory_pool.py` | Translate only the Qwen4 arguments into f02's monolithic `model_runner_kv_cache_mixin.py` and `memory_pool.py`: QSA full-attention layer IDs, PLE layer IDs, short-conv shape, n-gram context length, EOS, and their memory charge. Do not import the two new configurator frameworks. |
| Architecture flags | 4cc `configs/model_config.py`, `arg_groups/overrides.py`, `server_args.py` | Add only checks needed to identify `Qwen4ExpForConditionalGeneration`, text-only operation, and QSA selection. Keep MTP, graph capture, speculative decoding, radix reuse, PP, and TP>1 disabled for the first executable profile. |

`qwen4_exp_mtp.py`, MTP shared sparse indices, graph metadata, CUDA fused kernels, ModelOpt/NVFP4, and multimodal processor changes are outside the first integration. Their presence in 4cc is not a prerequisite for the stated MTP-OFF text path.

## Ordered implementation and evidence gates

1. **Freeze the base.** Create a clean copy/worktree at f02, record the f02 tree hash, the overlay and helper hashes, and a post-overlay diff. Do not install or copy 4cc as a package.
2. **Restore the proven MoE plane.** Install the four-file KTEP overlay unit above, retain the f02 Ascend `unquant.py` invariants, and verify source-level KTEP registration. A Qwen4 MoE layer must still instantiate f02 `FusedMoE`; the registered wrapper is the only CPU/NPU coordinator.
3. **Add config discovery only.** Port Qwen4 config/export/HF registration and resolve the checkpoint config in text-only, TP1, MTP-OFF mode before constructing a model. This gate must not allocate model weights.
4. **Add the model skeleton and GR.** Adapt `qwen4_exp.py` to f02 Qwen3.5 APIs, load one layer's hyperconnection tensors by their original names, and compare its mix/combine result with the Round 9 eager reference.
5. **Attach GDN through f02.** Reuse f02's Qwen3.5 projections, `GDNAttnBackend`, Ascend selection, and `MambaPool`; use the Round 9 GDN implementation only as the independent oracle. The initial reference profile uses the audited whole-module GDN equations inside the actual f02 model and its real MambaPool. Report29 qualifies this reference adapter, not the optimized backend.
6. **Add PLE with owned state.** Port bounded shard-row loading, then extend the Mamba slot lifecycle before enabling PLE in a model layer. Gate prefill versus tokenwise equivalence, chunk continuation, reset, fork/copy, and request-slot reuse.
7. **Add eager QSA.** Implement the f02 backend adapter around the Round 9 equations and ordinary paged KV addresses. Gate logical-position causality, partial compression groups, duplicate/invalid selection handling, chunked prefill, and decode continuation. No CUDA/Triton module may be imported on this path.
8. **Close weight loading.** Audit every Qwen4 checkpoint prefix, including GR, PLE projection/norm/conv, PLE shard table, QSA index projections/norm, GDN `A_log`/`dt_bias`, shared expert, router, routed experts, and output projection. Fail on unexpected missing or duplicate loads; preserve `A_log` and recurrent state FP32 semantics.
9. **Construct a bounded model.** With Graph OFF, MTP OFF, radix cache OFF, text only, TP1, and one request, construct and load a bounded layer subset. Compare per-boundary outputs against the already-qualified eager references before attempting all layers.
10. **Run the full-model admission sequence.** Only after bounded construction passes: full checkpoint identity, full load accounting, prefill/decode continuity, request reset/isolation, KTEP route/weight/shared-expert exactly-once checks, memory admission, then deterministic generation/quality. A successful primitive or bounded-layer run is not full-model admission.

## Dependency decision for the torch 2.9/CANN 9 image

The current image records Python 3.11, `torch 2.9.0+cpu`, `torch_npu 2.9.0.post2`, and CANN 9.0. Earlier Round 8 evidence records successful torch/torch_npu device detection, KT import, and representative Qwen/GDN/expert primitives on that stack. That supports preserving the system accelerator stack.

The f02 `python/pyproject.toml` nevertheless declares `torch==2.9.1`, `torchaudio==2.9.1`, `torchvision==0.24.1`, `sgl-kernel==0.3.21`, `flashinfer_*==0.6.3`, and `transformers-kt==5.6.0.post3`. A normal dependency-resolving install would therefore try to replace the image's torch stack and pull CUDA-oriented packages. That install route is not acceptable.

The bounded import candidate is:

- keep system `torch 2.9.0+cpu`/`torch_npu 2.9.0.post2`/CANN 9 unchanged;
- expose the f02 Python tree by a no-dependency editable install or `PYTHONPATH`;
- rebuild the Round9 KT native package with Ascend enabled and requalify its imports, streams and callbacks; older Round8 binaries do not transfer qualification;
- install only missing pure-Python dependencies into the private venv, with exact versions and `--no-deps` where required;
- avoid importing CUDA-only FlashInfer, Triton QSA, CuTe, ModelOpt, or CUDA kernel modules in the Qwen4 Ascend profile.

Source inspection therefore says a wholesale upgrade is unnecessary and would be unsafe for the qualified accelerator pairing. It does **not** establish that the full f02 package currently imports in the image. Before any model claim, capture four separate import gates in the target container: `torch`+`torch_npu`; `kt_kernel`; the f02 KTEP module with its three overlay helpers; and the new Qwen4 config/model/backend modules. Record versions and resolved file paths. Any failure should be fixed at its exact optional-import or API seam rather than by running the f02 dependency resolver.

## First supported profile

The smallest defensible integration profile is text-only, TP1, one request, eager execution, Graph OFF, MTP OFF, speculation OFF, radix cache OFF, and synchronous PLE/QSA state transitions. KTEP remains enabled only through `KTEPWrapperMethod`; shared expert and routed scaling stay at the f02 Qwen MoE boundary and must each apply exactly once. Expansion to graph capture, radix reuse, TP/PP, multimodal, MTP, or optimized QSA requires separate state-lifecycle and backend evidence.
