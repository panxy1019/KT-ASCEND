# Qwen4-Exp minimal text decoder/model port design

Date: 2026-09-20
Target runtime: recovered SGLang/KTEP `f02f304a25364d1e8fe9c52515a22f20f5089a52`
Semantic donor: SGLang `4ccff141dbe992794f9da6c3aa23535b4f72000d`
Initial execution profile: text only, TP=1, batch=1, eager execution, radix cache OFF, speculative decoding/MTP OFF, graph capture OFF.

This is a source-derived implementation design, not an import or full-model test result. It preserves KTEP/SGLang as the sole scheduler, request owner, memory-pool owner, and model runner. The Round 9 PyTorch GDN, QSA, PLE, and hyperconnection modules are computation components inside that runtime; they do not introduce a second coordinator.

## 1. Source facts that constrain the port

The recovered base is `third_party/sglang/python/sglang/srt/models/qwen3_5.py` at f02. Its reusable public boundaries are:

```python
class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config, layer_id, quant_config=None,
                 alt_stream=None, prefix="") -> None: ...
    def forward(self, hidden_states, forward_batch): ...

class Qwen3_5LinearDecoderLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None,
                 prefix="", alt_stream=None) -> None: ...
    def forward(self, hidden_states, residual, **kwargs): ...

class Qwen3_5AttentionDecoderLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None,
                 prefix="", alt_stream=None) -> None: ...
    def forward(self, positions, hidden_states, residual,
                forward_batch, **kwargs): ...

class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix="") -> None: ...
    def forward(self, input_ids, positions, forward_batch,
                input_embeds=None, pp_proxy_tensors=None,
                input_deepstack_embeds=None): ...
```

Two donor assumptions are absent in f02 and cannot be copied verbatim:

1. f02 `Qwen3_5ForCausalLM` hardcodes its module-level `ALL_DECODER_LAYER_TYPES`; it has no `decoder_layer_types` override hook, `_build_embed_tokens`, or `is_nextn` constructor argument.
2. Both f02 Qwen3.5 decoder constructors accept only `config.model_type == "qwen3_5_text"` or `"qwen3_5_moe_text"`. The checkpoint text config is `qwen4_exp_text` and has 512 experts, so direct donor `super().__init__(config, ...)` raises `ValueError`.

The actual checkpoint metadata fixes the first profile at 48 layers with the repeating block pattern `[linear_attention, linear_attention, linear_attention, full_attention]`, hence 36 GDN layers and 12 full/QSA layers. It has `hidden_size=2560`, `hc_count=4`, `hc_lowrank=320`, one PLE at one-based `ple_layer_ids=[2]`, `num_experts=512`, and `num_experts_per_tok=10`. The checkpoint architecture is `Qwen4ExpForConditionalGeneration`; its top-level metadata currently says `language_model_only=False`, although this first admission profile deliberately loads only the language model.

## 2. Files to port or add

The minimal source patch inside the f02 overlay is ordered as follows.

1. Add `python/sglang/srt/configs/qwen4_exp.py`, porting `Qwen4ExpTextConfig` and `Qwen4ExpConfig` from donor `python/sglang/srt/configs/qwen4_exp.py`. Keep `Qwen4ExpTextConfig(Qwen3NextConfig)` so existing hybrid-GDN cache sizing applies. Preserve `layers_block_type`, `short_conv_layer_ids`, `short_conv_state_shape`, and `ngram_context_len`. For this profile, the vision config may remain parse-only; it must not instantiate a vision tower.
2. Export `Qwen4ExpConfig` and `Qwen4ExpTextConfig` from `python/sglang/srt/configs/__init__.py`, and add `Qwen4ExpConfig` to `_CONFIG_REGISTRY` in `python/sglang/srt/utils/hf_transformers_utils.py`. This registration must occur before `AutoConfig.from_pretrained`.
3. Add `python/sglang/srt/models/qwen4_exp.py` with the model classes below and `EntryClass = [Qwen4ExpForConditionalGeneration]`. f02 model registry discovery is package based, so the `EntryClass` is sufficient for architecture registration.
4. Add a Qwen4 text-only processor exemption only if startup tries to create a multimodal processor from the architecture name. Do not add Qwen4 to `multimodal_model_archs` for this first profile. The model class itself rejects image/video inputs.
5. Port donor `python/sglang/srt/mem_cache/ple_state_pool.py` and the narrow `HybridReqToTokenPool`/`MambaPool` sibling-state plumbing from donor `memory_pool.py`: `ShortConvPool`, `NGramPool`, `register_slot_state`, and reset/copy/clear hooks. Pass `short_conv_layer_ids`, `short_conv_state_shape`, `ngram_context_len`, and EOS from `model_runner_kv_cache_mixin.py` when constructing the hybrid request pool. f02 has no `short_conv_layer_cache` or N-gram context pool, so model code alone is insufficient.
6. Port the bounded compressed-QSA subset of donor `python/sglang/srt/mem_cache/qsa_kv_pool.py` and select it instead of `HybridLinearKVPool` in `model_runner_kv_cache_mixin.py` when the Qwen4 compressed-QSA profile is active. Require a page size greater than one and divisible by the compression ratio, even with radix disabled.
7. Add thin SGLang adapters for the existing Round 9 components under the overlay, importing or vendoring the reviewed equations: hyperconnection, synchronous PLE, QSA, and GDN reference. The adapters own layout and cache operations; the math modules remain stateless or caller-stateful.

`Qwen4ExpTextConfig` inherits `Qwen3NextConfig`, so f02 `ModelRunner.hybrid_gdn_config` already recognizes it through its existing `isinstance(config, Qwen3NextConfig)` arm after `get_text_config()`. Do not replace `HybridLinearAttnBackend`, request pools, or KTEP scheduling.

## 3. Concrete class boundary

### 3.1 Compatibility construction

Use a shallow compatibility copy for the two inherited decoder constructors:

```python
def _qwen35_moe_compat(config: Qwen4ExpTextConfig):
    compat = copy.copy(config)
    compat.model_type = "qwen3_5_moe_text"
    return compat
```

Pass `compat` to the f02 parent constructor, then restore `self.config = config`. Do not mutate the shared Qwen4 config in place: all 48 layers are built from it, and construction-time mutation can leak into model configuration, cache sizing, or another layer on exceptional exit. The MoE and projection modules retain the same numeric attributes through the shallow copy.

### 3.2 Layer extension and decoder classes

Port these donor method names and call boundaries, adjusted to f02 constructors:

```python
class Qwen4ExpLayerExtensionMixin:
    def _init_qwen4_exp_layer_extensions(
        self, config, layer_id, quant_config=None, prefix=""
    ) -> None: ...

    def _prepare_qwen4_exp_attn(
        self, hidden_states, residual, forward_batch, *, ple_batch
    ): ...

    def _prepare_qwen4_exp_mlp(
        self, hidden_states, residual, forward_batch
    ): ...

    def _run_qwen4_exp_mlp(self, hidden_states, forward_batch): ...

    def _postprocess_qwen4_exp_layer(
        self, hidden_states, residual, forward_batch
    ): ...

class Qwen4ExpLinearDecoderLayer(
    Qwen4ExpLayerExtensionMixin, Qwen3_5LinearDecoderLayer
):
    def __init__(self, config, layer_id, quant_config=None,
                 prefix="", alt_stream=None) -> None: ...
    def forward(self, hidden_states, residual, **kwargs): ...

class Qwen4ExpAttentionDecoderLayer(
    Qwen4ExpLayerExtensionMixin, Qwen3_5AttentionDecoderLayer
):
    def __init__(self, config, layer_id, quant_config=None,
                 prefix="", alt_stream=None) -> None: ...
    def self_attention(self, positions, hidden_states, forward_batch): ...
    def forward(self, positions, hidden_states, residual,
                forward_batch, **kwargs): ...
```

The f02 versions must omit donor `is_nextn`; MTP is outside this profile. Each constructor calls its f02 parent with `_qwen35_moe_compat(config)`, assigns the original Qwen4 config to `self.config`, installs the selected GDN/QSA adapter, then calls `_init_qwen4_exp_layer_extensions`.

`ALL_DECODER_LAYER_TYPES` is local to `qwen4_exp.py`:

```python
{
    "attention": Qwen4ExpAttentionDecoderLayer,
    "full_attention": Qwen4ExpAttentionDecoderLayer,
    "linear_attention": Qwen4ExpLinearDecoderLayer,
}
```

The model constructor maps config `layers_block_type` to this table itself. It adds `.self_attn` to attention-layer construction prefixes and `.linear_attn` to GDN-layer prefixes, matching checkpoint names.

### 3.3 Model and wrapper

Because f02 lacks donor construction hooks, `Qwen4ExpModel` should be a standalone `nn.Module` or subclass `Qwen3_5ForCausalLM` only for helper compatibility while fully overriding `__init__` and `forward`. It must not call f02 `Qwen3_5ForCausalLM.__init__`.

```python
class Qwen4ExpModel(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig,
                 quant_config=None, prefix="") -> None: ...
    def get_input_embeddings(self): ...
    def get_hidden_dim(self, module_name: str, layer_idx: int): ...
    def forward(self, input_ids, positions, forward_batch,
                input_embeds=None, pp_proxy_tensors=None): ...

class Qwen4ExpForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    def __init__(self, config: Qwen4ExpConfig,
                 quant_config=None, prefix="") -> None: ...
    def forward(self, input_ids, positions, forward_batch,
                get_embedding=False, pp_proxy_tensors=None): ...
    def load_weights(self, weights): ...
    @classmethod
    def get_model_config_for_expert_location(cls, config): ...
```

For this profile the wrapper sets `self.config = config.text_config`, constructs `self.model = Qwen4ExpModel(config.text_config, ...)`, `ParallelLMHead`, `LogitsProcessor`, and the usual PP group. It does not construct `Qwen3VLForConditionalGeneration`, a visual tower, a multimodal processor, mRoPE image positions, or `general_mm_embed_routine`. `forward` rejects `forward_batch.contains_image_inputs()` and calls the text model directly, then the standard f02 logits processor. This is required because f02's Qwen3-VL base predates donor `language_model_only` handling and unconditionally constructs/reads `self.visual`.

The wrapper name remains exactly `Qwen4ExpForConditionalGeneration` so it matches the checkpoint architecture without metadata rewriting.

The explicit `packed_modules_mapping` above is required because f02 `Qwen3_5ForCausalLM` has no such class attribute; the donor assignment to `Qwen3_5ForCausalLM.packed_modules_mapping` depends on a newer base. The mapping is metadata for quantization/module matching. The strict loader below still handles expert tensors and the already-packed GDN tensors explicitly.

## 4. Exact decoder ordering and tensor shapes

Let `H=2560`, `N=4`, and `X` be `[tokens, N*H]` after the first expansion.

1. On the first decoder layer only, an embedding output `[tokens,H]` is duplicated by concatenation into four branches `[tokens,N*H]`. Later layers already receive this width.
2. If the layer owns PLE (checkpoint layer index 1, because `ple_layer_ids=[2]` is one-based), compute the PLE query as `X` when residual is absent, otherwise `X + residual`. Add the PLE result to `X` before the attention hyperconnection mix.
3. `attn_hyper_connection.mix(X)` returns the block input `[tokens,H]` and its mix residual/state.
4. Run either Qwen4 GDN or QSA/full attention on `[tokens,H]`, producing `[tokens,H]`.
5. In TP=1 the attention reduction is an identity. Then call `attn_hyper_connection.combine(attn_output, attn_residual)`, yielding `[tokens,N*H]`.
6. Call `mlp_hyper_connection.mix`, yielding the MoE input `[tokens,H]` and MLP residual/state.
7. Run the inherited f02 `Qwen2MoeSparseMoeBlock(hidden_states, forward_batch, ...)` with existing KTEP expert routing.
8. Call `mlp_hyper_connection.combine`, yielding `[tokens,N*H]`; return this and `residual=None`.

Do not retain f02 input/post-attention RMSNorm or `LayerCommunicator` on Qwen4 layers. The mixin removes those parent-created modules. Their checkpoint weights do not exist, and applying them changes the Qwen4 equations.

After layer 47, save the branch tensor only if a future MTP path needs it, then run:

```python
hidden_states, _ = hyper_connection_mixer.mix(hc_hidden_states)
```

The final mixer has `use_combine=False` and returns `[tokens,H]`. There is no final `model.norm` in the checkpoint and no final Gemma RMSNorm in the model. In the MTP-OFF profile, return only `[tokens,H]` to `LogitsProcessor`; no hidden-state side channel is needed.

## 5. Request and state ownership

### 5.1 Common identity

GDN and PLE derive identity from the live `ForwardBatch`, never a Python-global request dictionary:

```python
req_slots = forward_batch.req_pool_indices
state_slots = forward_batch.req_to_token_pool.get_mamba_indices(req_slots).long()
```

For batch=1 the adapter requires exactly one request slot and one mamba state slot. Allocation, zeroing, radix fork/copy, and free remain the responsibility of f02 `ReqToTokenPool`/`MambaPool`. Slot 0 remains reserved for padding. First-profile startup must reject graph padding, radix reuse, and speculative metadata rather than silently assign real state to the dummy slot.

### 5.2 GDN native and `REFERENCE_PATH`

The inherited native path remains `Qwen3_5GatedDeltaNet -> RadixLinearAttention -> HybridLinearAttnBackend -> GDNAttnBackend`. `GDNAttnBackend` already obtains per-layer state through:

```python
layer_cache = req_to_token_pool.mamba2_layer_cache(layer.layer_id)
cache_indices = forward_metadata.mamba_cache_indices
```

The eager Round 9 GDN is also a valid production **reference** backend for the bounded profile when integrated through `Qwen4ReferenceGatedDeltaNet`, a subclass/replacement of `Qwen3_5GatedDeltaNet` selected by an explicit `REFERENCE_PATH` setting. It is not inserted as `self.attn`: `runtime.gdn.eager_gdn` already performs the sigmoid-gated RMSNorm, while the inherited module applies `self.norm` after `self.attn`; that substitution would normalize and gate twice.

`Qwen4ReferenceGatedDeltaNet.forward(hidden_states, forward_batch)` therefore:

1. Reuses inherited `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `conv1d`, `A_log`, `dt_bias`, `norm`, and `out_proj` parameters.
2. Requires batch=1 and eager decode or one packed extend row. It derives the request's mamba slot as above.
3. Reads `conv_state = layer_cache.conv[0][state_slot]` and `recurrent_state = layer_cache.temporal[state_slot]`. The recurrent state stays FP32; the convolution state matches the projection dtype.
4. Calls `eager_gdn` once for the request row, including convolution, recurrence, and sigmoid-gated RMSNorm, then applies only `out_proj`.
5. Copies both returned states back to the same pool slot after successful computation. It never keeps module-global state.

The reference path may be used in production for this bounded profile after full-model equivalence and lifecycle tests. It is expected to be slow. No old native binary qualification is relevant: the current Ascend extension was freshly rebuilt, and native versus reference is a runtime backend choice, not a provenance claim.

### 5.3 PLE

Prepare one request layout once per model forward and pass it as `ple_batch` to every layer. In the first profile it contains one row, its token length, its request/mamba slot, and current N-gram history. Layer 1 uses the synchronous PLE equations and `MambaPool` short-convolution cache; `_commit_ple_batch` advances N-gram context only after all PLE consumers finish.

The PLE short-convolution state is addressed by `state_slot` and the layer id, not held inside `SynchronousPLE`. The adapter reads the state before the call and commits its returned state afterward. Decode, chunked prefill, reset, and request-slot reuse must agree with the same mamba allocation lifecycle as GDN. In this profile asynchronous PLE prefetch, host offload, graph buffers, verify intermediates, and radix track snapshots stay disabled.

### 5.4 QSA

QSA needs the two stores used by donor `QSATokenToKVPool`:

* Ordinary K/V remains in the existing `token_to_kv_pool`; current tokens are written at `forward_batch.out_cache_loc`.
* Compressed index keys live in a token/page side pool. With ratio four, `compressed_slot = full_kv_slot // 4`; therefore the full-KV page size must be a multiple of four. The incomplete group is a four-row pending ring indexed by `req_pool_idx * 4 + logical_position % 4`, and its RoPE coordinate ring uses the same location. Request slot 0 is the inert/padding ring. This is deliberately keyed by the request-pool slot, while GDN/PLE use the request's mamba slot.

For the first profile, port only the BF16 donor layout: one pending index-key ring per full-attention layer, one shared pending RoPE-position ring, and one compressed-key buffer per full-attention layer. Allocation sizing comes from `num_request_slots`, the full-KV slot capacity, `indexer_kv_heads=1`, `indexer_head_dim=128`, and `indexer_compress_ratio=4`. The token allocator owns compressed-slot lifetime arithmetically; no independent QSA allocator or global request dictionary is introduced.

The QSA adapter projects `index_query` and raw index keys with checkpoint `index_qk_proj`, applies the checkpoint Q/K norms and text RoPE, compresses only completed groups of four, and calls the canonical selector. `runtime.qsa.select_tokens` returns **logical token positions**. Resolve each nonnegative logical position through:

```python
physical = forward_batch.req_to_token_pool.req_to_token[
    req_slot, logical_position
]
```

Use those physical slots to gather K/V, then evaluate canonical sparse GQA for this reference profile. Do not pass logical IDs directly to an Ascend sparse-attention API as physical page/slot IDs. The incomplete tail (up to three tokens) is appended causally for each query. The query's visible boundary is its own logical position, so prefill must not expose later tokens in the same chunk.

`RadixAttention.forward` in f02 accepts `**kwargs`, but this alone does not qualify QSA: f02's `AscendAttnBackend.forward_sparse` is shaped around its existing sparse/latent-attention contract. Native QSA requires a separately validated backend/pool port. The bounded reference adapter may gather from the ordinary pool and use `runtime.qsa.sparse_gqa` without replacing KTEP scheduling.

## 6. Complete checkpoint-to-runtime name mapping for the first profile

Normalize the outer prefix first:

```text
model.language_model. -> model.
```

Skip only these declared out-of-profile groups:

```text
model.visual.*  # text-only profile
mtp.*           # MTP OFF
rotary_emb.inv_freq  # derived rotary buffer, if present
```

Every other checkpoint tensor must load exactly once. Unknown or missing non-skipped tensors are fatal; warning-and-continue would create a false full-model pass.

The `layers.L.*` and `hyper_connection_mixer.*` patterns below are relative to the normalized `model.` prefix.

| Checkpoint pattern | Runtime pattern | Packing / axis |
|---|---|---|
| `model.language_model.embed_tokens.weight` | `model.embed_tokens.weight` | direct vocabulary rows |
| `lm_head.weight` | `lm_head.weight` | direct; embeddings are not tied |
| `layers.L.attn_hyper_connection.hc_norm.weight` | same after outer-prefix normalization | direct |
| `layers.L.attn_hyper_connection.input_mix_weight_down.weight` | same | direct |
| `layers.L.attn_hyper_connection.input_mix_weight_up.weight` | same | direct |
| `layers.L.attn_hyper_connection.block_inject_weight.weight` | same | direct |
| `layers.L.mlp_hyper_connection.{hc_norm,input_mix_weight_down,input_mix_weight_up,block_inject_weight}.weight` | same | direct |
| `hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}.weight` | same | direct; final mixer has no block-inject parameter |
| `layers.L.linear_attn.in_proj_qkv.weight` | same | direct prepacked output axis `[Q,K,V]`; TP1 |
| `layers.L.linear_attn.in_proj_z.weight` | same | direct output axis |
| `layers.L.linear_attn.in_proj_b.weight` | same | direct value-head axis |
| `layers.L.linear_attn.in_proj_a.weight` | same | direct value-head axis |
| `layers.L.linear_attn.conv1d.weight` | same | direct channel axis `[Q,K,V]`; loader accepts checkpoint 2-D and runtime view is depthwise 3-D |
| `layers.L.linear_attn.A_log` | same | direct/sharded head axis; retain FP32 |
| `layers.L.linear_attn.dt_bias` | same | direct/sharded head axis |
| `layers.L.linear_attn.norm.weight` | same | direct value-head-dimension vector |
| `layers.L.linear_attn.out_proj.weight` | same | direct row-parallel input axis; TP1 |
| `layers.L.self_attn.q_proj.weight` | `layers.L.qkv_proj.weight`, shard `"q"` | output-axis pack; q tensor already contains query and output-gate channels |
| `layers.L.self_attn.k_proj.weight` | `layers.L.qkv_proj.weight`, shard `"k"` | output-axis pack |
| `layers.L.self_attn.v_proj.weight` | `layers.L.qkv_proj.weight`, shard `"v"` | output-axis pack |
| `layers.L.self_attn.q_norm.weight` | `layers.L.q_norm.weight` | remove `.self_attn` namespace after layer construction prefix normalization |
| `layers.L.self_attn.k_norm.weight` | `layers.L.k_norm.weight` | same |
| `layers.L.self_attn.o_proj.weight` | `layers.L.o_proj.weight` | same |
| `layers.L.self_attn.indexer.index_qk_proj.weight` | `layers.L.indexer.index_qk_proj.weight` | remove `.self_attn`; preserve donor projection's packed index-query/index-key output axis |
| `layers.L.self_attn.indexer.q_layernorm.weight` | `layers.L.indexer.q_layernorm.weight` | direct after namespace normalization |
| `layers.L.self_attn.indexer.k_layernorm.weight` | `layers.L.indexer.k_layernorm.weight` | direct after namespace normalization |
| `layers.L.mlp.gate.weight` | same | direct router output axis over 512 experts |
| `layers.L.mlp.shared_expert.gate_proj.weight` | runtime shared expert `gate_up_proj`, shard `0` | output-axis stack |
| `layers.L.mlp.shared_expert.up_proj.weight` | runtime shared expert `gate_up_proj`, shard `1` | output-axis stack |
| `layers.L.mlp.shared_expert.down_proj.weight` | same | direct |
| `layers.L.mlp.shared_expert_gate.weight` | same | direct |
| `layers.L.mlp.experts.gate_up_proj` | `layers.L.mlp.experts.w13_weight` | checkpoint shape `[expert, 2*intermediate, hidden]`; split `dim=-2`, load first half as `w1`, second as `w3`, once per expert |
| `layers.L.mlp.experts.down_proj` | `layers.L.mlp.experts.w2_weight` | checkpoint shape `[expert, hidden, intermediate]`; load as `w2`, once per expert |
| `layers.1.ple.key_proj.weight` | same | direct output axis `4H` |
| `layers.1.ple.value_proj.weight` | same | direct output axis `H` |
| `layers.1.ple.{norm_key,norm_query,norm_conv}.weight` | same | direct grouped-norm vectors |
| `layers.1.ple.conv1d.weight` | same | direct depthwise channel axis `4H` |
| `layers.1.ple.ple_embedding.layer_multipliers` | matching registered buffer | exact shape copy |
| `layers.1.ple.ple_embedding.ngram_heads_offsets` | matching registered buffer | exact shape copy |
| `layers.1.ple.ple_embedding.ngram_heads_vocab_sizes` | matching registered buffer | exact shape copy |
| `layers.1.ple.ple_embedding.ngram_embedding.shard_N.weight` | one logical `ngram_embedding.weight` | `N=0..127`; row range uses `ceil(org_vocab_size/128)`, clip final shard, copy overlap into local TP vocabulary rows; TP1 covers all rows |

The donor's additional mappings from split GDN projections into `in_proj_qkvz`/`in_proj_ba` are for its newer fused Qwen3.5 implementation. f02 owns split `in_proj_qkv`, `in_proj_z`, `in_proj_b`, and `in_proj_a`, exactly matching this checkpoint, so those donor fusion mappings must not be applied.

The loader should record canonical runtime names after mapping and compare them with all expected parameters/buffers. The observed non-visual, non-MTP census is: 48 pairs of hyperconnection blocks, 36 GDN parameter sets, 12 attention/QSA sets, 48 MoE sets, one PLE set with 128 embedding shards, one embedding, one final mixer, and one LM head. There is no `model.norm` tensor.

## 7. Ordered implementation and qualification sequence

1. Port and unit-test Qwen4 config parsing. Assert the exact 48-layer pattern, PLE layer index conversion, mamba/GDN cache shapes, and `language_model_only` startup override for this profile.
2. Implement the text wrapper and empty decoder skeleton; verify registry resolution by exact architecture name and assert that no vision module is constructed.
3. Implement the Qwen4 hyperconnection layer wrapper and final mixer. Load only embeddings, GR weights, and LM head in a tiny layer fixture; verify no final norm is created.
4. Add the MoE-compatible decoder construction using the shallow config copy. Verify all inherited expert modules retain Qwen4 dimensions and the original shared config remains `qwen4_exp_text`.
5. Integrate native GDN first, then the explicit GDN `REFERENCE_PATH`. For the reference path, compare one full layer prefill plus tokenwise decode, pool states, reset/reuse, and output projection against an independent equation.
6. Integrate synchronous PLE with request-owned N-gram and convolution state. Verify that it executes before attention GR mixing on layer 1 and commits history once per model forward.
7. Integrate reference QSA with ordinary K/V pool writes, request-owned compressed-key state, logical-to-physical mapping, and per-query causal visibility. Compare full-layer prefill and decode with dense gather over the selected tokens.
8. Make the weight loader strict and produce a receipt: source key, normalized key, destination parameter/buffer, shard selector, source/destination shapes, dtype, and content hash. Admission requires zero unexpected non-skipped keys and zero missing destination tensors.
9. Run a 48-layer real-weight eager prefill at TP1/batch1 with radix/spec/MTP/graph disabled, followed by tokenwise decode on the same request and a fresh-request reset comparison. Only after this passes should native/reference component substitutions be compared.
10. Treat radix sharing, batch>1, TP>1, graph capture, speculative/MTP, multimodal input, PLE offload/prefetch, and native sparse QSA as later profiles with separate lifecycle and numerical evidence. They are not implied by the first-profile pass.

## 8. Failure guards required in the first implementation

Startup or forward must fail explicitly when any of these is observed: image/video inputs, TP other than 1, batch other than 1, graph capture, radix prefix reuse, target-verify/draft modes, MTP weights requested, multiple packed request rows, a QSA logical position without a live physical token slot, request slot 0 used as real state, non-FP32 GDN recurrent storage, PLE state with a mismatched request slot, an unknown checkpoint tensor, or a missing expected tensor.

These guards keep a bounded successful run from being reported as qualification of unsupported KTEP modes. They do not prevent the eager GDN `REFERENCE_PATH` from being used as the production computation path inside its stated profile once its full-model and state-lifecycle evidence passes.
