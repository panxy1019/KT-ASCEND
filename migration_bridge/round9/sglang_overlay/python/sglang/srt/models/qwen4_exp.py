"""Bounded Qwen4-Exp text model for the recovered f02 SGLang runtime.

Profile: eager text, TP1, batch1, graph/radix/speculation/MTP off.  KTEP owns
request scheduling, KV/state pools, and the native FusedMoE modules.  Stateful
reference adapters fail closed until the pool APIs listed in
``ROUND9_REQUIRED_POOL_APIS`` are installed by the runtime overlay.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Protocol, Set, Tuple, Union

import torch
from torch import nn

from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig
from sglang.srt.distributed import get_pp_group
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.round9.gdn import GDNConfig, GDNState, eager_gdn
from sglang.srt.layers.round9.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)
from sglang.srt.layers.round9.ple_hash import NGramHasher
from sglang.srt.layers.round9.ple_sync import SynchronousPLE
from sglang.srt.layers.round9.qsa import select_tokens, sparse_gqa, text_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5LinearDecoderLayer,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers


ROUND9_REQUIRED_POOL_APIS = {
    "request_state": (
        "get_mamba_indices(req_pool_indices)",
        "mamba2_layer_cache(layer_id) -> .conv[0], .temporal",
        "short_conv_layer_cache(layer_id)",
        "get_ngram_context(state_indices)",
        "set_ngram_context(state_indices, context)",
    ),
    "qsa_kv": (
        "set_kv_buffer(attn_layer, out_cache_loc, key, value)",
        "get_kv_buffer(layer_id)",
        "get_qsa_key_state_buffer(layer_id)",
        "set_qsa_key_state_buffer(layer_id, locations, values)",
        "set_qsa_rope_position_buffer(locations, positions)",
        "get_qsa_rope_position_buffer(locations)",
        "reset_qsa_request(layer_id, request_slot)",
        "get_qsa_compressed_k_buffer(layer_id)",
        "set_qsa_compressed_k_buffer(layer_id, locations, values)",
        "qsa_compress_ratio == 4",
    ),
}


class OriginalPLETable(Protocol):
    """Immutable original-checkpoint table installed by the model runner."""

    def gather_rows(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Return BF16 rows with shape ``[*row_ids.shape, head_dim]``."""


def _missing_pool(owner, names: Tuple[str, ...], component: str) -> None:
    missing = [name for name in names if not hasattr(owner, name)]
    if missing:
        raise RuntimeError(
            f"Round9 {component} requires pool APIs not installed in f02: {missing}"
        )


def _compat_config(config: Qwen4ExpTextConfig) -> Qwen4ExpTextConfig:
    compat = copy.copy(config)
    compat.model_type = "qwen3_5_moe_text"
    return compat


def _speculation_is_active(spec_algorithm) -> bool:
    """Interpret f02's SpeculativeAlgorithm.NONE without importing the enum."""
    if spec_algorithm is None:
        return False
    is_none = getattr(spec_algorithm, "is_none", None)
    if not callable(is_none):
        return True
    return not bool(is_none())


def _mean_in_storage_dtype(values: torch.Tensor) -> torch.Tensor:
    """Accumulate a QSA compression group in FP32, then restore its dtype."""
    if not values.is_floating_point():
        raise TypeError("QSA compression values must be floating point")
    return values.float().mean(dim=0).to(dtype=values.dtype)


def _flatten_token_kv(
    buffer: torch.Tensor, num_kv_heads: int, head_dim: int
) -> torch.Tensor:
    """Expose token slots for both flat and NPU page-shaped KV buffers."""
    if buffer.ndim < 3 or tuple(buffer.shape[-2:]) != (
        num_kv_heads,
        head_dim,
    ):
        raise RuntimeError(
            "unexpected ordinary KV buffer shape "
            f"{tuple(buffer.shape)} for ({num_kv_heads}, {head_dim})"
        )
    return buffer.reshape(-1, num_kv_heads, head_dim)


def _qsa_rope_settings(config, index_head_dim: int) -> Tuple[int, float]:
    rope = getattr(config, "rope_parameters", None) or getattr(
        config, "rope_scaling", {}
    )
    partial = rope.get(
        "partial_rotary_factor",
        getattr(config, "partial_rotary_factor", 1.0),
    )
    attention_head_dim = getattr(config, "head_dim", None) or (
        int(config.hidden_size) // int(config.num_attention_heads)
    )
    # Donor build_qsa_indexer receives self.rotary_emb from the main attention
    # module. Its rotary width is therefore based on the 256-wide attention
    # head, while the projected index head itself is 128 wide.
    rotary_dim = int(attention_head_dim * partial)
    theta = float(rope.get("rope_theta", getattr(config, "rope_theta", 10000.0)))
    if rotary_dim < 2 or rotary_dim > index_head_dim or rotary_dim % 2:
        raise ValueError(f"invalid QSA rotary dimension {rotary_dim}")
    return rotary_dim, theta


def _guard_profile(forward_batch: ForwardBatch) -> None:
    if get_attention_tp_size() != 1:
        raise RuntimeError("Round9 reference model requires TP=1")
    if get_is_capture_mode():
        raise RuntimeError("Round9 reference model does not support graph capture")
    if not get_global_server_args().disable_radix_cache:
        raise RuntimeError("Round9 reference model requires --disable-radix-cache")
    if int(forward_batch.batch_size) != 1:
        raise RuntimeError("Round9 reference model requires batch_size=1")
    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_extend()):
        raise RuntimeError(f"Round9 reference model rejects forward mode {mode!r}")
    if forward_batch.spec_info is not None or _speculation_is_active(
        getattr(forward_batch, "spec_algorithm", None)
    ):
        raise RuntimeError("Round9 reference model requires speculation/MTP OFF")
    if forward_batch.contains_image_inputs():
        raise RuntimeError("Round9 reference model is text-only")


class Qwen4ReferenceGatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Whole-module GDN reference; includes gated norm exactly once."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_reference_path()

    def _enable_reference_path(self):
        # f02 creates A_log using the ambient model dtype. The Round9 GDN
        # contract retains this decay parameter in FP32 before loading.
        self.A_log.data = self.A_log.data.float()
        self.reference_config = GDNConfig(
            num_key_heads=self.num_k_heads,
            num_value_heads=self.num_v_heads,
            key_head_dim=self.head_k_dim,
            value_head_dim=self.head_v_dim,
            conv_kernel_size=self.conv_kernel_size,
            eps=self.layer_norm_epsilon,
        )

    @classmethod
    def convert(cls, module: Qwen3_5GatedDeltaNet):
        """Reuse the parent's already-allocated projections and parameters."""
        if type(module) is not Qwen3_5GatedDeltaNet:
            raise TypeError("only an exact f02 Qwen3_5GatedDeltaNet can be converted")
        module.__class__ = cls
        module._enable_reference_path()
        return module

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        _guard_profile(forward_batch)
        req_pool = forward_batch.req_to_token_pool
        _missing_pool(
            req_pool,
            ("get_mamba_indices", "mamba2_layer_cache"),
            "GDN",
        )
        state_indices = req_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        ).long()
        if state_indices.numel() != 1 or int(state_indices[0]) == 0:
            raise RuntimeError("Round9 GDN requires one non-padding mamba slot")
        layer_cache = req_pool.mamba2_layer_cache(self.layer_id)
        conv_pool = layer_cache.conv[0]
        recurrent_pool = layer_cache.temporal
        slot = state_indices[0]

        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        a, _ = self.in_proj_a(hidden_states)
        b, _ = self.in_proj_b(hidden_states)
        z = z.reshape(-1, self.num_v_heads, self.head_v_dim)
        conv_state = conv_pool[slot]
        recurrent_state = recurrent_pool[slot]
        if recurrent_state.dtype != torch.float32:
            raise RuntimeError("Round9 GDN recurrent pool must be FP32")
        state = GDNState(conv_state=conv_state, recurrent_state=recurrent_state)
        normalized, next_state = eager_gdn(
            mixed_qkv,
            a,
            b,
            z,
            self.conv1d.weight,
            self.A_log,
            self.dt_bias,
            self.norm.weight,
            self.reference_config,
            initial_state=state,
        )
        conv_pool[slot].copy_(next_state.conv_state)
        recurrent_pool[slot].copy_(next_state.recurrent_state)
        output, _ = self.out_proj(normalized.flatten(-2))
        return output


class Qwen4ReferenceIndexer(nn.Module):
    """Text-only compressed-QSA index projection and pool update."""

    def __init__(self, config, layer_id, quant_config=None, prefix=""):
        super().__init__()
        self.layer_id = int(layer_id)
        self.n_heads = int(config.indexer_n_heads)
        self.kv_heads = int(config.indexer_kv_heads)
        self.head_dim = int(config.indexer_head_dim)
        self.ratio = int(config.indexer_compress_ratio)
        self.token_topk = int(config.indexer_budget)
        if self.kv_heads != 1 or self.ratio != 4:
            raise ValueError("Round9 QSA requires one index KV head and ratio=4")
        if self.token_topk % self.ratio:
            raise ValueError("QSA token budget must divide into compression groups")
        self.index_qk_proj = ReplicatedLinear(
            config.hidden_size,
            (self.n_heads + self.kv_heads) * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("index_qk_proj", prefix),
        )
        self.q_layernorm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_dim, self.rope_theta = _qsa_rope_settings(
            config, self.head_dim
        )

    def _rope(self, x, positions):
        return text_rope(
            x,
            positions,
            rotary_dim=self.rotary_dim,
            theta=self.rope_theta,
        )

    def forward(self, hidden_states, positions, forward_batch):
        pool = forward_batch.token_to_kv_pool
        _missing_pool(
            pool,
            (
                "get_qsa_key_state_buffer",
                "set_qsa_key_state_buffer",
                "set_qsa_rope_position_buffer",
                "get_qsa_rope_position_buffer",
                "get_qsa_compressed_k_buffer",
                "set_qsa_compressed_k_buffer",
                "reset_qsa_request",
            ),
            "QSA",
        )
        if int(getattr(pool, "qsa_compress_ratio", -1)) != self.ratio:
            raise RuntimeError("QSA pool compression ratio does not match the model")
        req_slot = forward_batch.req_pool_indices.long()[0]
        logical = positions.long()
        if logical.ndim != 1:
            raise RuntimeError("Round9 QSA reference accepts one-axis text positions")
        if logical.numel() == 0:
            raise RuntimeError("Round9 QSA reference rejects an empty active forward")
        if int(logical[0]) == 0:
            pool.reset_qsa_request(self.layer_id, int(req_slot))
        qk, _ = self.index_qk_proj(hidden_states)
        q_width = self.n_heads * self.head_dim
        query = qk[:, :q_width].reshape(-1, self.n_heads, self.head_dim)
        raw_key = qk[:, q_width:].reshape(-1, self.head_dim)
        query = self._rope(
            self.q_layernorm(query.reshape(-1, self.head_dim)).reshape_as(query),
            logical,
        )

        ring_locs = req_slot * self.ratio + logical.remainder(self.ratio)
        ring = pool.get_qsa_key_state_buffer(self.layer_id)
        for row in range(logical.numel()):
            loc = ring_locs[row : row + 1]
            pool.set_qsa_key_state_buffer(
                self.layer_id, loc, raw_key[row : row + 1, None, :]
            )
            pool.set_qsa_rope_position_buffer(loc, logical[row : row + 1])
            if int(logical[row] + 1) % self.ratio:
                continue
            end = int(logical[row])
            member_locs = req_slot * self.ratio + torch.arange(
                self.ratio, device=logical.device
            )
            member_positions = pool.get_qsa_rope_position_buffer(member_locs)
            expected_positions = torch.arange(
                end - self.ratio + 1,
                end + 1,
                device=member_positions.device,
                dtype=member_positions.dtype,
            )
            member_positions = member_positions.reshape(-1)
            if member_positions.shape != expected_positions.shape or not torch.equal(
                member_positions, expected_positions
            ):
                raise RuntimeError("QSA compression ring has stale or missing positions")
            pooled = _mean_in_storage_dtype(ring.index_select(0, member_locs))
            pooled = self.k_layernorm(pooled).reshape(1, self.head_dim)
            pooled = self._rope(
                pooled, logical.new_tensor([end - self.ratio + 1])
            )
            first_token_slot = forward_batch.req_to_token_pool.req_to_token[
                req_slot, end - self.ratio + 1
            ].long()
            if int(first_token_slot) <= 0 or int(first_token_slot) % self.ratio:
                raise RuntimeError("QSA compression group is not KV-page aligned")
            pool.set_qsa_compressed_k_buffer(
                self.layer_id,
                (first_token_slot // self.ratio).reshape(1),
                pooled[:, None, :],
            )

        completed = int((logical[-1] + 1) // self.ratio)
        compressed = pool.get_qsa_compressed_k_buffer(self.layer_id)
        if completed:
            starts = torch.arange(completed, device=logical.device) * self.ratio
            physical = forward_batch.req_to_token_pool.req_to_token[
                req_slot, starts
            ].long()
            if bool((physical <= 0).any()) or bool(
                physical.remainder(self.ratio).ne(0).any()
            ):
                raise RuntimeError("QSA compressed logical groups lack aligned KV slots")
            compressed_keys = compressed.index_select(0, physical // self.ratio)
            compressed_keys = compressed_keys.reshape(completed, self.head_dim)
        else:
            compressed_keys = raw_key.new_empty((0, self.head_dim))
        selected = select_tokens(
            query,
            compressed_keys,
            logical,
            ratio=self.ratio,
            topk_blocks=self.token_topk // self.ratio,
        )
        return query, selected


class Qwen4ReferencePLE(SynchronousPLE):
    """Original-table PLE with request-owned history and convolution state."""

    def __init__(self, config, layer_id):
        super().__init__(
            hidden_size=config.hidden_size,
            embedding_size=config.ple_embed_dim,
            kernel_size=config.ple_conv_kernel_size,
            dilation=config.ngram_size,
            dtype=getattr(config, "torch_dtype", torch.bfloat16),
        )
        self.layer_id = int(layer_id)
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.head_dim = config.ple_embed_dim // self.ngram_heads
        self.vocab_size = int(config.vocab_size)
        self.eos_token_id = int(config.eos_token_id)
        self.register_buffer("layer_multipliers", torch.zeros(self.ngram_size, dtype=torch.long))
        self.register_buffer("ngram_heads_offsets", torch.zeros(self.ngram_heads, dtype=torch.long))
        self.register_buffer("ngram_heads_vocab_sizes", torch.zeros(self.ngram_heads, dtype=torch.long))
        self._table: Optional[OriginalPLETable] = None

    def bind_original_table(self, table: OriginalPLETable) -> None:
        self._table = table

    def prepare(self, input_ids, forward_batch):
        if self._table is None:
            raise RuntimeError(
                "Round9 PLE original checkpoint table is not bound; install an "
                "OriginalPLETable provider before forward"
            )
        req_pool = forward_batch.req_to_token_pool
        _missing_pool(
            req_pool,
            ("get_mamba_indices", "get_ngram_context"),
            "PLE",
        )
        slot = req_pool.get_mamba_indices(forward_batch.req_pool_indices).long()
        if slot.numel() != 1 or int(slot[0]) == 0:
            raise RuntimeError("Round9 PLE requires one non-padding mamba slot")
        history = req_pool.get_ngram_context(slot)[0].tolist()
        hasher = NGramHasher(
            ngram_size=self.ngram_size,
            heads_per_ngram=self.heads_per_ngram,
            vocab_size=self.vocab_size,
            eos_token_id=self.eos_token_id,
            multipliers=self.layer_multipliers.tolist(),
            head_sizes=self.ngram_heads_vocab_sizes.tolist(),
            head_offsets=self.ngram_heads_offsets.tolist(),
        )
        rows, next_history = hasher.hash_tokens(input_ids.long().tolist(), history)
        row_ids = torch.tensor(rows, dtype=torch.long, device=input_ids.device)
        embeddings = self._table.gather_rows(row_ids)
        expected = (*row_ids.shape, self.head_dim)
        if tuple(embeddings.shape) != expected:
            raise RuntimeError(
                f"PLE table returned {tuple(embeddings.shape)}, expected {expected}"
            )
        return slot, embeddings.flatten(1), torch.tensor(
            next_history, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)

    def run(self, hidden_states, embeddings, state_slot, forward_batch):
        req_pool = forward_batch.req_to_token_pool
        _missing_pool(req_pool, ("short_conv_layer_cache",), "PLE")
        state_pool = req_pool.short_conv_layer_cache(self.layer_id)
        state = state_pool[state_slot[0]]
        output, next_state = super().forward(hidden_states, embeddings, state)
        state_pool[state_slot[0]].copy_(next_state)
        return output


class Qwen4ExpLayerExtensionMixin:
    def _init_qwen4_extensions(self, config, layer_id):
        for name in ("input_layernorm", "post_attention_layernorm", "layer_communicator"):
            if hasattr(self, name):
                delattr(self, name)
        hc = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidual(hc, use_mix=True, use_combine=True)
        self.mlp_hyper_connection = GatedResidual(hc, use_mix=True, use_combine=True)
        self.ple = (
            Qwen4ReferencePLE(config, layer_id)
            if layer_id + 1 in set(config.ple_layer_ids)
            else None
        )

    def _before_attention(self, hidden, residual, forward_batch, ple_data):
        width = self.config.hc_count * self.config.hidden_size
        if hidden.shape[-1] != width:
            if hidden.shape[-1] != self.config.hidden_size:
                raise RuntimeError("invalid Qwen4 hyperconnection width")
            hidden = torch.cat([hidden] * self.config.hc_count, dim=-1)
        if self.ple is not None:
            if ple_data is None:
                raise RuntimeError("Qwen4 PLE layer is missing prepared table rows")
            slot, embeddings, _ = ple_data
            query = hidden if residual is None else hidden + residual
            hidden = hidden + self.ple.run(query, embeddings, slot, forward_batch)
        return self.attn_hyper_connection.mix(hidden)

    def _after_attention(self, hidden, residual, forward_batch):
        hidden = self.attn_hyper_connection.combine(hidden, residual)
        hidden, residual = self.mlp_hyper_connection.mix(hidden)
        hidden = self.mlp(hidden, forward_batch, False)
        hidden = self.mlp_hyper_connection.combine(hidden, residual)
        return hidden, None


class Qwen4ExpLinearDecoderLayer(
    Qwen4ExpLayerExtensionMixin, Qwen3_5LinearDecoderLayer
):
    def __init__(self, config, layer_id, quant_config=None, prefix="", alt_stream=None):
        compat = _compat_config(config)
        super().__init__(compat, layer_id, quant_config, prefix, alt_stream)
        self.config = config
        self.linear_attn = Qwen4ReferenceGatedDeltaNet.convert(self.linear_attn)
        self._init_qwen4_extensions(config, layer_id)

    def forward(self, hidden_states, residual, **kwargs):
        forward_batch = kwargs["forward_batch"]
        hidden_states, residual = self._before_attention(
            hidden_states, residual, forward_batch, kwargs.get("ple_data")
        )
        hidden_states = self.linear_attn(hidden_states, forward_batch)
        return self._after_attention(hidden_states, residual, forward_batch)


class Qwen4ExpAttentionDecoderLayer(
    Qwen4ExpLayerExtensionMixin, Qwen3_5AttentionDecoderLayer
):
    def __init__(self, config, layer_id, quant_config=None, prefix="", alt_stream=None):
        compat = _compat_config(config)
        compat.attn_output_gate = True
        super().__init__(compat, layer_id, quant_config, prefix, alt_stream)
        self.config = config
        self.indexer = Qwen4ReferenceIndexer(
            config,
            layer_id,
            quant_config,
            add_prefix("indexer", prefix),
        )
        self._init_qwen4_extensions(config, layer_id)

    def self_attention(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        q_gate, k, v = qkv.split(
            [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
        )
        q, gate = q_gate.reshape(-1, self.num_heads, self.head_dim * 2).chunk(2, -1)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        q_flat, k_flat = self._apply_qk_norm(q.flatten(-2), k.flatten(-2))
        q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)
        q = q_flat.reshape(-1, self.num_heads, self.head_dim)
        k = k_flat.reshape(-1, self.num_kv_heads, self.head_dim)

        pool = forward_batch.token_to_kv_pool
        _missing_pool(
            pool,
            ("set_kv_buffer", "get_key_buffer", "get_value_buffer"),
            "ordinary QSA KV",
        )
        pool.set_kv_buffer(self.attn, forward_batch.out_cache_loc, k, v)
        _, selected = self.indexer(hidden_states, positions, forward_batch)
        req_slot = forward_batch.req_pool_indices.long()[0]
        logical_ids = selected[selected >= 0]
        max_logical = int(positions.max())
        if logical_ids.numel() and int(logical_ids.max()) > max_logical:
            raise RuntimeError("QSA selected a future logical token")
        logical_slots = forward_batch.req_to_token_pool.req_to_token[
            req_slot, : max_logical + 1
        ].long()
        if bool((logical_slots <= 0).any()):
            raise RuntimeError("ordinary QSA KV map contains an unallocated slot")
        k_pool = _flatten_token_kv(
            pool.get_key_buffer(self.attn.layer_id),
            self.num_kv_heads,
            self.head_dim,
        )
        v_pool = _flatten_token_kv(
            pool.get_value_buffer(self.attn.layer_id),
            self.num_kv_heads,
            self.head_dim,
        )
        keys = k_pool.index_select(0, logical_slots)
        values = v_pool.index_select(0, logical_slots)
        attn_output = sparse_gqa(q, keys, values, selected)
        attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output.flatten(-2))
        return output

    def forward(self, positions, hidden_states, residual, forward_batch, **kwargs):
        hidden_states, residual = self._before_attention(
            hidden_states, residual, forward_batch, kwargs.get("ple_data")
        )
        hidden_states = self.self_attention(positions, hidden_states, forward_batch)
        return self._after_attention(hidden_states, residual, forward_batch)


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen4ExpAttentionDecoderLayer,
    "full_attention": Qwen4ExpAttentionDecoderLayer,
    "linear_attention": Qwen4ExpLinearDecoderLayer,
}


class Qwen4ExpModel(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.pp_group = get_pp_group()
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        if self.pp_group.world_size != 1:
            raise RuntimeError("Round9 reference model requires pipeline size 1")
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            enable_tp=False,
        )

        def build(idx, prefix):
            block = config.layers_block_type[idx]
            cls = ALL_DECODER_LAYER_TYPES[block]
            suffix = "self_attn" if block in ("attention", "full_attention") else "linear_attn"
            return cls(
                config,
                idx,
                quant_config,
                add_prefix(suffix, prefix),
                None,
            )

        self.layers = make_layers(
            config.num_hidden_layers, build, prefix=add_prefix("layers", prefix)
        )
        hc = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(hc, use_combine=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def bind_original_ple_table(self, table: OriginalPLETable) -> None:
        found = 0
        for layer in self.layers:
            if getattr(layer, "ple", None) is not None:
                layer.ple.bind_original_table(table)
                found += 1
        if found != len(set(self.config.ple_layer_ids)):
            raise RuntimeError("PLE table binding did not find every configured layer")

    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        del pp_proxy_tensors
        _guard_profile(forward_batch)
        hidden = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        residual = None
        ple_data = None
        ple_layer = next(
            (layer.ple for layer in self.layers if getattr(layer, "ple", None) is not None),
            None,
        )
        if ple_layer is not None:
            ple_data = ple_layer.prepare(input_ids, forward_batch)
        for i, layer in enumerate(self.layers):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden, residual = layer(
                    positions=positions,
                    hidden_states=hidden,
                    residual=residual,
                    forward_batch=forward_batch,
                    ple_data=ple_data,
                )
        if residual is not None:
            raise RuntimeError("final Qwen4 hyperconnection residual must be None")
        hidden, _ = self.hyper_connection_mixer.mix(hidden)
        if ple_data is not None:
            req_pool = forward_batch.req_to_token_pool
            _missing_pool(req_pool, ("set_ngram_context",), "PLE")
            req_pool.set_ngram_context(ple_data[0], ple_data[2])
        return hidden


@dataclass
class WeightLoadAudit:
    loaded: Set[str] = field(default_factory=set)
    seen_sources: Set[str] = field(default_factory=set)
    skipped_visual: Set[str] = field(default_factory=set)
    skipped_mtp: Set[str] = field(default_factory=set)
    ple_shards: Set[int] = field(default_factory=set)
    ple_shard_rows: int = 0
    ple_shard_row_counts: Dict[int, int] = field(default_factory=dict)
    packed_parts: Dict[str, Set[Union[str, int]]] = field(default_factory=dict)


_PLE_BUFFER_RENAMES = {
    ".ple.ple_embedding.layer_multipliers": ".ple.layer_multipliers",
    ".ple.ple_embedding.ngram_heads_offsets": ".ple.ngram_heads_offsets",
    ".ple.ple_embedding.ngram_heads_vocab_sizes": ".ple.ngram_heads_vocab_sizes",
}
_PLE_BUFFER_SUFFIXES = tuple(_PLE_BUFFER_RENAMES.values())


def _normalize_checkpoint_name(source_name: str) -> str:
    name = source_name
    prefix = "model.language_model."
    if name.startswith(prefix):
        name = "model." + name[len(prefix) :]
    for source, destination in _PLE_BUFFER_RENAMES.items():
        name = name.replace(source, destination)
    return name.replace(".self_attn.", ".")


def _ple_shard_index(name: str, expected_layer: Optional[int] = None) -> Optional[int]:
    marker = ".ple.ple_embedding.ngram_embedding.shard_"
    if marker not in name:
        return None
    match = re.fullmatch(
        r"model\.language_model\.layers\.(\d+)\.ple\.ple_embedding\."
        r"ngram_embedding\.shard_(\d+)\.weight",
        name,
    )
    if match is None:
        raise KeyError(f"malformed PLE shard key: {name}")
    layer = int(match.group(1))
    if expected_layer is not None and layer != expected_layer:
        raise KeyError(
            f"PLE shard belongs to layer {layer}, expected {expected_layer}: {name}"
        )
    return int(match.group(2))


def _validate_packed_parts(
    packed_parts: Dict[str, Set[Union[str, int]]]
) -> None:
    for destination, parts in packed_parts.items():
        required: Set[Union[str, int]] = (
            {"q", "k", "v"}
            if destination.endswith(".qkv_proj.weight")
            else {0, 1}
        )
        if parts != required:
            raise RuntimeError(
                f"packed parameter {destination} has components "
                f"{sorted(map(str, parts))}, expected {sorted(map(str, required))}"
            )


class Qwen4ExpForConditionalGeneration(nn.Module):
    fall_back_to_pt_during_load = False
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, config: Qwen4ExpConfig, quant_config=None, prefix=""):
        super().__init__()
        self.outer_config = config
        self.config = config.text_config
        self.pp_group = get_pp_group()
        self.model = Qwen4ExpModel(
            self.config, quant_config, add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=quant_config,
            use_attn_tp_group=False,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(self.config)
        self.weight_load_audit: Optional[WeightLoadAudit] = None

    def bind_original_ple_table(self, table: OriginalPLETable) -> None:
        self.model.bind_original_ple_table(table)

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        if get_embedding:
            raise RuntimeError("Round9 reference model does not expose embeddings")
        hidden = self.model(
            input_ids,
            positions,
            forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return self.logits_processor(
            input_ids, hidden, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters(remove_duplicate=False))
        buffers = dict(self.named_buffers())
        audit = WeightLoadAudit()
        ple_layers = sorted({int(layer) - 1 for layer in self.config.ple_layer_ids})
        if len(ple_layers) != 1:
            raise RuntimeError(
                "Round9 original-table loader requires exactly one configured PLE layer"
            )
        ple_layer = ple_layers[0]
        stacked = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        for source_name, tensor in weights:
            if source_name in audit.seen_sources:
                raise ValueError(f"duplicate checkpoint tensor: {source_name}")
            audit.seen_sources.add(source_name)
            if source_name.startswith("model.visual."):
                audit.skipped_visual.add(source_name)
                continue
            if source_name.startswith("mtp."):
                audit.skipped_mtp.add(source_name)
                continue
            if "rotary_emb.inv_freq" in source_name:
                continue
            shard = _ple_shard_index(source_name, ple_layer)
            if shard is not None:
                if shard in audit.ple_shards:
                    raise ValueError(f"duplicate PLE shard {shard}")
                if tensor.ndim != 2 or tensor.shape[1] != self.config.ple_embed_dim // (
                    (self.config.ngram_size - 1) * self.config.heads_per_ngram
                ):
                    raise ValueError(f"invalid PLE shard shape for {source_name}")
                audit.ple_shards.add(shard)
                audit.ple_shard_rows += int(tensor.shape[0])
                audit.ple_shard_row_counts[shard] = int(tensor.shape[0])
                continue
            name = _normalize_checkpoint_name(source_name)

            mapped = False
            for dst, src, shard_id in stacked:
                if src not in name or "mlp.experts" in name:
                    continue
                candidate = name.replace(src, dst)
                if candidate not in params:
                    continue
                loader = getattr(params[candidate], "weight_loader", None)
                if loader is None:
                    raise RuntimeError(f"packed parameter lacks weight_loader: {candidate}")
                parts = audit.packed_parts.setdefault(candidate, set())
                if shard_id in parts:
                    raise ValueError(
                        f"duplicate packed component {shard_id!r} for {candidate}"
                    )
                loader(params[candidate], tensor, shard_id)
                parts.add(shard_id)
                audit.loaded.add(candidate)
                mapped = True
                break
            if mapped:
                continue

            if "mlp.experts.gate_up_proj" in name:
                candidate = name.replace("experts.gate_up_proj", "experts.w13_weight")
                if candidate not in params:
                    raise KeyError(f"missing fused expert destination {candidate}")
                if candidate in audit.loaded:
                    raise ValueError(f"duplicate destination tensor: {candidate}")
                if tensor.ndim < 3 or tensor.shape[0] != self.config.num_experts:
                    raise ValueError(f"invalid fused expert shape for {source_name}")
                if tensor.shape[-2] % 2:
                    raise ValueError(f"odd fused gate/up width for {source_name}")
                gate, up = tensor.chunk(2, dim=-2)
                loader = params[candidate].weight_loader
                for expert_id in range(self.config.num_experts):
                    loader(params[candidate], gate[expert_id], candidate, "w1", expert_id)
                    loader(params[candidate], up[expert_id], candidate, "w3", expert_id)
                audit.loaded.add(candidate)
                continue
            if "mlp.experts.down_proj" in name:
                candidate = name.replace("experts.down_proj", "experts.w2_weight")
                if candidate not in params:
                    raise KeyError(f"missing fused expert destination {candidate}")
                if candidate in audit.loaded:
                    raise ValueError(f"duplicate destination tensor: {candidate}")
                if tensor.ndim < 3 or tensor.shape[0] != self.config.num_experts:
                    raise ValueError(f"invalid fused expert shape for {source_name}")
                loader = params[candidate].weight_loader
                for expert_id in range(self.config.num_experts):
                    loader(params[candidate], tensor[expert_id], candidate, "w2", expert_id)
                audit.loaded.add(candidate)
                continue
            if name in buffers and name.endswith(_PLE_BUFFER_SUFFIXES):
                if name in audit.loaded:
                    raise ValueError(f"duplicate destination tensor: {name}")
                if buffers[name].shape != tensor.shape:
                    raise ValueError(f"buffer shape mismatch for {name}")
                buffers[name].copy_(
                    tensor.to(device=buffers[name].device, dtype=buffers[name].dtype)
                )
                audit.loaded.add(name)
                continue
            if name not in params:
                raise KeyError(f"unexpected non-skipped Qwen4 checkpoint tensor: {source_name}")
            if name in audit.loaded:
                raise ValueError(f"duplicate destination tensor: {name}")
            loader = getattr(params[name], "weight_loader", default_weight_loader)
            loader(params[name], tensor)
            audit.loaded.add(name)

        expected_shards = set(range(int(self.config.split_ngram_parts)))
        if audit.ple_shards != expected_shards:
            missing = sorted(expected_shards - audit.ple_shards)
            extra = sorted(audit.ple_shards - expected_shards)
            raise RuntimeError(
                f"PLE shard audit failed: missing={missing[:8]} extra={extra[:8]}"
            )
        _validate_packed_parts(audit.packed_parts)
        expected = set(dict(self.named_parameters()).keys())
        expected.update(
            name
            for name in buffers
            if name.endswith(
                (
                    ".ple.layer_multipliers",
                    ".ple.ngram_heads_offsets",
                    ".ple.ngram_heads_vocab_sizes",
                )
            )
        )
        missing = sorted(expected - audit.loaded)
        if missing:
            raise RuntimeError(f"Qwen4 checkpoint did not load parameters: {missing[:16]}")
        offset_buffers = [
            buffer
            for name, buffer in buffers.items()
            if name.endswith(".ple.ngram_heads_offsets")
        ]
        size_buffers = [
            buffer
            for name, buffer in buffers.items()
            if name.endswith(".ple.ngram_heads_vocab_sizes")
        ]
        if len(offset_buffers) != 1 or len(size_buffers) != 1:
            raise RuntimeError("expected exactly one loaded PLE hash-buffer set")
        logical_rows = int(offset_buffers[0][-1]) + int(size_buffers[0][-1])
        alignment = int(self.config.make_ngram_vocab_size_divisible_by)
        if alignment <= 0:
            raise ValueError("PLE vocabulary alignment must be positive")
        expected_ple_rows = ((logical_rows + alignment - 1) // alignment) * alignment
        if audit.ple_shard_rows != expected_ple_rows:
            raise RuntimeError(
                "PLE shard rows do not match loaded hash buffers: "
                f"{audit.ple_shard_rows} != {expected_ple_rows}"
            )
        shard_size = (expected_ple_rows + len(expected_shards) - 1) // len(expected_shards)
        for shard, actual_rows in audit.ple_shard_row_counts.items():
            expected_rows = max(0, min(shard_size, expected_ple_rows - shard * shard_size))
            if actual_rows != expected_rows:
                raise RuntimeError(f"PLE shard {shard} row boundary mismatch")
        self.weight_load_audit = audit
        return audit.loaded

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text = config.text_config
        return ModelConfigForExpertLocation(
            num_layers=text.num_hidden_layers,
            num_logical_experts=text.num_experts,
            num_groups=None,
        )


EntryClass = [Qwen4ExpForConditionalGeneration]
