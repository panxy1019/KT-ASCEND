"""Bounded eager QSA side buffers owned by SGLang's ordinary hybrid KV pool.

Ordinary K/V retain the native NPU pool layout. Compressed index keys use
physical token slot // 4; raw keys use request slot * 4 + logical position % 4.
The model resets each layer's ring at the start of a fresh request. Radix,
speculation, disaggregation and graph capture are outside this profile.
"""
import torch
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool


class Round9QSAKVPool(HybridLinearKVPool):
    def __init__(self, *, request_capacity, index_head_dim, compress_ratio=4, **kwargs):
        page = kwargs["page_size"]
        if compress_ratio != 4 or page < 4 or page % 4:
            raise ValueError("Round9 QSA requires page size divisible by four")
        if request_capacity < 1 or index_head_dim < 1 or kwargs.get("use_mla", False):
            raise ValueError("invalid Round9 QSA pool profile")
        if kwargs.get("enable_memory_saver", False):
            raise ValueError("Round9 QSA requires memory saver OFF")
        self.qsa_compress_ratio = compress_ratio
        self.request_capacity = request_capacity
        super().__init__(**kwargs)
        opts = dict(dtype=self.dtype, device=self.device)
        ring_size = request_capacity * compress_ratio
        group_size = (self.size + page) // compress_ratio
        self.qsa_raw_keys = torch.zeros(
            self.full_layer_nums, ring_size, 1, index_head_dim, **opts)
        self.qsa_compressed_keys = torch.zeros(
            self.full_layer_nums, group_size, 1, index_head_dim, **opts)
        self.qsa_rope_positions = torch.full(
            (ring_size,), -1, dtype=torch.long, device=self.device)
        self.qsa_bytes = sum(t.numel() * t.element_size() for t in (
            self.qsa_raw_keys, self.qsa_compressed_keys, self.qsa_rope_positions))
        self.mem_usage += self.qsa_bytes / (1024 ** 3)

    def get_qsa_key_state_buffer(self, layer_id):
        return self.qsa_raw_keys[self._transfer_full_attention_id(layer_id)]

    def set_qsa_key_state_buffer(self, layer_id, locations, values):
        self.get_qsa_key_state_buffer(layer_id)[locations.long()] = values

    def get_qsa_rope_position_buffer(self, locations):
        return self.qsa_rope_positions[locations.long()]

    def set_qsa_rope_position_buffer(self, locations, positions):
        self.qsa_rope_positions[locations.long()] = positions.long()

    def get_qsa_compressed_k_buffer(self, layer_id):
        return self.qsa_compressed_keys[self._transfer_full_attention_id(layer_id)]

    def set_qsa_compressed_k_buffer(self, layer_id, locations, values):
        self.get_qsa_compressed_k_buffer(layer_id)[locations.long()] = values

    def reset_qsa_request(self, layer_id, request_slot):
        if not 0 <= request_slot < self.request_capacity:
            raise ValueError("QSA request slot outside allocated pool")
        start = request_slot * self.qsa_compress_ratio
        stop = start + self.qsa_compress_ratio
        self.get_qsa_key_state_buffer(layer_id)[start:stop].zero_()
        self.qsa_rope_positions[start:stop].fill_(-1)

    def get_contiguous_buf_infos(self):
        raise RuntimeError("Round9 QSA does not support disaggregation/offload")
