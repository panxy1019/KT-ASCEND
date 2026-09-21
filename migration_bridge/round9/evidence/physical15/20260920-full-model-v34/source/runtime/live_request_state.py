"""Bounded eager proof that expert migration preserves a live request's state.

This diagnostic barrier is deliberately synchronous and is not a throughput
feature. It hashes only active token pages and the single request's recurrent
state, without copying, freeing, restoring or resetting those pools.
"""
import hashlib


def snapshot_live_request(runner, request):
    import torch
    if request.req_pool_idx is None or request.mamba_pool_idx is None or request.finished():
        raise RuntimeError('state snapshot requires a live allocated request')
    req_pool, kv_pool = runner.req_to_token_pool, runner.token_to_kv_pool
    slot, recurrent_slot = int(request.req_pool_idx), int(request.mamba_pool_idx)
    token_count = len(request.origin_input_ids)
    if not 0 < token_count <= 4096:
        raise RuntimeError('bounded prefill-state proof requires at most4096 inputs')
    mapping = req_pool.req_to_token[slot, :token_count].long()
    if len(set(mapping.detach().cpu().tolist())) != token_count:
        raise RuntimeError('live token mapping contains duplicate physical ownership')
    request_indices = torch.tensor([slot], dtype=torch.int64, device=mapping.device)
    actual_recurrent = req_pool.get_mamba_indices(request_indices).long()
    if actual_recurrent.numel() != 1 or int(actual_recurrent[0]) != recurrent_slot or recurrent_slot <= 0:
        raise RuntimeError('request recurrent allocation mapping differs')
    tensors = {}
    def add(name, value, owner=None):
        raw = value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
        tensors[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
            owner_pointer=int((value if owner is None else owner).data_ptr()),
            sha256=hashlib.sha256(raw).hexdigest())
    add('token_mapping', mapping, req_pool.req_to_token)
    add('ngram_history', req_pool.get_ngram_context(actual_recurrent), req_pool.ngram_pool.context)
    seen = {'gdn': set(), 'qsa': set(), 'ple': set()}
    for module in runner.model.modules():
        name = type(module).__name__
        if name == 'Qwen4ReferenceGatedDeltaNet':
            layer = module.layer_id
            if layer in seen['gdn']: raise RuntimeError('duplicate GDN state owner')
            seen['gdn'].add(layer)
            cache = req_pool.mamba2_layer_cache(layer)
            add(f'gdn{layer}.conv', cache.conv[0][recurrent_slot], cache.conv[0])
            add(f'gdn{layer}.temporal', cache.temporal[recurrent_slot], cache.temporal)
        elif name == 'Qwen4ReferencePLE':
            layer = module.layer_id
            if layer in seen['ple']: raise RuntimeError('duplicate PLE state owner')
            seen['ple'].add(layer)
            cache = req_pool.short_conv_layer_cache(layer)
            add(f'ple{layer}.conv', cache[recurrent_slot], cache)
        elif name == 'Qwen4ReferenceIndexer':
            layer = module.layer_id
            if layer in seen['qsa']: raise RuntimeError('duplicate QSA state owner')
            seen['qsa'].add(layer)
            for label, buffer in (('k', kv_pool.get_key_buffer(layer)),
                                  ('v', kv_pool.get_value_buffer(layer))):
                add(f'qsa{layer}.{label}', buffer.index_select(0, mapping), buffer)
            raw = kv_pool.get_qsa_key_state_buffer(layer)
            add(f'qsa{layer}.raw', raw[slot * 4:slot * 4 + 4], raw)
            compressed = kv_pool.get_qsa_compressed_k_buffer(layer)
            groups = torch.unique(mapping // 4)
            add(f'qsa{layer}.compressed', compressed.index_select(0, groups), compressed)
    if (seen['gdn'] != set(range(48)) - set(range(3, 48, 4)) or
            seen['qsa'] != set(range(3, 48, 4)) or seen['ple'] != {1}):
        raise RuntimeError('incomplete live Qwen3.8 state coverage')
    positions = kv_pool.qsa_rope_positions
    add('qsa.positions', positions[slot * 4:slot * 4 + 4], positions)
    return dict(request_id=request.rid, request_slot=slot, recurrent_slot=recurrent_slot,
                input_tokens=token_count, sampled_output_ids=list(request.output_ids),
                tensors=tensors)
