"""Exact source transforms for the bounded Round14a concurrency snapshot."""

from __future__ import annotations


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Round14a concurrency {label} precondition mismatch")
    return source.replace(old, new)


def replace_between(source: str, start: str, end: str, replacement: str, label: str) -> str:
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError(f"Round14a concurrency {label} boundary mismatch")
    lo = source.index(start)
    hi = source.index(end, lo)
    return source[:lo] + replacement + source[hi:]


def patch_qwen(source: str) -> str:
    source = replace_once(
        source,
        '    if int(forward_batch.batch_size) != 1:\n'
        '        raise RuntimeError("Round9 reference model requires batch_size=1")\n',
        '    from sglang.srt.layers.round9.round14a_concurrency_runtime import configured_max\n'
        '    batch_size = int(forward_batch.batch_size)\n'
        '    if not 1 <= batch_size <= configured_max():\n'
        '        raise RuntimeError("Round14a reference model exceeds bounded concurrency")\n',
        "model batch guard",
    )

    old_gdn = '''        state_indices = req_pool.get_mamba_indices(
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
'''
    new_gdn = '''        state_indices = req_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        ).long()
        if (state_indices.numel() != int(forward_batch.batch_size)
                or bool((state_indices == 0).any())):
            raise RuntimeError("Round14a GDN requires one non-padding state per request")
        layer_cache = req_pool.mamba2_layer_cache(self.layer_id)
        conv_pool = layer_cache.conv[0]
        recurrent_pool = layer_cache.temporal

        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        a, _ = self.in_proj_a(hidden_states)
        b, _ = self.in_proj_b(hidden_states)
        z = z.reshape(-1, self.num_v_heads, self.head_v_dim)
        from sglang.srt.layers.round9.round14a_concurrency_runtime import request_slices
        normalized_rows = []
        for ordinal, unused_req_slot, start, stop in request_slices(
                forward_batch, hidden_states.shape[0]):
            slot = state_indices[ordinal]
            conv_state = conv_pool[slot]
            recurrent_state = recurrent_pool[slot]
            if recurrent_state.dtype != torch.float32:
                raise RuntimeError("Round14a GDN recurrent pool must be FP32")
            state = GDNState(conv_state=conv_state, recurrent_state=recurrent_state)
            normalized, next_state = eager_gdn(
                mixed_qkv[start:stop], a[start:stop], b[start:stop], z[start:stop],
                self.conv1d.weight, self.A_log, self.dt_bias, self.norm.weight,
                self.reference_config, initial_state=state,
            )
            conv_pool[slot].copy_(next_state.conv_state)
            recurrent_pool[slot].copy_(next_state.recurrent_state)
            normalized_rows.append(normalized)
        normalized = torch.cat(normalized_rows, dim=0)
        output, _ = self.out_proj(normalized.flatten(-2))
        return output
'''
    source = replace_once(source, old_gdn, new_gdn, "GDN request state split")

    index_start = "    def forward(self, hidden_states, positions, forward_batch):\n"
    index_end = "\n\nclass Qwen4ReferencePLE(SynchronousPLE):"
    indexer = '''    def _forward_one(self, qk, positions, forward_batch, req_slot):
        pool = forward_batch.token_to_kv_pool
        logical = positions.long()
        if logical.ndim != 1 or logical.numel() == 0:
            raise RuntimeError("Round14a QSA requires a nonempty one-axis token span")
        if int(logical[0]) == 0:
            pool.reset_qsa_request(self.layer_id, int(req_slot))
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
                end - self.ratio + 1, end + 1,
                device=member_positions.device, dtype=member_positions.dtype,
            )
            member_positions = member_positions.reshape(-1)
            if (member_positions.shape != expected_positions.shape
                    or not torch.equal(member_positions, expected_positions)):
                raise RuntimeError("QSA compression ring has stale or missing positions")
            pooled = _mean_in_storage_dtype(ring.index_select(0, member_locs))
            pooled = self.k_layernorm(pooled).reshape(1, self.head_dim)
            pooled = self._rope(pooled, logical.new_tensor([end - self.ratio + 1]))
            first_token_slot = forward_batch.req_to_token_pool.req_to_token[
                req_slot, end - self.ratio + 1
            ].long()
            if int(first_token_slot) <= 0 or int(first_token_slot) % self.ratio:
                raise RuntimeError("QSA compression group is not KV-page aligned")
            pool.set_qsa_compressed_k_buffer(
                self.layer_id, (first_token_slot // self.ratio).reshape(1),
                pooled[:, None, :],
            )
        completed = int((logical[-1] + 1) // self.ratio)
        compressed = pool.get_qsa_compressed_k_buffer(self.layer_id)
        if completed:
            starts = torch.arange(completed, device=logical.device) * self.ratio
            physical = forward_batch.req_to_token_pool.req_to_token[
                req_slot, starts
            ].long()
            if (bool((physical <= 0).any())
                    or bool(physical.remainder(self.ratio).ne(0).any())):
                raise RuntimeError("QSA compressed logical groups lack aligned KV slots")
            compressed_keys = compressed.index_select(0, physical // self.ratio)
            compressed_keys = compressed_keys.reshape(completed, self.head_dim)
        else:
            compressed_keys = raw_key.new_empty((0, self.head_dim))
        selected = select_tokens(
            query, compressed_keys, logical, ratio=self.ratio,
            topk_blocks=self.token_topk // self.ratio,
        )
        return query, selected

    def forward(self, hidden_states, positions, forward_batch):
        pool = forward_batch.token_to_kv_pool
        _missing_pool(
            pool,
            ("get_qsa_key_state_buffer", "set_qsa_key_state_buffer",
             "set_qsa_rope_position_buffer", "get_qsa_rope_position_buffer",
             "get_qsa_compressed_k_buffer", "set_qsa_compressed_k_buffer",
             "reset_qsa_request"),
            "QSA",
        )
        if int(getattr(pool, "qsa_compress_ratio", -1)) != self.ratio:
            raise RuntimeError("QSA pool compression ratio does not match the model")
        qk, _ = self.index_qk_proj(hidden_states)
        from sglang.srt.layers.round9.round14a_concurrency_runtime import request_slices
        queries, selections = [], []
        for unused_ordinal, req_slot, start, stop in request_slices(
                forward_batch, hidden_states.shape[0]):
            query, selected = self._forward_one(
                qk[start:stop], positions[start:stop], forward_batch, req_slot)
            queries.append(query)
            selections.append(selected)
        return torch.cat(queries, dim=0), torch.cat(selections, dim=0)
'''
    source = replace_between(source, index_start, index_end, indexer, "QSA indexer")

    ple_start = "    def prepare(self, input_ids, forward_batch):\n"
    ple_end = "\n\n    def run(self, hidden_states, embeddings, state_slot, forward_batch):"
    ple_prepare = '''    def _prepare_one(self, input_ids, forward_batch, req_slot):
        req_pool = forward_batch.req_to_token_pool
        slot = req_pool.get_mamba_indices(req_slot.reshape(1)).long()
        if slot.numel() != 1 or int(slot[0]) == 0:
            raise RuntimeError("Round14a PLE requires one non-padding state slot")
        history = req_pool.get_ngram_context(slot)[0].tolist()
        hasher = NGramHasher(
            ngram_size=self.ngram_size, heads_per_ngram=self.heads_per_ngram,
            vocab_size=self.vocab_size, eos_token_id=self.eos_token_id,
            multipliers=self.layer_multipliers.tolist(),
            head_sizes=self.ngram_heads_vocab_sizes.tolist(),
            head_offsets=self.ngram_heads_offsets.tolist(),
        )
        rows, next_history = hasher.hash_tokens(input_ids.long().tolist(), history)
        row_ids = torch.tensor(rows, dtype=torch.long, device=input_ids.device)
        async_ple = getattr(self, "_round9_async_ple", None)
        embeddings = (async_ple.prepare(input_ids, slot, row_ids, self._table)
                      if async_ple is not None else self._table.gather_rows(row_ids))
        expected = (*row_ids.shape, self.head_dim)
        if tuple(embeddings.shape) != expected:
            raise RuntimeError(
                f"PLE table returned {tuple(embeddings.shape)}, expected {expected}"
            )
        return slot, embeddings.flatten(1), torch.tensor(
            next_history, dtype=torch.long, device=input_ids.device).unsqueeze(0)

    def prepare(self, input_ids, forward_batch):
        if self._table is None:
            raise RuntimeError("Round14a PLE original checkpoint table is not bound")
        req_pool = forward_batch.req_to_token_pool
        _missing_pool(req_pool, ("get_mamba_indices", "get_ngram_context"), "PLE")
        from sglang.srt.layers.round9.round14a_concurrency_runtime import request_slices
        spans = request_slices(forward_batch, input_ids.shape[0])
        rows = []
        for unused_ordinal, req_slot, start, stop in spans:
            rows.append((start, stop, *self._prepare_one(
                input_ids[start:stop], forward_batch,
                forward_batch.req_pool_indices[unused_ordinal])))
        if len(rows) == 1:
            return rows[0][2:]
        return rows
'''
    source = replace_between(source, ple_start, ple_end, ple_prepare, "PLE prepare")

    old_before = '''        if self.ple is not None:
            if ple_data is None:
                raise RuntimeError("Qwen4 PLE layer is missing prepared table rows")
            slot, embeddings, _ = ple_data
            query = hidden if residual is None else hidden + residual
            hidden = hidden + self.ple.run(query, embeddings, slot, forward_batch)
        return self.attn_hyper_connection.mix(hidden)
'''
    new_before = '''        if self.ple is not None:
            if ple_data is None:
                raise RuntimeError("Qwen4 PLE layer is missing prepared table rows")
            query = hidden if residual is None else hidden + residual
            if isinstance(ple_data, list):
                deltas = []
                for start, stop, slot, embeddings, unused_history in ple_data:
                    deltas.append(self.ple.run(
                        query[start:stop], embeddings, slot, forward_batch))
                hidden = hidden + torch.cat(deltas, dim=0)
            else:
                slot, embeddings, _ = ple_data
                hidden = hidden + self.ple.run(query, embeddings, slot, forward_batch)
        return self.attn_hyper_connection.mix(hidden)
'''
    source = replace_once(source, old_before, new_before, "PLE layer consume")

    self_start = "    def self_attention(self, positions, hidden_states, forward_batch):\n"
    self_end = "\n\n    def forward(self, positions, hidden_states, residual, forward_batch, **kwargs):"
    self_attention = '''    def self_attention(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        q_gate, k, v = qkv.split(
            [self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        q, gate = q_gate.reshape(-1, self.num_heads, self.head_dim * 2).chunk(2, -1)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        q_flat, k_flat = self._apply_qk_norm(q.flatten(-2), k.flatten(-2))
        q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)
        q = q_flat.reshape(-1, self.num_heads, self.head_dim)
        k = k_flat.reshape(-1, self.num_kv_heads, self.head_dim)
        pool = forward_batch.token_to_kv_pool
        _missing_pool(pool, ("set_kv_buffer", "get_key_buffer", "get_value_buffer"),
                      "ordinary QSA KV")
        pool.set_kv_buffer(self.attn, forward_batch.out_cache_loc, k, v)
        _, selected = self.indexer(hidden_states, positions, forward_batch)
        k_pool = _flatten_token_kv(
            pool.get_key_buffer(self.attn.layer_id), self.num_kv_heads, self.head_dim)
        v_pool = _flatten_token_kv(
            pool.get_value_buffer(self.attn.layer_id), self.num_kv_heads, self.head_dim)
        context_consumer = getattr(self, "_round9_context_consumer", None)
        from sglang.srt.layers.round9.round14a_concurrency_runtime import request_slices
        outputs = []
        for unused_ordinal, req_slot, start, stop in request_slices(
                forward_batch, hidden_states.shape[0]):
            local_positions = positions[start:stop]
            local_selected = selected[start:stop]
            logical_ids = local_selected[local_selected >= 0]
            max_logical = int(local_positions.max())
            if logical_ids.numel() and int(logical_ids.max()) > max_logical:
                raise RuntimeError("QSA selected a future logical token")
            logical_slots = forward_batch.req_to_token_pool.req_to_token[
                req_slot, : max_logical + 1].long()
            if bool((logical_slots <= 0).any()):
                raise RuntimeError("ordinary QSA KV map contains an unallocated slot")
            keys = k_pool.index_select(0, logical_slots)
            values = v_pool.index_select(0, logical_slots)
            if context_consumer is None:
                output = sparse_gqa_candidate(
                    q[start:stop], keys, values, local_selected)
            elif hasattr(context_consumer, "consume_for"):
                output = context_consumer.consume_for(
                    req_slot, int(self.attn.layer_id), q[start:stop], keys, values,
                    local_selected, local_positions, sparse_gqa_candidate)
            else:
                output = context_consumer.consume(
                    int(self.attn.layer_id), q[start:stop], keys, values,
                    local_selected, local_positions, sparse_gqa_candidate)
            outputs.append(output)
        attn_output = torch.cat(outputs, dim=0) * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output.flatten(-2))
        return output
'''
    source = replace_between(source, self_start, self_end, self_attention,
                             "QSA attention request split")

    old_commit = '''        if ple_data is not None:
            req_pool = forward_batch.req_to_token_pool
            _missing_pool(req_pool, ("set_ngram_context",), "PLE")
            req_pool.set_ngram_context(ple_data[0], ple_data[2])
        return hidden
'''
    new_commit = '''        if ple_data is not None:
            req_pool = forward_batch.req_to_token_pool
            _missing_pool(req_pool, ("set_ngram_context",), "PLE")
            if isinstance(ple_data, list):
                for unused_start, unused_stop, slot, unused_embeddings, history in ple_data:
                    req_pool.set_ngram_context(slot, history)
            else:
                req_pool.set_ngram_context(ple_data[0], ple_data[2])
        return hidden
'''
    source = replace_once(source, old_commit, new_commit, "PLE history commit")
    compile(source, "qwen4_exp.py", "exec")
    return source


def patch_context_reference_binding(source: str) -> str:
    source = replace_once(
        source,
        '        and args.disable_cuda_graph and args.max_running_requests == 1\n',
        '        and args.disable_cuda_graph\n'
        '        and args.max_running_requests == __import__(\n'
        '            __package__ + ".round14a_concurrency_runtime",\n'
        '            fromlist=["configured_max"]).configured_max()\n',
        "context scheduler gate",
    )
    old_runtime = '''    runtime = ContextReferenceRuntime(
        profile["mode"], event=event,
        approved_prefill_candidate=(
            runner._round9_context_reference_gate.get(
                "qsa_source_binding", {}).get("mode") == "APPROVED_CANDIDATE"),
    )
    for layer, owner in owners:
'''
    new_runtime = '''    approved = (runner._round9_context_reference_gate.get(
        "qsa_source_binding", {}).get("mode") == "APPROVED_CANDIDATE")
    def factory():
        return ContextReferenceRuntime(
            profile["mode"], event=event,
            approved_prefill_candidate=approved)
    from .round14a_concurrency_runtime import (
        BatchedContextReferenceRuntime, enabled as concurrency_enabled)
    runtime = (BatchedContextReferenceRuntime(factory, event=event)
               if concurrency_enabled() else factory())
    for layer, owner in owners:
'''
    source = replace_once(source, old_runtime, new_runtime, "context runtime factory")
    old_before = '''    runner = scheduler.tp_worker.model_runner
    runtime = getattr(runner, "_round9_context_reference", None)
    if runtime is None or len(batch.reqs) != 1:
        raise RuntimeError("enabled context reference requires one bound serial request")
    request = batch.reqs[0]
    previous = runtime.request_token
    if previous is not None and previous is not request:
        reason = getattr(previous, "finished_reason", None)
        if (runtime.active or reason is None
                or reason.__class__.__name__ != "FINISH_ABORT"):
            raise RuntimeError("context request switched without a drained abort")
        runtime.abort(previous)
    runtime.begin(request.rid, request)
    runner._round9_context_request = request
'''
    new_before = '''    runner = scheduler.tp_worker.model_runner
    runtime = getattr(runner, "_round9_context_reference", None)
    if runtime is None:
        raise RuntimeError("enabled context reference lacks a bound runtime")
    if hasattr(runtime, "begin_batch"):
        runtime.begin_batch(batch)
        runner._round9_context_request = tuple(batch.reqs)
        return
    if len(batch.reqs) != 1:
        raise RuntimeError("serial context reference received a multi-request batch")
    request = batch.reqs[0]
    previous = runtime.request_token
    if previous is not None and previous is not request:
        reason = getattr(previous, "finished_reason", None)
        if (runtime.active or reason is None
                or reason.__class__.__name__ != "FINISH_ABORT"):
            raise RuntimeError("context request switched without a drained abort")
        runtime.abort(previous)
    runtime.begin(request.rid, request)
    runner._round9_context_request = request
'''
    source = replace_once(source, old_before, new_before, "context before")
    old_after = '''    if runtime is None:
        raise RuntimeError("context reference batch is unleased")
    if request is None:
        if runtime.active:
            raise RuntimeError("context reference batch is unleased")
        return
    runtime.after(request, finished=bool(request.finished()))
    runner._round9_context_request = None
'''
    new_after = '''    if runtime is None:
        raise RuntimeError("context reference batch is unleased")
    if hasattr(runtime, "after_batch"):
        if request is None:
            if runtime.active:
                raise RuntimeError("context reference batch is unleased")
            return
        runtime.after_batch()
        runner._round9_context_request = None
        return
    if request is None:
        if runtime.active:
            raise RuntimeError("context reference batch is unleased")
        return
    runtime.after(request, finished=bool(request.finished()))
    runner._round9_context_request = None
'''
    source = replace_once(source, old_after, new_after, "context after")
    compile(source, "context_reference_binding.py", "exec")
    return source


def patch_ple_binding(source: str) -> str:
    source = replace_once(
        source,
        "            not args.disable_cuda_graph or args.max_running_requests != 1 or args.tp_size != 1 or\n",
        "            not args.disable_cuda_graph or args.max_running_requests != __import__(\n"
        "                __package__ + '.round14a_concurrency_runtime',\n"
        "                fromlist=['configured_max']).configured_max() or args.tp_size != 1 or\n",
        "PLE scheduler gate",
    )
    old_runtime = '''    from .ple_runtime import PLEConsumerRuntime,bind_ple_consumer
    runtime=PLEConsumerRuntime(ring)
    bind_ple_consumer(ple,runtime)
'''
    new_runtime = '''    from .ple_runtime import PLEConsumerRuntime,bind_ple_consumer
    from .round14a_concurrency_runtime import (
        BatchedPLEConsumerRuntime, enabled as concurrency_enabled)
    runtime=(BatchedPLEConsumerRuntime(ring, PLEConsumerRuntime)
             if concurrency_enabled() else PLEConsumerRuntime(ring))
    bind_ple_consumer(ple,runtime)
'''
    source = replace_once(source, old_runtime, new_runtime, "PLE runtime factory")
    old_before = '''    runtime=getattr(runner,'_round9_ple',None)
    if runtime is None:raise RuntimeError('async PLE model binding is missing')
    if len(batch.reqs)!=1:raise RuntimeError('async PLE requires one live request')
    if batch.forward_mode.is_extend():phase='prefill_input'
    elif batch.forward_mode.is_decode():phase='sampled_decode'
    else:raise RuntimeError('unsupported async PLE forward mode')
    request=batch.reqs[0]
    boundary=getattr(scheduler,'_round9_dynamic_boundary',None)
    epoch=boundary.generation if boundary is not None else 0
    if request.mamba_pool_idx is None:raise RuntimeError('async PLE request state is not allocated')
    runtime.begin(request.rid,epoch,batch.input_ids.detach().cpu().tolist(),
                  int(request.mamba_pool_idx),phase=phase)
    runner._round9_ple_request=request
    runner._round9_ple_async_before=runtime.async_batches
'''
    new_before = '''    runtime=getattr(runner,'_round9_ple',None)
    if runtime is None:raise RuntimeError('async PLE model binding is missing')
    if batch.forward_mode.is_extend():phase='prefill_input'
    elif batch.forward_mode.is_decode():phase='sampled_decode'
    else:raise RuntimeError('unsupported async PLE forward mode')
    boundary=getattr(scheduler,'_round9_dynamic_boundary',None)
    epoch=boundary.generation if boundary is not None else 0
    runner._round9_ple_async_before=runtime.async_batches
    if hasattr(runtime,'begin_batch'):
        runtime.begin_batch(batch,generation=epoch,phase=phase)
        runner._round9_ple_request=tuple(batch.reqs)
        return
    if len(batch.reqs)!=1:raise RuntimeError('serial PLE received a multi-request batch')
    request=batch.reqs[0]
    if request.mamba_pool_idx is None:raise RuntimeError('async PLE request state is not allocated')
    runtime.begin(request.rid,epoch,batch.input_ids.detach().cpu().tolist(),
                  int(request.mamba_pool_idx),phase=phase)
    runner._round9_ple_request=request
'''
    source = replace_once(source, old_before, new_before, "PLE before")
    old_after = '''    context=runtime.context
    if context is not None:
        asynchronous=runtime.async_batches>runner._round9_ple_async_before
        if asynchronous:runner._round9_ple_credit_requests.add(context[0])
        runtime.end()
        runner._round9_ple_event(dict(status='PLE_BATCH_DRAINED',request_id=context[0],
            generation=context[1],tokens=len(context[2]),phase=context[4],async_engaged=asynchronous))
    request=getattr(runner,'_round9_ple_request',None)
    if request is not None and request.finished() and runner._round9_joint_host_budget is None:
        if request.rid in runner._round9_ple_credit_requests:
            runtime.ring.budget.retire_request(request.rid)
            runner._round9_ple_credit_requests.remove(request.rid)
        runner._round9_ple_request=None
'''
    new_after = '''    request=getattr(runner,'_round9_ple_request',None)
    if hasattr(runtime,'after_batch'):
        if request is None:
            if runtime.context is not None:raise RuntimeError('batched PLE is unleased')
            return
        for owner,context,asynchronous in runtime.after_batch():
            if asynchronous:runner._round9_ple_credit_requests.add(context[0])
            runner._round9_ple_event(dict(status='PLE_BATCH_DRAINED',request_id=context[0],
                generation=context[1],tokens=len(context[2]),phase=context[4],
                async_engaged=asynchronous))
            if owner.finished() and runner._round9_joint_host_budget is None:
                if owner.rid in runner._round9_ple_credit_requests:
                    runtime.ring.budget.retire_request(owner.rid)
                    runner._round9_ple_credit_requests.remove(owner.rid)
        runner._round9_ple_request=None
        return
    context=runtime.context
    if context is not None:
        asynchronous=runtime.async_batches>runner._round9_ple_async_before
        if asynchronous:runner._round9_ple_credit_requests.add(context[0])
        runtime.end()
        runner._round9_ple_event(dict(status='PLE_BATCH_DRAINED',request_id=context[0],
            generation=context[1],tokens=len(context[2]),phase=context[4],async_engaged=asynchronous))
    if request is not None and request.finished() and runner._round9_joint_host_budget is None:
        if request.rid in runner._round9_ple_credit_requests:
            runtime.ring.budget.retire_request(request.rid)
            runner._round9_ple_credit_requests.remove(request.rid)
        runner._round9_ple_request=None
'''
    source = replace_once(source, old_after, new_after, "PLE after")
    compile(source, "ple_binding.py", "exec")
    return source


def patch_prefill_runtime(source: str) -> str:
    old = '''        if not batch.forward_mode.is_extend(): return False
        if len(batch.reqs) != 1: raise ValueError('layerwise requires serial requests')
        req = batch.reqs[0]
'''
    new = '''        if not batch.forward_mode.is_extend(): return False
        if len(batch.reqs) != 1:
            from .round14a_concurrency_runtime import validate_scheduler_requests
            validate_scheduler_requests(batch, threshold=self.threshold)
            return False
        req = batch.reqs[0]
'''
    source = replace_once(source, old, new, "prefill H129 per-request gate")
    compile(source, "prefill_runtime.py", "exec")
    return source


def patch_prefill_binding(source: str) -> str:
    source = replace_once(
        source,
        "            not args.disable_cuda_graph or args.max_running_requests != 1 or\n",
        "            not args.disable_cuda_graph or args.max_running_requests != __import__(\n"
        "                __package__ + '.round14a_concurrency_runtime',\n"
        "                fromlist=['configured_max']).configured_max() or\n",
        "prefill scheduler gate",
    )
    compile(source, "prefill_binding.py", "exec")
    return source


def patch_context_binding(source: str) -> str:
    source = replace_once(
        source,
        "            args.max_running_requests != 1 or args.max_mamba_cache_size != 1 or\n",
        "            args.max_running_requests != __import__(\n"
        "                __package__ + '.round14a_concurrency_runtime',\n"
        "                fromlist=['configured_max']).configured_max() or\n"
        "            args.max_mamba_cache_size != args.max_running_requests or\n",
        "context pool scheduler gate",
    )
    compile(source, "context_binding.py", "exec")
    return source


def patch_hybrid_probe(source: str) -> str:
    source = replace_once(source, "state = {'context': None, 'rows': 0}",
                          "state = {'contexts': (), 'rows': 0}",
                          "hybrid probe state")
    old_before = '''    def before(batch, *, placement_epoch):
        state['context'] = None
        selected = original_before(batch, placement_epoch=placement_epoch)
        if batch.forward_mode.is_extend():
            req = batch.reqs[0]
            state['context'] = dict(request_id=req.rid, generation=placement_epoch,
                request_tokens=len(req.origin_input_ids), batch_tokens=req.extend_input_len,
                start=req.last_extend_prefix_len, threshold=runtime.threshold,
                path='LAYERWISE' if selected else 'HYBRID', basis='REQUEST_TOTAL')
            emit(dict(state['context'], stage='SELECTION'))
        return selected
'''
    new_before = '''    def before(batch, *, placement_epoch):
        state['contexts'] = ()
        selected = original_before(batch, placement_epoch=placement_epoch)
        if batch.forward_mode.is_extend():
            contexts = []
            for req in batch.reqs:
                context = dict(request_id=req.rid, generation=placement_epoch,
                    request_tokens=len(req.origin_input_ids), batch_tokens=req.extend_input_len,
                    start=req.last_extend_prefix_len, threshold=runtime.threshold,
                    path='LAYERWISE' if selected else 'HYBRID', basis='REQUEST_TOTAL',
                    engine_batch_size=len(batch.reqs))
                contexts.append(context)
                emit(dict(context, stage='SELECTION'))
            state['contexts'] = tuple(contexts)
        return selected
'''
    source = replace_once(source, old_before, new_before, "hybrid probe before")
    source = replace_once(source, "            state['context'] = None\n",
                          "            state['contexts'] = ()\n",
                          "hybrid probe after")
    old_call = '''            context = state['context']
            if context is None or context['path'] != 'HYBRID':
                return original(*args, **kwargs)
            started = time.monotonic_ns()
            result = original(*args, **kwargs)
            emit(dict(context, layer=layer, stage=stage, outcome='RETURN',
                      start_ns=started, end_ns=time.monotonic_ns()))
            return result
'''
    new_call = '''            contexts = state['contexts']
            if not contexts or any(context['path'] != 'HYBRID' for context in contexts):
                return original(*args, **kwargs)
            started = time.monotonic_ns()
            result = original(*args, **kwargs)
            ended = time.monotonic_ns()
            for context in contexts:
                emit(dict(context, layer=layer, stage=stage, outcome='RETURN',
                          span_scope='SHARED_ENGINE_BATCH', start_ns=started, end_ns=ended))
            return result
'''
    source = replace_once(source, old_call, new_call, "hybrid probe dispatch")
    compile(source, "hybrid_probe.py", "exec")
    return source


def patch_kv_cache_mixin(source: str) -> str:
    old = '''            if (not server_args.disable_radix_cache or not server_args.disable_cuda_graph
                    or server_args.max_running_requests != 1 or not self.spec_algorithm.is_none()
                    or server_args.disaggregation_mode != "null" or server_args.enable_memory_saver
                    or self.page_size % 4):
                raise ValueError("Round9 requires eager batch1, aligned pages, no radix/spec/disagg/memory-saver")
'''
    new = '''            from sglang.srt.layers.round9.round14a_concurrency_runtime import configured_max
            if (not server_args.disable_radix_cache or not server_args.disable_cuda_graph
                    or server_args.max_running_requests != configured_max()
                    or server_args.max_mamba_cache_size != configured_max()
                    or not self.spec_algorithm.is_none()
                    or server_args.disaggregation_mode != "null" or server_args.enable_memory_saver
                    or self.page_size % 4):
                raise ValueError("Round14a requires bounded eager batches, aligned pages, no radix/spec/disagg/memory-saver")
'''
    source = replace_once(source, old, new, "Qwen Mamba pool scheduler gate")
    compile(source, "model_runner_kv_cache_mixin.py", "exec")
    return source


def patch_framework_hooks(source: str) -> str:
    old_ple = '''    ple = getattr(runner, '_round9_ple', None)
    if ple is not None:
        if ple.context is not None:
            raise RuntimeError('cancel cleanup reached active PLE consumer')
        if getattr(runner, '_round9_ple_request', None) is request:
            budget = getattr(runner, '_round9_joint_host_budget', None)
            if request.rid in runner._round9_ple_credit_requests:
                (budget if budget is not None else ple.ring.budget).retire_request(
                    request.rid)
                runner._round9_ple_credit_requests.remove(request.rid)
            if budget is not None:
                snapshot = budget.snapshot()
                baseline = getattr(runner, '_round9_joint_budget_baseline', None)
                if baseline is None or snapshot != baseline:
                    raise RuntimeError(
                        'cancelled joint host budget did not return to baseline')
            # A small batch can take the synchronous PLE fallback and therefore
            # hold no async credit.  It still owns this request token until the
            # normal after-hook or cancellation cleanup retires it.
            runner._round9_ple_request = None

    context = getattr(runner, '_round9_context_reference', None)
    if context is not None and context.request_token is request:
        if context.active:
            raise RuntimeError('cancel cleanup reached active context consumer')
        context.abort(request)
'''
    new_ple = '''    ple = getattr(runner, '_round9_ple', None)
    if ple is not None:
        batched = hasattr(ple, 'abort_request')
        if batched:
            ple.abort_request(request)
        elif ple.context is not None:
            raise RuntimeError('cancel cleanup reached active PLE consumer')
        owned = getattr(runner, '_round9_ple_request', None)
        owns_request = (request in owned if isinstance(owned, tuple) else owned is request)
        if owns_request or batched:
            budget = getattr(runner, '_round9_joint_host_budget', None)
            if request.rid in runner._round9_ple_credit_requests:
                (budget if budget is not None else ple.ring.budget).retire_request(request.rid)
                runner._round9_ple_credit_requests.remove(request.rid)
            if budget is not None:
                snapshot = budget.snapshot()
                baseline = getattr(runner, '_round9_joint_budget_baseline', None)
                if baseline is None or snapshot != baseline:
                    raise RuntimeError('cancelled joint host budget did not return to baseline')
            if not isinstance(owned, tuple):
                runner._round9_ple_request = None

    context = getattr(runner, '_round9_context_reference', None)
    if context is not None and hasattr(context, 'abort_request'):
        context.abort_request(request)
    elif context is not None and context.request_token is request:
        if context.active:
            raise RuntimeError('cancel cleanup reached active context consumer')
        context.abort(request)
'''
    source = replace_once(source, old_ple, new_ple, "cancel plane cleanup")
    compile(source, "framework_hooks.py", "exec")
    return source


def patch_streaming_binding(source: str) -> str:
    source = replace_once(
        source,
        "from .streaming_prefill import StreamingPrefillCoordinator\n",
        "from .streaming_prefill import StreamingPrefillCoordinator\n\n\n"
        "def _round14a_event(path, item):\n"
        "    with path.open('a') as stream:\n"
        "        stream.write(json.dumps(item, allow_nan=False) + '\\n')\n",
        "streaming concurrency event writer",
    )
    old_fence = '''    if len(batch.reqs) != 1:
        raise RuntimeError("enabled streaming mode requires one serial request")
    if not batch.forward_mode.is_extend():
        return
    request = batch.reqs[0]
    # The context allocation epoch is created later in framework_before.  The
    # lifecycle deliberately permits filling it in on the same live identity.
    scheduler_request_admitted(scheduler, request, None)
    if not scheduler_before_normal_submit(
            scheduler, request, "STREAMING_CHUNK"):
        raise RuntimeError(
            "post-cancel streaming chunk reached model-submit boundary")
'''
    new_fence = '''    if not batch.forward_mode.is_extend():
        return
    from .round14a_concurrency_runtime import validate_scheduler_requests
    rows = validate_scheduler_requests(batch)
    for unused_slot, request, unused_total in rows:
        scheduler_request_admitted(scheduler, request, None)
        if not scheduler_before_normal_submit(scheduler, request, "STREAMING_CHUNK"):
            raise RuntimeError("post-cancel streaming chunk reached model-submit boundary")
'''
    source = replace_once(source, old_fence, new_fence, "streaming submit fence")
    marker = '''    request = batch.reqs[0]
    previous = coordinator.request_token
'''
    multi = '''    from .round14a_concurrency_runtime import (
        enabled as concurrency_enabled, validate_scheduler_requests)
    if concurrency_enabled():
        import time
        rows = validate_scheduler_requests(batch)
        runner._round14a_streaming_batch_requests = tuple(batch.reqs)
        runner._round9_streaming_request = tuple(batch.reqs)
        log = Path('/out/round14a-concurrency-events.jsonl')
        _round14a_event(log, dict(status='ROUND14A_ENGINE_BATCH_BEGIN',
                         mode=('extend' if batch.forward_mode.is_extend() else 'decode'),
                         batch_size=len(rows), request_ids=[req.rid for _,req,_ in rows],
                         request_total_tokens=[total for _,_,total in rows],
                         monotonic_ns=time.monotonic_ns()))
        return False
    request = batch.reqs[0]
    previous = coordinator.request_token
'''
    source = replace_once(source, marker, multi, "streaming multi before")
    source = replace_once(
        source,
        '''    if coordinator is None or len(batch.reqs) != 1:
        raise RuntimeError("enabled streaming mode requires one bound serial request")
''',
        '''    if coordinator is None:
        raise RuntimeError("enabled streaming mode requires one bound coordinator")
''',
        "streaming coordinator gate",
    )
    after_marker = '''    request = getattr(runner, "_round9_streaming_request", None)
    if coordinator is None:
        raise RuntimeError("streaming scheduler batch is unleased")
'''
    after_multi = '''    request = getattr(runner, "_round9_streaming_request", None)
    if coordinator is None:
        raise RuntimeError("streaming scheduler batch is unleased")
    if isinstance(request, tuple):
        log = Path('/out/round14a-concurrency-events.jsonl')
        import time
        _round14a_event(log, dict(status='ROUND14A_ENGINE_BATCH_DRAINED',
                         batch_size=len(request), request_ids=[req.rid for req in request],
                         monotonic_ns=time.monotonic_ns()))
        for owner in request:
            if owner.finished():
                scheduler_request_retired(scheduler, owner)
        runner._round9_streaming_request = None
        runner._round14a_streaming_batch_requests = None
        return
'''
    source = replace_once(source, after_marker, after_multi, "streaming multi after")
    compile(source, "streaming_binding.py", "exec")
    return source


def patch_all_runtime(path, source: str) -> str:
    name = path.rsplit("/", 1)[-1]
    functions = {
        "context_reference_binding.py": patch_context_reference_binding,
        "ple_binding.py": patch_ple_binding,
        "prefill_runtime.py": patch_prefill_runtime,
        "prefill_binding.py": patch_prefill_binding,
        "context_binding.py": patch_context_binding,
        "hybrid_probe.py": patch_hybrid_probe,
        "framework_hooks.py": patch_framework_hooks,
        "streaming_binding.py": patch_streaming_binding,
    }
    return functions[name](source) if name in functions else source


__all__ = (
    "patch_qwen", "patch_all_runtime", "patch_kv_cache_mixin",
    "replace_once", "replace_between",
)
