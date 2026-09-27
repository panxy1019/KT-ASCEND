"""Experimental process-local cache for immutable, provenance-checked GGUF.

Not imported by the production runtime. Cached values are private CPU BF16
tensors; existing pinned slots and caller H2D event fences remain unchanged.
"""


class VerifiedExpertCache:
    def __init__(self, binding, expert_count, bytes_per_expert):
        self.binding = binding
        self.expert_count = expert_count
        self.bytes_per_expert = bytes_per_expert
        self.values = {}
        self.hits = 0
        self.misses = 0

    def get(self, binding, expert, build):
        if binding != self.binding:
            raise ValueError('BF16_CACHE_SOURCE_BINDING_CHANGED')
        if not 0 <= expert < self.expert_count:
            raise ValueError('BF16_CACHE_EXPERT_INDEX')
        if expert in self.values:
            self.hits += 1
            return self.values[expert]
        # A failed validation must never publish a partial cache entry.
        value = build()
        self.values[expert] = value
        self.misses += 1
        return value

    @property
    def resident_bytes(self):
        return len(self.values) * self.bytes_per_expert


def source_binding(writer, wrapper):
    # Hold the actual mmap tensor objects in writer.projections. Runtime must
    # also enforce the immutable read-only GGUF manifest; tensor versions cannot
    # detect an external file writer or unsafe .data mutation.
    return (id(wrapper.gguf_loader), int(wrapper.layer_idx), tuple(
        (name, value.data_ptr(), tuple(value.shape), str(value.dtype), value._version)
        for name, value in sorted(writer.projections.items())))


def cached_stager_class(base):
    class CachedLlamafileBF16Stager(base):
        def __init__(self, wrapper, buffers, tp_world_size):
            super().__init__(wrapper, buffers, tp_world_size)
            self.wrapper = wrapper
            binding = source_binding(self, wrapper)
            cache = getattr(wrapper, '_round7_verified_bf16_cache', None)
            if cache is None:
                size = 3 * int(wrapper.hidden_size) * self.intermediate * 2
                cache = VerifiedExpertCache(binding, self.num_experts, size)
                wrapper._round7_verified_bf16_cache = cache
            if cache.binding != binding:
                raise ValueError('BF16_CACHE_SOURCE_BINDING_CHANGED')
            self.cache = cache

        def submit(self, expert_id, slot):
            if not 0 <= expert_id < self.num_experts or slot not in (0, 1):
                raise ValueError('LLAMAFILE_BF16_STAGING_INDEX')
            torch = self.torch

            def build():
                rounded = {}
                for name, projection in self.projections.items():
                    source = projection[expert_id]
                    value = source.to(torch.bfloat16)
                    if not torch.equal(source, value.float()):
                        raise ValueError('LLAMAFILE_BF16_STAGING_WEIGHT_IDENTITY')
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError('LLAMAFILE_BF16_STAGING_NONFINITE_WEIGHT')
                    rounded[name] = value
                return rounded

            values = self.cache.get(source_binding(self, self.wrapper), expert_id, build)
            w13 = self.buffers['w13_weight'][slot]
            if self.layout == 'raw':
                w13[:self.intermediate, :].copy_(values['gate'])
                w13[self.intermediate:, :].copy_(values['up'])
                self.buffers['w2_weight'][slot].copy_(values['down'])
            else:
                w13[:, :self.intermediate].copy_(values['gate'].T)
                w13[:, self.intermediate:].copy_(values['up'].T)
                self.buffers['w2_weight'][slot].copy_(values['down'].T)

    return CachedLlamafileBF16Stager
