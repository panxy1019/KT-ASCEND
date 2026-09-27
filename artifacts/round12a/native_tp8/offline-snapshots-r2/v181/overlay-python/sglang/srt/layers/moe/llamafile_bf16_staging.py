"""TP1 F32 GGUF staging for the existing BF16 full-context double buffers.

No CPU expert arithmetic or ownership changes. The caller retains the official
device-event fences before reusing either pinned slot. Native writers, when
available, still use the existing upstream path.
"""


def validate_layout(num_experts, hidden, intermediate, shapes, tp_world_size):
    if tp_world_size != 1:
        raise ValueError('LLAMAFILE_BF16_STAGING_TP1_ONLY')
    if min(num_experts, hidden, intermediate) <= 0:
        raise ValueError('LLAMAFILE_BF16_STAGING_DIMENSIONS')
    raw = {'w13_weight': (2, 2 * intermediate, hidden),
           'w2_weight': (2, hidden, intermediate)}
    inference = {'w13_weight': (2, hidden, 2 * intermediate),
                 'w2_weight': (2, intermediate, hidden)}
    if shapes == raw:
        return 'raw'
    if shapes == inference:
        return 'ascend_inference'
    raise ValueError('LLAMAFILE_BF16_STAGING_ASCEND_LAYOUT')


class LlamafileBF16Stager:
    def __init__(self, wrapper, buffers, tp_world_size):
        import torch
        if type(wrapper).__name__ != 'LlamafileMoEWrapper':
            raise ValueError('UNSUPPORTED_BF16_WEIGHT_WRITER')
        self.torch = torch
        self.buffers = buffers
        self.num_experts = int(wrapper.num_experts)
        hidden = int(wrapper.hidden_size)
        intermediate = int(wrapper.moe_intermediate_size)
        self.intermediate = intermediate
        self.layout = validate_layout(self.num_experts, hidden, intermediate,
            {name: tuple(buffers[name].shape) for name in ('w13_weight', 'w2_weight')},
            tp_world_size)
        for name in ('w13_weight', 'w2_weight'):
            buf = buffers[name]
            if buf.device.type != 'cpu' or buf.dtype != torch.bfloat16 or not buf.is_pinned():
                raise ValueError('LLAMAFILE_BF16_STAGING_REQUIRES_PINNED_BF16')
        self.projections = {}
        for name, shape in (('gate', (self.num_experts, intermediate, hidden)),
                            ('up', (self.num_experts, intermediate, hidden)),
                            ('down', (self.num_experts, hidden, intermediate))):
            key = f'blk.{wrapper.layer_idx}.ffn_{name}_exps.weight'
            raw, ggml_type = wrapper.gguf_loader.get_undequanted_tensor_and_ggml_type(key)
            # GGML F32 is0. Do not reinterpret another quantization type asF32.
            if int(ggml_type) != 0 or raw.device.type != 'cpu' or raw.dtype != torch.uint8:
                raise ValueError('LLAMAFILE_BF16_STAGING_REQUIRES_F32_GGUF')
            count = self.num_experts * hidden * intermediate
            if raw.numel() != count * 4:
                raise ValueError('LLAMAFILE_BF16_STAGING_SOURCE_SIZE')
            # Keep mmap-backed views alive throughout this context load.
            self.projections[name] = raw.view(torch.float32).reshape(shape)

    def submit(self, expert_id, slot):
        if not 0 <= expert_id < self.num_experts or slot not in (0, 1):
            raise ValueError('LLAMAFILE_BF16_STAGING_INDEX')
        torch = self.torch
        # The frozen GGUF consists of exact BF16-expanded F32 weights. Reject
        # lossy conversion or nonfinite source rather than silently quantizing.
        rounded = {}
        for name, projection in self.projections.items():
            source = projection[expert_id]
            value = source.to(torch.bfloat16)
            if not torch.equal(source, value.float()):
                raise ValueError('LLAMAFILE_BF16_STAGING_WEIGHT_IDENTITY')
            if not bool(torch.isfinite(value).all()):
                raise ValueError('LLAMAFILE_BF16_STAGING_NONFINITE_WEIGHT')
            rounded[name] = value
        w13 = self.buffers['w13_weight'][slot]
        if self.layout == 'raw':
            w13[:self.intermediate, :].copy_(rounded['gate'])
            w13[self.intermediate:, :].copy_(rounded['up'])
            self.buffers['w2_weight'][slot].copy_(rounded['down'])
        else:
            w13[:, :self.intermediate].copy_(rounded['gate'].T)
            w13[:, self.intermediate:].copy_(rounded['up'].T)
            self.buffers['w2_weight'][slot].copy_(rounded['down'].T)

    def sync(self):
        # CPU tensor staging above is synchronous. H2D remains asynchronous and
        # slot reuse remains guarded by the caller's original event fences.
        pass
