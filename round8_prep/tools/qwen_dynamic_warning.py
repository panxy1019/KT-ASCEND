"""Temporary diagnostic: bracket real Dynamic copy and audit operations."""
import functools


def make_structure(config):
    import torch
    import torch_npu
    import sglang.srt.layers.moe.kt_ep_wrapper as ep
    from qwen_dynamic_audit import make_structure as original_factory
    original_copy = ep.copy_experts_weights_bf16
    original_update = ep.KTEPWrapperMethod._update_gpu_experts_from_batch

    def mark(message):
        torch.npu.synchronize()
        print('R8_WARNING_TRACE ' + message, flush=True)

    @functools.wraps(original_copy)
    def copy(src_layer, dst_layer, selected_experts):
        mark('COPY_BEGIN')
        for name in ('w13_weight', 'w2_weight'):
            for label, obj in [('source', src_layer), ('dest', dst_layer)]:
                t = getattr(obj, name)
                print('R8_WARNING_OPERAND', name, label, tuple(t.shape),
                      t.stride(), t.storage_offset(), torch_npu.get_npu_format(t), flush=True)
        result = original_copy(src_layer, dst_layer, selected_experts)
        mark('COPY_END')
        return result

    @functools.wraps(original_update)
    def update(self, *args, **kwargs):
        mark('UPDATE_BEGIN')
        result = original_update(self, *args, **kwargs)
        mark('UPDATE_END_AUDIT_NEXT')
        return result

    ep.copy_experts_weights_bf16 = copy
    ep.KTEPWrapperMethod._update_gpu_experts_from_batch = update
    return original_factory(config)
