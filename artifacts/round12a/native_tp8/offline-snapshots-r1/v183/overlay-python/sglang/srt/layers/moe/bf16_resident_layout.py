"""Normalize full resident BF16 tensors before expert-slice copies into raw context.

The Ascend unquantized loader transposes and NZ-packs the full resident tensor.
Convert the full tensor (not an NZ expert slice) to ND, then undo that transpose.
No numerical conversion or checkpoint injection is performed.
"""


def resident_to_raw(source, destination):
    import torch
    import torch_npu
    if source.device.type != 'npu' or destination.device.type != 'npu':
        raise RuntimeError('BF16_RESIDENT_NPU_ONLY')
    if source.dtype != torch.bfloat16 or destination.dtype != torch.bfloat16:
        raise RuntimeError('BF16_RESIDENT_DTYPE')
    if source.ndim != 3 or destination.ndim != 3:
        raise RuntimeError('BF16_RESIDENT_RANK')
    # A raw full-context is newly allocated ND; never copy into an NZ slice.
    if torch_npu.get_npu_format(destination) not in (0, 2):
        raise RuntimeError('BF16_FULL_CONTEXT_NOT_BASE_FORMAT')
    current = tuple(source.shape[1:])
    target = tuple(destination.shape[1:])
    if current == target:
        transpose = False
    elif current[::-1] == target:
        transpose = True
    else:
        raise RuntimeError('BF16_RESIDENT_LAYOUT_SHAPE')
    normalized = torch_npu.npu_format_cast(source.detach(), 2)
    if transpose:
        normalized = normalized.transpose(1, 2)
    return normalized.contiguous()
