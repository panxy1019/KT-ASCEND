"""Default-off real operand and physical-layout observer; never timed."""
import time
import json
from pathlib import Path


def make_mlp_hook(config):
    import torch
    from r65_structure_hook import make_hook
    from common import write
    root = Path(config['output'])
    if not str(root).startswith('/workspace/c64-dev/'):
        raise ValueError('OBSERVER_SCOPE')
    structure = make_hook(dict(profile=config['profile'], output=str(root / 'physical-structure.jsonl')))
    calls = {}
    operands = root / 'operands'; operands.mkdir(exist_ok=True)

    def hook(module, inputs, output):
        layer = getattr(module, 'layer_id', None)
        if layer not in range(1, 27): return
        structure(module, inputs, output)
        call = calls.get(layer, 0)
        if call in (1, 4, 7):
            x = inputs[0].detach().cpu().contiguous()
            if x.dtype != torch.bfloat16 or tuple(x.shape) != (1, 2048):
                raise ValueError('REAL_DECODE_OPERAND_IDENTITY')
            torch.save(dict(layer=layer, call=call, tensor=x), operands / f'L{layer:02d}-C{call}.pt')
        if not torch.isfinite(output).all().item():
            raise ValueError('NONFINITE_MLP_OUTPUT')
        calls[layer] = call + 1
        if layer == 26:
            write(root / 'framework-memory.json', dict(time_unix=time.time(),
                  allocated=torch.npu.memory_allocated(), reserved=torch.npu.memory_reserved(),
                  peak_allocated=torch.npu.max_memory_allocated(), calls=calls))
    return hook


def make_route_hook(config):
    import torch
    path = Path(config['output']); counts = {}
    def hook(module, inputs, output):
        layer = module.layer_id
        ids = output.topk_ids.detach().cpu().contiguous()
        weights = output.topk_weights.detach().cpu().float().contiguous()
        if layer not in range(1,27) or ids.ndim != 2 or ids.shape[1] != 6 or ids.shape != weights.shape:
            raise ValueError('ROUTE_IDENTITY')
        if (ids < 0).any() or (ids >= 64).any() or not torch.isfinite(weights).all():
            raise ValueError('ROUTE_VALUE')
        sequence = counts.get(layer, 0)
        with path.open('a') as f:
            f.write(json.dumps(dict(layer=layer, sequence=sequence, ids=ids.tolist(), weights=weights.tolist()))+'\n')
        counts[layer] = sequence + 1
    return hook
