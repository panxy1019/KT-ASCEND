"""Compact diagnostic TopK trace through SGLang's existing forward-hook API."""
import json
from pathlib import Path


def make_hook(config):
    path=Path(config['output']);counts={}
    if not str(path).startswith('/workspace/results/r65-'):raise ValueError('ROUTE_SCOPE')
    def hook(module,inputs,output):
        import torch
        layer=module.layer_id
        if layer not in range(1,27):raise ValueError('ROUTE_LAYER')
        ids=output.topk_ids.detach().cpu().contiguous()
        weights=output.topk_weights.detach().cpu().float().contiguous()
        if ids.ndim!=2 or ids.shape[1]!=6 or ids.shape!=weights.shape or (ids<0).any() or (ids>=64).any():
            raise ValueError('ROUTE_IDS_SHAPE')
        if not torch.isfinite(weights).all():raise ValueError('ROUTE_NONFINITE')
        sequence=counts.get(layer,0)
        with path.open('a') as f:f.write(json.dumps(dict(layer=layer,sequence=sequence,
            ids=ids.tolist(),weights=weights.tolist()))+'\n')
        counts[layer]=sequence+1
    return hook
