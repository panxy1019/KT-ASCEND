"""Diagnostic-only structural/route assertions. Exclude from timed performance."""
import json
from pathlib import Path


def make_structure(config):
    import torch
    root=Path(config['output'])
    expected={r['layer']:r for r in json.loads(Path(config['profile']).read_text())['layers']}
    seen=set()
    def hook(module, inputs, output):
        layer=module.layer_id
        assert torch.isfinite(output).all().item(), 'NONFINITE_MLP'
        if layer in seen:return
        experts=module.experts;method=experts.quant_method
        row=expected[layer]
        ids=torch.where(method.gpu_experts_mask)[0].tolist()
        assert ids==row['npu_ids'], 'MASK_DRIFT'
        mapping=torch.tensor(row['logical_to_physical'],dtype=torch.int32)
        assert method.logical_to_gpu_index.dtype==torch.int32
        assert torch.equal(method.logical_to_gpu_index.cpu(),mapping),'MAPPING_DRIFT'
        assert torch.equal(method.gpu_experts_mask_cuda.cpu(),method.gpu_experts_mask)
        assert method.wrapper.max_deferred_experts_per_token==0
        assert module.shared_expert is not None
        for name in ('w13_weight','w2_weight'):
            w=getattr(experts,name)
            assert w.dtype==torch.bfloat16 and w.shape[0]==64, 'RESIDENT_LAYOUT_DRIFT'
        with (root/'structure.jsonl').open('a') as f:
            f.write(json.dumps(dict(layer=layer,slots=64,npu_ids=ids,
                mapping_exact=True,finite=True,deferred=0,shared_present=True))+'\n')
        seen.add(layer)
    return hook


def make_routes(config):
    import torch
    root=Path(config['output'])
    rows={r['layer']:r for r in json.loads(Path(config['profile']).read_text())['layers']}
    def hook(module,inputs,output):
        ids=output.topk_ids.detach().cpu().long()
        w=output.topk_weights.detach().cpu().float()
        assert ids.ndim==2 and ids.shape[1]==8 and ids.shape==w.shape
        assert ((ids>=0)&(ids<256)).all() and torch.isfinite(w).all()
        mask=torch.zeros(256,dtype=torch.bool);mask[rows[module.layer_id]['npu_ids']]=True
        hit=mask[ids]
        with (root/'routing.jsonl').open('a') as f:
            f.write(json.dumps(dict(layer=module.layer_id,tokens=len(ids),
                cpu_hits=int((~hit).sum()),npu_hits=int(hit.sum()),finite=True))+'\n')
    return hook
