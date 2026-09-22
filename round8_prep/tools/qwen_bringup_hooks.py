"""Diagnostic-only structural/route assertions. Exclude from timed performance."""
import json
from pathlib import Path

_shared_parts={}


def make_shared_part(config):
    """Capture first-forward operands before the parent's in-place addition."""
    layer=config['layer'];kind=config['kind'];done=False
    def hook(module,inputs,output):
        nonlocal done
        if done:
            assert layer not in _shared_parts, 'SHARED_PART_CALLED_TWICE_IN_FIRST_FORWARD'
            return
        if isinstance(output,tuple):output=output[0]
        parts=_shared_parts.setdefault(layer,{})
        assert kind not in parts, 'SHARED_PART_DUPLICATE'
        parts[kind]=output.detach().clone()
        done=True
    return hook


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
        if config.get('shared_oracle',False):
            parts=_shared_parts.pop(layer)
            assert set(parts)=={'shared','gate','routed'}
            expected_output=parts['routed']+torch.sigmoid(parts['gate'])*parts['shared']
            assert torch.equal(output,expected_output), 'SHARED_EXACTLY_ONCE_ORACLE_MISMATCH'
        for name in ('w13_weight','w2_weight'):
            w=getattr(experts,name)
            assert w.dtype==torch.bfloat16 and w.shape[0]==len(row['npu_ids']), 'RESIDENT_LAYOUT_DRIFT'
        with (root/'structure.jsonl').open('a') as f:
            f.write(json.dumps(dict(layer=layer,slots=len(row['npu_ids']),npu_ids=ids,
                mapping_exact=True,finite=True,deferred=0,shared_present=True,
                shared_exactly_once_oracle=config.get('shared_oracle',False)))+'\n')
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
