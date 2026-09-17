"""One-shot diagnostic via SGLang's existing --forward-hooks extension point."""
import json
from pathlib import Path


def make_hook(config):
    profile=json.loads(Path(config['profile']).read_text())
    expected={r['layer']:r for r in profile['rows']}
    path=Path(config['output']);seen=set()
    if not str(path).startswith('/workspace/c64-dev/'):raise ValueError('STRUCTURE_SCOPE')
    def hook(module,inputs,output):
        layer=getattr(module,'layer_id',None)
        if layer not in expected or layer in seen:return
        import torch
        expert=module.experts;method=expert.quant_method;row=expected[layer]
        if hasattr(method,'gpu_experts_mask'):
            mask=method.gpu_experts_mask
            if mask.device.type!='cpu' or mask.dtype!=torch.bool or tuple(mask.shape)!=(64,):raise ValueError('R65_MASK_BOUNDARY')
            npu_ids=torch.where(mask)[0].tolist()
            mapping=method.logical_to_gpu_index
            reference=torch.full((64,),-1,dtype=torch.int32)
            reference[npu_ids]=torch.arange(len(npu_ids),dtype=torch.int32)
            if mapping.dtype!=torch.int32 or not torch.equal(mapping.cpu(),reference):raise ValueError('R65_MAPPING_BOUNDARY')
        else:
            npu_ids=list(range(64));mapping=None
        if npu_ids!=row['npu_ids']:raise ValueError('R65_ACTUAL_OWNERSHIP')
        params={}
        for name in ('w13_weight','w2_weight'):
            tensor=getattr(expert,name)
            if tensor.shape[0]!=len(npu_ids) or tensor.dtype!=torch.bfloat16 or tensor.device.type!='npu':
                raise ValueError('R65_PHYSICAL_WEIGHT_SHAPE_DTYPE')
            params[name]=dict(shape=list(tensor.shape),dtype=str(tensor.dtype),device=str(tensor.device),
                              logical_bytes=tensor.numel()*tensor.element_size())
        data=dict(layer=layer,npu_ids=npu_ids,cpu_ids=row['cpu_ids'],weights=params,
                  mapping_dtype=str(mapping.dtype) if mapping is not None else 'N/A_ALL_NPU',
                  npu_allocated_bytes=torch.npu.memory_allocated(),
                  npu_reserved_bytes=torch.npu.memory_reserved(),
                  npu_peak_allocated_bytes=torch.npu.max_memory_allocated())
        with path.open('a') as f:f.write(json.dumps(data)+'\n')
        seen.add(layer)
    return hook
