"""Test-only hooks via the existing SGLang extension, no second coordinator."""
import json
from pathlib import Path
import time

METHODS = {}
CURRENT_FORWARD = {}


def install_forward_metadata(root):
    from sglang.srt.eplb.expert_distribution import _ExpertDistributionRecorderReal
    cls=_ExpertDistributionRecorderReal
    if getattr(cls,'_round7_audited',False): return
    start_record=cls.start_record
    start_pass=cls._on_forward_pass_start
    def begin(self):
        self._round7_epoch=getattr(self,'_round7_epoch',0)+1
        return start_record(self)
    def capture(self,batch):
        if self._recording and not (root/'observer-off').exists():
            CURRENT_FORWARD.clear()
            CURRENT_FORWARD.update(epoch=self._round7_epoch,
                forward_id=self._current_forward_pass_id.value,mode=str(batch.forward_mode),
                mode_name=batch.forward_mode.name,is_decode=batch.forward_mode.is_decode())
            append(root,'forward-metadata.jsonl',dict(epoch=self._round7_epoch,
                forward_id=self._current_forward_pass_id.value,
                mode=str(batch.forward_mode),
                mode_name=batch.forward_mode.name,is_decode=batch.forward_mode.is_decode(),
                input_ids=batch.input_ids.detach().cpu().tolist(),
                positions=batch.positions.detach().cpu().tolist()))
        return start_pass(self,batch)
    cls.start_record=begin;cls._on_forward_pass_start=capture;cls._round7_audited=True


def append(root, name, row):
    with (root/name).open('a') as f:
        f.write(json.dumps(dict(time_unix=time.time(),**row))+'\n')


def make_structure(config):
    import torch
    root = Path(config['output'])
    install_forward_metadata(root)
    expected = {r['layer']:r['npu_ids'] for r in json.loads(Path(config['profile']).read_text())['rows']}
    if config.get('dynamic_audit',False):
        from dynamic_audit import install
        install(root,expected)
    seen = set()
    def hook(module, inputs, output):
        if (root/'observer-off').exists(): return
        layer = getattr(module,'layer_id',None)
        if layer not in expected: return
        if config.get('deferred_audit',False) and layer==26:
            from kt_kernel.experts_base import BaseMoEWrapper
            assert not any(BaseMoEWrapper._layer_has_pending_deferred.values()),'PENDING_DEFERRED_AT_TERMINAL_OUTPUT'
        expert, method = module.experts, module.experts.quant_method
        METHODS[layer] = method
        if not torch.isfinite(output).all().item(): raise ValueError('NONFINITE_MLP')
        if layer in seen: return
        ids = torch.where(method.gpu_experts_mask)[0].tolist()
        if getattr(method,'_round7_generation',0)==0:
            assert ids == expected[layer], 'INITIAL_MASK_DRIFT'
        reference = torch.full((64,),-1,dtype=torch.int32)
        reference[ids] = torch.arange(len(ids),dtype=torch.int32)
        assert method.logical_to_gpu_index.dtype == torch.int32
        assert torch.equal(method.logical_to_gpu_index.cpu(),reference), 'MAPPING_DRIFT'
        assert torch.equal(method.gpu_experts_mask_cuda.cpu(),method.gpu_experts_mask), 'DEVICE_MASK_DRIFT'
        for name in ('w13_weight','w2_weight'):
            w = getattr(expert,name)
            assert w.dtype == torch.bfloat16 and w.shape[0] == len(ids), 'WEIGHT_LAYOUT_DRIFT'
        append(root,'structure.jsonl',dict(layer=layer,npu_ids=ids,cpu_ids=sorted(set(range(64))-set(ids)),
            resolved_deferred_budget=method.wrapper.max_deferred_experts_per_token,
            slots=len(ids),mapping_dtype=str(reference.dtype),allocated=torch.npu.memory_allocated(),
            reserved=torch.npu.memory_reserved()))
        seen.add(layer)
    return hook


def make_routes(config):
    import torch
    root = Path(config['output'])
    masks = {r['layer']:r['npu_ids'] for r in json.loads(Path(config['profile']).read_text())['rows']}
    def hook(module, inputs, output):
        if (root/'observer-off').exists(): return
        layer = module.layer_id
        ids = output.topk_ids.detach().cpu().long()
        weights = output.topk_weights.detach().cpu().float()
        assert ids.ndim == 2 and ids.shape[1] == 6 and ids.shape == weights.shape
        assert ((ids>=0)&(ids<64)).all() and torch.isfinite(weights).all()
        mask = METHODS[layer].gpu_experts_mask if layer in METHODS else torch.zeros(64,dtype=torch.bool)
        if layer not in METHODS: mask[masks[layer]] = True
        hit = mask[ids]
        initial_mask=torch.zeros(64,dtype=torch.bool);initial_mask[masks[layer]]=True
        initial_hit=initial_mask[ids]
        counts = torch.bincount(ids.flatten(),minlength=64).tolist()
        append(root,'routing.jsonl',dict(layer=layer,tokens=len(ids),assignments=ids.numel(),
            forward_context=dict(CURRENT_FORWARD),
            generation=getattr(METHODS.get(layer),'_round7_generation',0),
            initial_static_npu_hits=int(initial_hit.sum()),
            initial_static_cpu_weight=float(weights[~initial_hit].sum()),
            npu_hits=int(hit.sum()),cpu_hits=int((~hit).sum()),
            total_weight=float(weights.sum()),cpu_weight=float(weights[~hit].sum()),
            fully_npu_tokens=int(hit.all(dim=1).sum()),fully_npu_row_flags=hit.all(dim=1).tolist(),counts=counts))
    return hook
