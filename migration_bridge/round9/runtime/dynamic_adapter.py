"""Concrete original-BF16 -> resident tensor and KTEP mask publication adapter.

CPU tensor path is host-testable. NPU path is explicitly opt-in for future gated
probes; it has no resource admission, request scheduler or native qualification
of its own. Never call outside PlacementController's drained request boundary.
"""

class TorchKTEPAdapter:
    def __init__(self,layers,methods,source,*,complete_cpu_backing_ids,host_staging_limit,
                 allow_unqualified_npu_probe=False):
        import torch
        ids=tuple(complete_cpu_backing_ids)
        if any(type(i) is not int for i in ids) or ids!=tuple(range(source.experts)):raise ValueError('demotion requires complete original-logical CPU backing')
        self.expert_bytes=source.hidden*source.intermediate*3*2
        # raw reads + owned CPU storage + concatenation; no complete expert layer.
        if type(host_staging_limit) is not int or host_staging_limit<4*self.expert_bytes:
            raise ValueError('host staging budget must cover four expert payloads')
        if not layers or set(layers)!=set(methods):raise ValueError('layer/method binding mismatch')
        self.layers=dict(layers);self.methods=dict(methods);self.source=source
        self.copied_bytes=0;self.devices=set()
        for index,layer in self.layers.items():
            if type(index) is not int or not 0<=index<source.layers:raise ValueError('invalid bound layer')
            method=self.methods[index];resident=len(method.gpu_index_to_logical)
            expected={'w13_weight':(resident,source.hidden,2*source.intermediate),
                      'w2_weight':(resident,source.intermediate,source.hidden)}
            for name,shape in expected.items():
                tensor=getattr(layer,name)
                if tensor.dtype!=torch.bfloat16 or tuple(tensor.shape)!=shape:raise ValueError('expected processed transposed BF16 resident weights')
                if tensor.device.type not in ('cpu','npu'):raise ValueError('unsupported device')
                if tensor.device.type=='npu' and not allow_unqualified_npu_probe:
                    raise RuntimeError('DEVICE_TEST_PENDING_NPU_PROMOTION_REQUIRES_SEPARATE_GUARDED_PROBE')
                self.devices.add(tensor.device)
            for owner,name in ((method,'gpu_experts_mask'),(method,'gpu_experts_mask_cuda'),(method.wrapper,'gpu_experts_mask')):
                value=getattr(owner,name)
                if value.dtype!=torch.bool or tuple(value.shape)!=(source.experts,):raise ValueError('invalid bound mask')
            for name in ('logical_to_gpu_index','logical_to_gpu_index_cuda'):
                value=getattr(method,name)
                if value.dtype!=torch.int32 or tuple(value.shape)!=(source.experts,):raise ValueError('invalid bound logical map')
            if method.gpu_index_to_logical.dtype!=torch.int32:raise ValueError('invalid reverse map dtype')
        if len(self.devices)!=1:raise ValueError('TP1 adapter requires one device')

    def synchronize(self):
        import torch
        for device in self.devices:
            if device.type=='npu':torch.npu.synchronize(device)

    def copy_expert(self,layer,slot,logical):
        import torch
        target=self.layers[layer]
        if type(slot) is not int or not 0<=slot<target.w13_weight.shape[0]:raise ValueError('invalid destination slot')
        def read(name,shape):
            raw=self.source.read(layer,logical,name)
            result=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(shape)
            if not bool(torch.isfinite(result).all()):raise ValueError('nonfinite original expert')
            return result
        h=self.source.hidden;i=self.source.intermediate
        gate=read('gate',(i,h));up=read('up',(i,h));down=read('down',(h,i))
        gu=torch.cat((gate,up),dim=0)
        with torch.no_grad():
            target.w13_weight[slot].copy_(gu.T,non_blocking=False)
            target.w2_weight[slot].copy_(down.T,non_blocking=False)
        self.copied_bytes+=self.expert_bytes

    def publish(self,layer,slots,generation):
        import torch
        method=self.methods[layer];slots=tuple(slots)
        if len(slots)!=len(method.gpu_index_to_logical) or len(set(slots))!=len(slots) or any(type(i) is not int or not 0<=i<self.source.experts for i in slots):
            raise ValueError('invalid published placement')
        selected=torch.tensor(slots,dtype=torch.int32,device='cpu')
        mask=torch.zeros(self.source.experts,dtype=torch.bool,device='cpu');mask[selected.long()]=True
        mapping=torch.full((self.source.experts,),-1,dtype=torch.int32,device='cpu');mapping[selected.long()]=torch.arange(len(slots),dtype=torch.int32)
        with torch.no_grad():
            # The native CPU kernel retains wrapper mask's pinned pointer.
            method.wrapper.gpu_experts_mask.copy_(mask)
            method.gpu_experts_mask.copy_(mask)
            method.gpu_experts_mask_cuda.copy_(mask)
            method.logical_to_gpu_index.copy_(mapping)
            method.logical_to_gpu_index_cuda.copy_(mapping)
            method.gpu_index_to_logical.copy_(selected)
        method._round9_generation=generation


def bind_controller(adapter):
    """Derive initial slots from real KTEP maps; reject inconsistent startup state."""
    import torch
    from .dynamic import PlacementController
    residents={};counts=set()
    for layer,method in adapter.methods.items():
        if getattr(method,'_round9_generation',0)!=0:raise ValueError('controller rebind requires a freshly loaded generation0 model')
        slots=tuple(method.gpu_index_to_logical.cpu().tolist())
        if len(set(slots))!=len(slots) or any(type(i) is not int or not 0<=i<adapter.source.experts for i in slots):raise ValueError('invalid initial resident IDs')
        mask=torch.zeros(adapter.source.experts,dtype=torch.bool)
        mapping=torch.full((adapter.source.experts,),-1,dtype=torch.int32)
        for slot,logical in enumerate(slots):mask[logical]=True;mapping[logical]=slot
        for actual in (method.gpu_experts_mask,method.gpu_experts_mask_cuda,method.wrapper.gpu_experts_mask):
            if not torch.equal(actual.cpu(),mask):raise ValueError('initial ownership mask mismatch')
        for actual in (method.logical_to_gpu_index,method.logical_to_gpu_index_cuda):
            if not torch.equal(actual.cpu(),mapping):raise ValueError('initial logical map mismatch')
        residents[layer]=slots;counts.add(len(slots))
    if len(counts)!=1:raise ValueError('all layers must use same resident budget')
    return PlacementController(residents,adapter,experts=adapter.source.experts,resident=counts.pop())
