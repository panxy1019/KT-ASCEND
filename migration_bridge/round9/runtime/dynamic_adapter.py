"""Concrete original-BF16 -> resident tensor and KTEP mask publication adapter.

CPU tensor path is host-testable. NPU path is explicitly opt-in for future gated
probes; it has no resource admission, request scheduler or native qualification
of its own. Never call outside PlacementController's drained request boundary.
"""

class TorchKTEPAdapter:
    def __init__(self,layers,methods,source,*,complete_cpu_backing_ids,host_staging_limit,
                 allow_unqualified_npu_probe=False,qualified_npu_formats=(2,)):
        import torch
        if not isinstance(qualified_npu_formats,tuple) or not qualified_npu_formats or any(type(f) is not int or f not in (2,29) for f in qualified_npu_formats):
            raise ValueError('explicit ND2/NZ29 qualification is required')
        self.qualified_npu_formats=qualified_npu_formats
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
                if tensor.device.type=='npu':
                    import torch_npu
                    actual_format=int(torch_npu.get_npu_format(tensor))
                    if actual_format not in qualified_npu_formats:
                        raise RuntimeError(f'DYNAMIC_NPU_LAYOUT_NOT_QUALIFIED actual={actual_format} admitted={qualified_npu_formats}')
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
            for destination,original in ((target.w13_weight,gu.T),(target.w2_weight,down.T)):
                fmt=2
                if destination.device.type=='npu':
                    import torch_npu
                    fmt=int(torch_npu.get_npu_format(destination))
                    if fmt not in self.qualified_npu_formats:raise RuntimeError('resident layout changed before migration')
                if fmt==29:
                    # Selecting an NZ slot can materialize an ND copy. Mutate the
                    # actual resident tensor rather than a detached slice result.
                    index=torch.tensor([slot],dtype=torch.int64,device=destination.device)
                    values=original.contiguous().unsqueeze(0).to(destination.device)
                    destination.index_copy_(0,index,values)
                    if int(torch_npu.get_npu_format(destination))!=29:raise RuntimeError('NZ publication changed resident layout')
                else:
                    destination[slot].copy_(original,non_blocking=False)
        self.copied_bytes+=self.expert_bytes

    def copy_experts(self,layer,moves):
        """Stage one projection of one layer; publish maps only after all copies.

        NZ index_copy_ can convert a whole resident projection. Do it once per
        layer/projection, while streaming original experts through the existing
        bounded host staging. Maximum additional ND device staging is400MiB.
        """
        import torch
        target=self.layers[layer];moves=tuple(moves)
        if not moves:return
        if (len({slot for slot,_ in moves})!=len(moves) or
            any(type(slot) is not int or not 0<=slot<target.w13_weight.shape[0] or
                type(logical) is not int or not 0<=logical<self.source.experts for slot,logical in moves)):
            raise ValueError('invalid layer migration batch')
        if target.w13_weight.device.type!='npu':
            for slot,logical in moves:self.copy_expert(layer,slot,logical)
            return
        import torch_npu
        formats=[int(torch_npu.get_npu_format(getattr(target,n))) for n in ('w13_weight','w2_weight')]
        if any(f not in self.qualified_npu_formats for f in formats):raise RuntimeError('resident layout changed before migration')
        if formats!=[29,29]:
            for slot,logical in moves:self.copy_expert(layer,slot,logical)
            return
        h=self.source.hidden;i=self.source.intermediate
        if len(moves)*h*2*i*2>400*1024**2:raise ValueError('layer device staging exceeds400MiB')
        def read(logical,name,shape):
            value=torch.frombuffer(bytearray(self.source.read(layer,logical,name)),dtype=torch.bfloat16).reshape(shape)
            if not bool(torch.isfinite(value).all()):raise ValueError('nonfinite original expert')
            return value
        with torch.no_grad():
            index=torch.tensor([slot for slot,_ in moves],dtype=torch.int64,device=target.w13_weight.device)
            for name in ('w13_weight','w2_weight'):
                destination=getattr(target,name)
                staged=torch.empty((len(moves),*destination.shape[1:]),dtype=torch.bfloat16,device=destination.device)
                if int(torch_npu.get_npu_format(staged))!=2:raise RuntimeError('expected ND staging tensor')
                for position,(_,logical) in enumerate(moves):
                    if name=='w13_weight':
                        original=torch.cat((read(logical,'gate',(i,h)),read(logical,'up',(i,h))),dim=0).T
                    else:original=read(logical,'down',(h,i)).T
                    staged[position].copy_(original,non_blocking=False)
                    del original
                destination.index_copy_(0,index,staged)
                self.synchronize()
                if int(torch_npu.get_npu_format(destination))!=29:raise RuntimeError('NZ publication changed resident layout')
                del staged
        self.copied_bytes+=len(moves)*self.expert_bytes

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
