"""Default-off update telemetry, no additional accelerator work or readbacks.

Use for untimed quality/diagnostics only. Exact source/destination/generation
validation belongs to the separate heavy migration auditor. Host call elapsed
is NOT a device-event measurement and is not formal benchmark evidence.
"""
import json
from pathlib import Path
import threading
import time


def event(layer,before,selected,copied_bytes,elapsed,components):
    if len(before)!=len(selected) or len(set(selected))!=len(selected):
        raise ValueError('TELEMETRY_RESIDENT_SLOT_DRIFT')
    return dict(status='COMPLETE',time_unix=time.time(),layer=layer,
        previous_resident=before,selected=selected,migration_bytes=copied_bytes,
        official_update_host_seconds=elapsed,component_host_seconds=components,
        scope='UNTIMED_HOST_CALL_TELEMETRY_NO_EXTRA_DEVICE_WORK_NOT_WEIGHT_CERTIFICATION')


def install(root):
    import sglang.srt.layers.moe.kt_ep_wrapper as module
    cls=module.KTEPWrapperMethod
    if getattr(cls,'_round7_light_telemetry',False):return
    active=threading.local()
    for name in ('select_top_experts_from_batch','copy_experts_weights_bf16',
                 'update_gpu_expert_mappings','update_kt_wrapper_masks'):
        original=getattr(module,name)
        def phase(*args,_name=name,_original=original,**kwargs):
            if not getattr(active,'recording',False):return _original(*args,**kwargs)
            started=time.perf_counter()
            try:return _original(*args,**kwargs)
            finally:
                active.components[_name]=active.components.get(_name,0.)+time.perf_counter()-started
        setattr(module,name,phase)
    original_update=cls._update_gpu_experts_from_batch
    def update(self,layer,ctx,dispatch_output):
        assert not getattr(active,'recording',False),'NESTED_DYNAMIC_TELEMETRY'
        before=self.gpu_experts_mask.nonzero().flatten().tolist()  # Already CPU.
        active.components={};active.recording=True
        started=time.perf_counter()
        try:
            result=original_update(self,layer,ctx,dispatch_output)
            elapsed=time.perf_counter()-started
            selected=self.gpu_index_to_logical.tolist()  # Already CPU.
            copied=sum(getattr(layer,name).numel()*getattr(layer,name).element_size()
                       for name in ('w13_weight','w2_weight'))
            row=event(self.kt_config.layer_idx,before,selected,copied,elapsed,active.components)
            with (Path(root)/'dynamic-telemetry.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            return result
        finally:active.recording=False
    cls._update_gpu_experts_from_batch=update;cls._round7_light_telemetry=True


def make_hook(config):
    install(Path(config['output']))
    def hook(module,inputs,output):pass
    return hook
