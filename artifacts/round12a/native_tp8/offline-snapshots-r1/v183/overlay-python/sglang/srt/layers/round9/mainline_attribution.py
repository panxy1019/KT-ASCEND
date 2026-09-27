"""Default-OFF, sampled host/device durations and one real decode fixture.

No worker-time inference; device durations are NOT calibrated absolute timestamps.
Capture is explicitly intrusive and restricted to request r9-mainline-capture.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import time


def validate_capture(document, expected):
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError('capture identity mismatch: ' + key)
    if (document.get('request_id') != 'r9-mainline-capture'
            or document.get('intrusive_capture') is not True
            or type(document.get('selected_row')) is not int or document['selected_row'] != 0
            or type(document.get('original_physical_rows')) is not int
            or not 1 <= document['original_physical_rows'] <= 32
            or document.get('actual_weight_formats') != {'w13_weight':29,'w2_weight':29}):
        raise ValueError('capture provenance/padding mismatch')
    if (document.get('mode') != 'decode' or document.get('tokens') != 1
            or document.get('hidden') != 2560 or document.get('topk') != 10
            or document.get('resident_ids') != list(range(0, 512, 16))):
        raise ValueError('capture outside frozen Q32 layer0 decode contract')


def observe(original, sink, component, layer, context, device_factory=None, on_error=None):
    """Observer failures never skip a provider or replace its result/exception."""
    @functools.wraps(original)
    def call(*args, **kwargs):
        ctx = context()
        if ctx is None:
            return original(*args, **kwargs)
        errors=[]
        def note(error):
            errors.append(type(error).__name__+': '+str(error)[:512])
            if on_error is not None:
                try: on_error(errors[-1])
                except Exception: pass  # Preserve provider semantics even if the error sink fails.
        pair=None
        try:
            pair=device_factory() if device_factory is not None else None
            if pair is not None:pair[0].record()
        except Exception as error:
            note(error);pair=None
        started = time.monotonic_ns()
        outcome = 'error'
        try:
            value = original(*args, **kwargs)
            outcome = 'return'
            return value
        finally:
            ended = time.monotonic_ns()
            try:
                if pair is not None:pair[1].record()
            except Exception as error:
                note(error);pair=None
            try:
                sink(dict(component=component, layer=layer, start_ns=started,
                          end_ns=ended, duration_ns=ended-started, outcome=outcome,
                          device_ms=None, observer_errors=errors, **ctx,
                          **({'_device': pair} if pair else {})))
            except Exception as error:note(error)
    return call


def validate_scope(values):
    expected=dict(storage='MXFP4',placement='static',capacity=32,workers=128,
                  device='npu',revision='de4b8e4d43b917e7706784d8bb445c9af86a3540')
    if values != expected: raise ValueError('mainline observer outside frozen narrow scope')


def _bind(runner):
    from .candidate_config import load_candidate_profile
    candidate, _, _ = load_candidate_profile(os.environ.get('ROUND9_CANDIDATE_PROFILE'))
    validate_scope(dict(storage=candidate.storage,placement=candidate.placement,
                        capacity=runner.server_args.kt_num_gpu_experts,
                        workers=runner.server_args.kt_cpuinfer,device=runner.server_args.device,
                        revision=Path(runner.server_args.model_path).name))
    import torch
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
    state = {'context': None, 'spans': [], 'dropped': 0, 'captured': False,
             'bound': [], 'device_pairs': 0, 'observer_errors': [],
             'device_timing_broken': False}
    runner._round9_mainline = state
    def context(): return state['context']
    def sink(row):
        if len(state['spans']) < 128: state['spans'].append(row)
        else: state['dropped'] += 1
    def error_sink(error):
        state['device_timing_broken']=True
        if len(state['observer_errors'])<8:state['observer_errors'].append(error)
    def devices():
        if state['device_timing_broken']:return None
        if not context() or context()['request_id'] not in (
                'r9-mainline-profile', 'r9-mainline-chunk-profile', 'r9-mainline-capture'):
            return None
        if state['device_pairs'] >= 64: return None
        state['device_pairs'] += 1
        return torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    def wrap(obj, attribute, component, layer, device=True):
        original = getattr(obj, attribute)
        setattr(obj, attribute, observe(original, sink, component, layer,
                                        context, devices if device else None, error_sink))
        state['bound'].append(dict(component=component,layer=layer,
                                   provider=type(obj).__module__+'.'+type(obj).__name__,
                                   symbol=attribute))
    for name, module in runner.model.named_modules():
        # Sample full GDN0 / QSA3 / GR0+3, preserving all other layers unwrapped.
        import re
        match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', name)
        index = int(match.group(1)) if match else None
        if index not in (0, 3): continue
        kind = type(module).__name__
        if kind == 'Qwen4ReferenceGatedDeltaNet': wrap(module,'forward','GDN',index)
        elif kind == 'Qwen4ReferenceIndexer': wrap(module,'forward','QSA_INDEXER',index)
        elif kind == 'Qwen4ExpAttentionDecoderLayer':
            wrap(module,'self_attention','QSA_ATTENTION',index)
        elif kind == 'GatedResidual':
            wrap(module,'mix','GR_MIX',index);wrap(module,'combine','GR_COMBINE',index)
        method = getattr(module,'quant_method',None)
        if isinstance(method,KTEPWrapperMethod):
            wrap(method,'apply','ROUTED_EXPERT',index)
            wrap(method,'_submit_cpu_forward','CPU_SUBMIT',index)
            wrap(method,'_sync_cpu_forward','CPU_RESULT_WAIT',index)
            wrap(method.gpu_method,'apply','NPU_RESIDENT',index)
            if index == 0:
                original=method.apply
                def capture(layer, dispatch_output, _original=original, _method=method):
                    ctx=context()
                    take=(ctx is not None and ctx['request_id']=='r9-mainline-capture'
                          and ctx['mode']=='decode' and not state['captured'])
                    if not take: return _original(layer,dispatch_output)
                    state['captured']=True
                    x=dispatch_output.hidden_states
                    if x.ndim!=2 or not 1<=x.shape[0]<=32 or x.shape[1]!=2560:
                        raise RuntimeError('decode capture exceeds bounded workspace')
                    values={'hidden_states':x[:1].detach().cpu().clone(),
                            'topk_ids':dispatch_output.topk_output.topk_ids[:1].detach().cpu().clone(),
                            'topk_weights':dispatch_output.topk_output.topk_weights[:1].detach().cpu().clone()}
                    result=_original(layer,dispatch_output)
                    values['output']=result.hidden_states[:1].detach().cpu().clone()
                    if any(not bool(torch.isfinite(t).all()) for t in values.values()):
                        raise RuntimeError('nonfinite real expert capture')
                    import torch_npu
                    formats={name:int(torch_npu.get_npu_format(getattr(layer,name)))
                             for name in ('w13_weight','w2_weight')}
                    if set(formats.values())!={29}:raise RuntimeError('capture actual layout is not NZ29')
                    admission=json.loads(Path('/out/admission.json').read_text())
                    meta=dict(model_revision=Path(runner.server_args.model_path).name,
                              bank_sha256=admission['bank_identity']['layer0_sha256'],
                              layer=0,layout='NZ29',cpu_precision='MXFP4',npu_precision='BF16',
                              tokens=1,hidden=2560,topk=10,resident_ids=_method.gpu_index_to_logical.tolist(),
                              intrusive_capture=True,original_physical_rows=int(x.shape[0]),
                              selected_row=0,actual_weight_formats=formats,**ctx)
                    path=Path('/out/expert-replay-decode.pt')
                    if path.exists(): raise FileExistsError(path)
                    torch.save(values,path)
                    if path.stat().st_size>1048576:raise RuntimeError('capture byte budget exceeded')
                    meta['tensor_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                    path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
                    return result
                method.apply=capture
    Path('/out/mainline-bindings.json').write_text(json.dumps(state['bound'],indent=2)+'\n')
    return state


def before(scheduler,batch):
    if os.environ.get('ROUND9_MAINLINE_ENABLE') != '1': return
    runner=scheduler.tp_worker.model_runner
    state=getattr(runner,'_round9_mainline',None) or _bind(runner)
    if state['context'] is not None:raise RuntimeError('mainline observation batch still active')
    if len(batch.reqs)!=1:raise RuntimeError('mainline requires serial request')
    state['spans']=[];state['dropped']=0;state['device_pairs']=0;state['observer_errors']=[]
    identity=runner._round9_context_reference.identity
    if identity is None:raise RuntimeError('mainline missing allocation identity')
    state['context']=dict(request_id=batch.reqs[0].rid,placement_epoch=0,
                          allocation_epoch=identity.allocation_epoch,
                          mode='extend' if batch.forward_mode.is_extend() else
                          'decode' if batch.forward_mode.is_decode() else 'other')


def after(scheduler):
    if os.environ.get('ROUND9_MAINLINE_ENABLE') != '1': return
    state=getattr(scheduler.tp_worker.model_runner,'_round9_mainline',None)
    if state is None or state['context'] is None:return
    from .working_set_timeline import event
    for row in state['spans']:
        pair=row.pop('_device',None)
        row['device_clock']='NPU_EVENT_DURATION_ONLY'
        row['device_status']='NOT_SAMPLED'
        if pair:
            # Query only: never insert a device synchronization to obtain timing.
            try:
                if pair[1].query():
                    row['device_ms']=float(pair[0].elapsed_time(pair[1]))
                    row['device_status']='COMPLETE'
                else:row['device_status']='PENDING_AT_EXISTING_BOUNDARY'
            except Exception as error:
                row['device_status']='OBSERVER_ERROR';row['device_ms']=None
                row['observer_errors'].append(type(error).__name__+': '+str(error)[:512])
                state['device_timing_broken']=True
        event('mainline','span',**row)
    event('mainline','batch_coverage',sampled_layers=[0,3],dropped_spans=state['dropped'],
          cpu_worker_time='UNKNOWN',device_clock_calibration='NONE',
          observer_errors=state['observer_errors'],device_timing_disabled=state['device_timing_broken'],**state['context'])
    state['spans']=[];state['context']=None
