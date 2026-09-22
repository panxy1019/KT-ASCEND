"""Two-case real-provider BF16 shadow probe; no model or HTTP request.

A uses captured decode input/routes, B changes the final expert to231. Neither
is a captured prefill workload. All-BF16 reference is independent of v20 mixed output.
"""
import argparse, ctypes, hashlib, json, pathlib, sys, time

def digest(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def derive_routes(ids):
    if ids!=[20,150,14,111,203,414,211,358,33,230]:raise ValueError('unexpected captured routes')
    return ids[:-1]+[231]

def validate_execution_peak(extra, projection, workspace):
    if extra<0 or extra>projection+workspace:raise ValueError('projection + workspace execution budget exceeded')

def compare_modes(off,on):
    for key in ('copy_calls','copied_logical_bytes'):
        if off[key]!=on[key]:raise ValueError('OFF/ON mismatch '+key)
    a=[(x['name'],x['output_sha256']) for x in off['cases']]
    b=[(x['name'],x['output_sha256']) for x in on['cases']]
    if a!=b:raise ValueError('OFF/ON output mismatch')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['off','on'],required=True)
    p.add_argument('--capture',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--source-manifest',type=pathlib.Path,required=True)
    p.add_argument('--source-manifest-sha256',required=True)
    p.add_argument('--off-result',type=pathlib.Path)
    a=p.parse_args()
    if (a.mode=='on')!=(a.off_result is not None):raise ValueError('ON requires frozen OFF result')
    if a.output.exists():raise FileExistsError(a.output)
    base=pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(base));sys.path.insert(0,str(base/'tools'))
    if digest(a.source_manifest)!=a.source_manifest_sha256:raise ValueError('source manifest identity')
    manifest=json.loads(a.source_manifest.read_text())
    for name,h in manifest.items():
        q=(base/name).resolve();q.relative_to(base.resolve())
        if digest(q)!=h:raise ValueError('frozen source changed '+name)
    from runtime.mainline_attribution import validate_capture
    if digest(a.capture)!='de82a710a2fd56c7587a91dbc5c37b8989ff0265a502391d797588c77c6e8fc6' or digest(a.capture.with_suffix('.json'))!='6724418fa2e5def0b27cb97be2708da13a5471c8ad02cd82d85d4d14dea6d2cc':raise ValueError('frozen capture changed')
    meta=json.loads(a.capture.with_suffix('.json').read_text())
    rev='de4b8e4d43b917e7706784d8bb445c9af86a3540'
    validate_capture(meta,dict(model_revision=rev,layer=0,layout='NZ29',cpu_precision='MXFP4',npu_precision='BF16',bank_sha256='e9d6b27f352d501a9ed3737ec5d3ab140b9d1f5bda22d2e36ef9ef0bcc45b08b'))
    if a.capture.stat().st_size>1048576 or digest(a.capture)!=meta['tensor_sha256']:raise ValueError('capture bytes')
    from probe_identity import verify_probe_identity
    model=pathlib.Path('/models')/rev
    identity=verify_probe_identity(model=model,receipt=str(model)+'.round9-support/destination-verification/verification-1789886190714568317.json',expected_model=model,expected_receipt_sha256='7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239',revision=rev,manifest_sha256='62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e')
    import torch
    fixture=torch.load(a.capture,map_location='cpu',weights_only=True)
    specs={'hidden_states':((1,2560),torch.bfloat16),'output':((1,2560),torch.bfloat16),'topk_ids':((1,10),torch.int32),'topk_weights':((1,10),torch.float32)}
    if set(fixture)!=set(specs):raise ValueError('tensor keys')
    for n,(shape,dtype) in specs.items():
        t=fixture[n]
        if tuple(t.shape)!=shape or t.dtype!=dtype or not bool(torch.isfinite(t).all()):raise ValueError('tensor contract '+n)
    ids_a=fixture['topk_ids'][0].tolist();ids_b=derive_routes(ids_a)
    if bool((fixture['topk_weights']<0).any()):raise ValueError('negative routes')
    import torch_npu
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from runtime.expert_source import OriginalExpertSource
    from runtime.framework_config import BufferIdentity
    from runtime.layerwise import AscendBF16TransportBackend,make_reservation,group_topk_routes
    from runtime.shadow_decomposition import Recorder,activate,phase
    loaded_unquant=pathlib.Path(sys.modules[UnquantizedFusedMoEMethod.__module__].__file__).resolve()
    if digest(loaded_unquant)!='56b46c170c8150afb437cb1bc0ea4453a456ca14f849170e7d0f27358cabcafb':raise ValueError('frozen UnquantizedFusedMoE provider changed')
    if torch.__version__!='2.9.0+cpu' or torch_npu.__version__!='2.9.0.post2':raise ValueError('frozen torch versions changed')
    torch.set_num_threads(8);torch_npu.npu.config.allow_internal_format=True
    torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(.04,0)
    limit=int(torch.npu.mem_get_info()[1]*.04)
    if limit>2631841873:raise ValueError('allocator exceeds v20 component cap')
    acl=ctypes.CDLL('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so');physical=ctypes.c_int32(-1)
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes=[ctypes.c_int32,ctypes.POINTER(ctypes.c_int32)]
    if acl.aclrtGetPhyDevIdByLogicDevId(0,ctypes.byref(physical))!=0 or physical.value!=15 or torch.npu.device_count()!=1:raise RuntimeError('physical device identity')
    if list(pathlib.Path('/dev').glob('davinci[0-9]*'))!=[pathlib.Path('/dev/davinci15')]:raise RuntimeError('device isolation')
    source=OriginalExpertSource(model)
    def original(expert):
        def read(name,shape):return torch.frombuffer(bytearray(source.read(0,expert,name)),dtype=torch.bfloat16).reshape(shape)
        return read('gate',(640,2560)),read('up',(640,2560)),read('down',(2560,640))
    def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    template=torch.nn.Module()
    for k,v in dict(top_k=10,num_experts=32,num_local_experts=32,num_gpu_experts=32,moe_ep_size=1,moe_ep_rank=0,moe_tp_size=1,moe_tp_rank=0,hidden_size=2560,intermediate_size_per_partition=640,num_fused_shared_experts=0).items():setattr(template,k,v)
    config=MoeRunnerConfig(num_experts=32,num_local_experts=32,hidden_size=2560,intermediate_size_per_partition=640,layer_id=0,top_k=10,params_dtype=torch.bfloat16)
    template.moe_runner_config=config;method=UnquantizedFusedMoEMethod(use_triton_kernels=False,use_flashinfer_trtllm_moe=False);template.quant_method=method
    with torch.device('npu:0'):method.create_weights(template,32,2560,640,torch.bfloat16)
    with torch.no_grad():
        for slot in range(32):
            gate,up,down=original(slot)
            template.w13_weight[slot].copy_(torch.cat((gate,up),dim=0).to('npu:0'));template.w2_weight[slot].copy_(down.to('npu:0'))
    method.process_weights_after_loading(template);method.create_moe_runner(template,config)
    template_hash={n:tsha(torch_npu.npu_format_cast(getattr(template,n),2)) for n in ['w13_weight','w2_weight']}
    torch.npu.synchronize();base_alloc=int(torch.npu.memory_allocated())
    reservation=make_reservation(capacity=32,hidden=2560,intermediate=640,max_group_rows=32,allocator_limit_bytes=limit,observed_base_peak_bytes=base_alloc,observed_workspace_bytes=512*1024**2)
    backend=AscendBF16TransportBackend(template,model,capacity=32,layers=48,experts=512,host_staging_limit=64*1024**2);backend.preallocate(reservation)
    recorder=Recorder(run_id='shadow-'+a.mode,device_factory=lambda:(torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)))
    class Proxy:
        def __init__(self,inner):self.inner=inner;self.calls=0
        @property
        def copied_bytes(self):return self.inner.copied_bytes
        def copy_experts(self,layer,moves):
            self.calls+=1
            with phase('prepare',layer=layer,moves=tuple(moves),generation=self.calls,source_revision=rev,precision='BF16',layout='NZ29'):
                return self.inner.copy_experts(layer,moves)
        def synchronize(self):return self.inner.synchronize()
    backend.adapter=Proxy(backend.adapter)
    tensor_identity={n:dict(pointer=getattr(backend.shadow,n).data_ptr(),shape=list(getattr(backend.shadow,n).shape),stride=list(getattr(backend.shadow,n).stride()),format=int(torch_npu.get_npu_format(getattr(backend.shadow,n)))) for n in ['w13_weight','w2_weight']}
    hidden=fixture['hidden_states'].to('npu:0');weights=fixture['topk_weights'].to('npu:0')
    torch.npu.synchronize();baseline=int(torch.npu.memory_allocated());torch.npu.reset_peak_memory_stats()
    outputs=[];case_info=[];report=dict(status='RUNNING',mode=a.mode,source_identity=identity,capture_manifest=meta,scope='CAPTURE_INPUT_DRIVEN_BF16_SHADOW_COMPONENT_NOT_CAPTURED_PREFILL',allocator_limit_bytes=limit,reservation=reservation.__dict__,persistent_tensor_identity=tensor_identity,initialization='receipt/sizes+template original0..31 load and hash+preallocate+inputs before A; references after A/B; no cold-cache claim',cases=case_info)
    try:
        with activate(recorder if a.mode=='on' else None):
            for index,(name,ids) in enumerate([('A',ids_a),('B',ids_b)],1):
                group=group_topk_routes([ids],capacity=32)[0];lease=BufferIdentity('shadow-component-'+name,0,index)
                recorder.context=dict(case=name,request_id=lease.request_id,placement_epoch=lease.placement_epoch,allocation_epoch=lease.allocation_epoch,shadow_lifetime_id='layer0-'+str(index))
                started=time.monotonic_ns()
                result=backend.execute_group(0,group,hidden,weights,lease)
                with phase('consumer_drain',case=name):backend.synchronize(lease)
                with phase('lease_release',case=name):backend.release(lease)
                outputs.append(backend.assemble(result,hidden));del result
                recorder.drain_events()
                case_info.append(dict(name=name,provenance='CAPTURE_INPUT_DRIVEN_SHADOW' if name=='A' else 'DERIVED_COMPONENT_CASE',ids=ids,logical_experts=group.logical_experts,wall_ns=time.monotonic_ns()-started,lease_returned=backend._current is None and backend._synchronized is None))
        execution_peak=int(torch.npu.max_memory_allocated())
        # References and layout identity extraction are after both measured cases.
        for row,actual in zip(case_info,outputs):
            expected=torch.zeros(1,2560,dtype=torch.float32);x=fixture['hidden_states'].float()
            for col,expert in enumerate(row['ids']):
                gate,up,down=(v.float() for v in original(expert))
                gv=(x@gate.T).bfloat16().float();uv=(x@up.T).bfloat16().float()
                activated=(torch.nn.functional.silu(gv)*uv).bfloat16().float()
                contribution=(activated@down.T).bfloat16().float()
                expected+=contribution*fixture['topk_weights'][0,col]
            want=expected.bfloat16();value=actual.cpu()
            error=float((value.float()-want.float()).double().norm()/want.float().double().norm().clamp_min(1e-30))
            if not bool(torch.isfinite(value).all()) or error>.01:raise AssertionError(('numeric',row['name'],error))
            row.update(relative_l2=error,tolerance=.01,output_sha256=tsha(value),reference_sha256=tsha(want))
        for n in ['w13_weight','w2_weight']:
            if tsha(torch_npu.npu_format_cast(getattr(template,n),2))!=template_hash[n]:raise AssertionError('template changed')
            if getattr(backend.shadow,n).data_ptr()!=tensor_identity[n]['pointer']:raise AssertionError('shadow reallocated')
        last=case_info[-1]['logical_experts']
        for n in ['w13_weight','w2_weight']:
            nd=torch_npu.npu_format_cast(getattr(backend.shadow,n),2).cpu()
            for slot,expert in enumerate(last):
                gate,up,down=original(expert);want=torch.cat((gate,up),dim=0).T if n=='w13_weight' else down.T
                if not torch.equal(nd[slot],want):raise AssertionError('shadow source mismatch')
            del nd
        del actual;outputs.clear();torch.npu.synchronize()
        after=int(torch.npu.memory_allocated())
        if backend.adapter.calls!=2 or backend._current is not None:raise AssertionError('unexpected prepare or lease')
        if after!=baseline:raise AssertionError(('live allocation not back to baseline',baseline,after))
        validate_execution_peak(execution_peak-baseline,reservation.max_projection_bytes,reservation.observed_workspace_bytes)
        report.update(status='COMPONENT_CASES_VALIDATED',copy_calls=backend.adapter.calls,copied_logical_bytes=backend.adapter.copied_bytes,execution_peak_extra_bytes=execution_peak-baseline,baseline_allocated_bytes=baseline,final_allocated_bytes=after,template_unchanged=True,final_shadow_exact=True,decomposition=recorder.report() if a.mode=='on' else {'status':'OBSERVATION_OFF'})
        if a.mode=='on':
            compare_modes(json.loads(a.off_result.read_text()),report)
            report['off_on']='EXACT_OUTPUT_AND_COPY_COUNTS_MATCH'
        if a.mode=='on' and report['decomposition']['status']!='VALID_HOST_DECOMPOSITION':raise AssertionError('incomplete observation')
    except BaseException as exc:
        report.update(status='FAIL',error=repr(exc));raise
    finally:
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        data=''.join(json.dumps(x)+'\n' for x in recorder.events)
        if len(data.encode())>8*1024**2:raise RuntimeError('trace budget')
        a.output.with_name('shadow-decomposition-events.jsonl').write_text(data)
    print(report['status'],flush=True)

if __name__=='__main__':main()
