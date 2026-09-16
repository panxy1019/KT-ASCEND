"""Pre-QD operator diagnostic. Production LLAMA stages are NOT emulated captures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.nn.functional as F
import torch_npu  # noqa
from safetensors import safe_open
from kt_kernel import KTMoEWrapper
from kt_kernel.utils.llamafile import LlamafileMoEWrapper
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.hardware_backend.npu.utils import npu_format_cast


def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def metrics(x, ref):
    x, ref = x.double().cpu(), ref.double().cpu()
    diff = x-ref
    norm = ref.norm().item()
    denom = x.norm().item()*norm
    return dict(max_abs=diff.abs().max().item(), mean_abs=diff.abs().mean().item(),
        relative_l2=diff.norm().item()/max(norm, 1e-30),
        cosine=max(-1., min(1., (x.flatten()@ref.flatten()).item()/denom)) if denom else None,
        exact=torch.equal(x,ref), finite=bool(torch.isfinite(x).all() and torch.isfinite(ref).all()))


def reference(x, w, boundary):
    def round_(t):
        return t.bfloat16().float() if boundary else t
    gate=round_(F.linear(x.float(),w['gate'].float()))
    up=round_(F.linear(x.float(),w['up'].float()))
    activation=F.silu(gate)
    mul=round_(activation*up)
    down=round_(F.linear(mul,w['down'].float()))
    return dict(gate=gate,up=up,activation=activation,mul=mul,down=down)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--gguf',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage-capture',action='store_true')
    p.add_argument('--baseline',type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    baseline={}
    if a.baseline:
        baseline={(r['layer'],r['expert'],r['call']):r for r in json.loads(a.baseline.read_text())['rows']}
    if a.stage_capture and len(baseline)!=512:raise ValueError('STAGE_CAPTURE_REQUIRES_FROZEN_512_ROW_BASELINE')
    report=dict(status='RUNNING',scope='PHASE_MINUS_ONE_DIAGNOSTIC',rows=[],
        production_intermediates='NATIVE_BUFFERS' if a.stage_capture else 'NOT_CAPTURED_NOT_EMULATED',
        decode_calls=[2,33,55],prefill_calls=[1],model_qualification_allowed=False)
    def load(layer,source,stage,call):
        path=a.trace/f'{source}_layer{layer:02d}_{stage}_call{call:04d}.pt'
        return torch.load(path,map_location='cpu',weights_only=False)['tensor'].contiguous()
    method=NS(with_bias=False,moe_runner_config=NS(activation='silu'),_kt_return_fp32_contribution=True)
    fn=UnquantizedFusedMoEMethod.forward_npu
    try:
        for layer in [1,9,17,26]:
            mask=torch.tensor([e%2==0 for e in range(64)],dtype=torch.bool)
            LlamafileMoEWrapper._gguf_loader_instance=None
            LlamafileMoEWrapper._gguf_loader_path=None
            cpu=KTMoEWrapper(layer_idx=layer,num_experts=64,num_experts_per_tok=6,
                hidden_size=2048,moe_intermediate_size=1408,gpu_experts_mask=mask,
                cpuinfer_threads=32,threadpool_count=1,weight_path=str(a.gguf),
                chunked_prefill_size=16,max_deferred_experts_per_token=0,method='LLAMAFILE',numa_nodes=[0])
            cpu.load_weights(torch.arange(64,dtype=torch.int32))
            for expert in range(1,64,2):
                w={}
                for proj in ['gate','up','down']:
                    key=f'model.layers.{layer}.mlp.experts.{expert}.{proj}_proj.weight'
                    with safe_open(a.model/index[key],framework='pt',device='cpu') as sf:
                        w[proj]=sf.get_tensor(key).clone()
                    if w[proj].dtype!=torch.bfloat16:raise ValueError('WEIGHT_DTYPE')
                npu=NS(w13_weight=npu_format_cast(torch.cat([w['gate'],w['up']])[None].to('npu').transpose(1,2)),
                    w2_weight=npu_format_cast(w['down'][None].to('npu').transpose(1,2)),top_k=6)
                for call in [2,33,55,1]:
                    x=load(layer,'cpu','input_hidden_after_sync',call)
                    natural_ids=load(layer,'model','router_topk_ids',call)
                    natural_weights=load(layer,'model','router_weights',call).float()
                    if x.dtype!=torch.bfloat16 or x.shape[1]!=2048:raise ValueError('INPUT_CONTRACT')
                    if call!=1 and len(x)!=1:raise ValueError('DECODE_QLEN')
                    ids=torch.full((len(x),6),-1,dtype=torch.int64);ids[:,0]=expert
                    unit=torch.zeros((len(x),6),dtype=torch.float32);unit[:,0]=1
                    physical=ids.clone();physical[:,0]=0
                    stage_dir=a.output/f'native-L{layer}-E{expert}-C{call}'
                    if a.stage_capture:
                        stage_dir.mkdir();os.environ['KT_R66_STAGE_DIR']=str(stage_dir)
                    try:prod=cpu.forward(x,ids,unit).clone().float()
                    finally:os.environ.pop('KT_R66_STAGE_DIR',None)
                    hashes=[digest(prod)]
                    for _ in range(9):hashes.append(digest(cpu.forward(x,ids,unit).clone()))
                    xn=x.to('npu');pn=physical.to('npu',dtype=torch.int32);un=unit.to('npu')
                    refn=fn(method,npu,NS(hidden_states=xn,topk_output=(un,pn,None))).hidden_states.cpu().float()
                    n_hashes=[digest(refn)]
                    for _ in range(9):
                        n_hashes.append(digest(fn(method,npu,NS(hidden_states=xn,topk_output=(un,pn,None))).hidden_states))
                    pure=reference(x,w,False);bf=reference(x,w,True)
                    stages={}
                    if a.stage_capture:
                        prior=baseline[layer,expert,call]
                        if digest(prod)!=prior['production_hashes'][0] or digest(refn)!=prior['npu_hashes'][0]:
                            raise ValueError('DIAGNOSTIC_BUILD_OUTPUT_DIFFERS_FROM_PRODUCTION')
                        import numpy as np
                        for name,width in [('gate_raw',1408),('up_raw',1408),('mul_visible',1408),('down_raw',2048)]:
                            raw=torch.from_numpy(np.fromfile(stage_dir/f'E{expert}-{name}.f32',dtype=np.float32).copy()).reshape(len(x),width)
                            stages[name]=raw
                        captured={};original=torch.ops.npu.npu_grouped_matmul
                        def capture_gmm(*args,**kwargs):
                            out=original(*args,**kwargs)
                            if 'expanded_input' not in captured:
                                captured['expanded_input']=kwargs['x'][0].cpu().clone()
                                captured['gate_up']=out[0].cpu().clone()
                            else:
                                captured['mul']=kwargs['x'][0].cpu().clone()
                                captured['down']=out[0].cpu().clone()
                            return out
                        torch.ops.npu.npu_grouped_matmul=capture_gmm
                        try:observed=fn(method,npu,NS(hidden_states=xn,topk_output=(un,pn,None))).hidden_states.cpu()
                        finally:torch.ops.npu.npu_grouped_matmul=original
                        if not torch.equal(observed,refn):raise ValueError('NPU_OBSERVER_PERTURBATION')
                        selection=[]
                        for original_row in x:
                            matching=(captured['expanded_input']==original_row).all(1).nonzero().flatten()
                            if not len(matching):raise ValueError('NPU_EXPANDED_OPERAND_MAPPING')
                            for name in ['gate_up','mul','down']:
                                values=captured[name][matching]
                                if not torch.equal(values,values[:1].expand_as(values)):
                                    raise ValueError('NPU_DUPLICATE_OPERAND_OUTPUT_MISMATCH')
                            selection.append(int(matching[0]))
                        stages['npu_gate']=captured['gate_up'][selection,:1408].float()
                        stages['npu_up']=captured['gate_up'][selection,1408:].float()
                        stages['npu_mul']=captured['mul'][selection].float()
                        stages['npu_down']=captured['down'][selection].float()
                    natural=((natural_ids==expert)*natural_weights.bfloat16().float()).sum(1,keepdim=True)
                    comparisons={
                        'production_vs_canonical_fp32':metrics(prod,pure['down']),
                        'npu_vs_canonical_fp32':metrics(refn,pure['down']),
                        'production_vs_npu':metrics(prod,refn),
                        'production_vs_canonical_bf16_boundaries':metrics(prod,bf['down']),
                        'boundary_reference_vs_npu':metrics(bf['down'],refn),
                        'final_only_cast_vs_npu':metrics(pure['down'].bfloat16().float(),refn),
                        'natural_weighted_production_vs_npu':metrics(prod*natural,refn*natural)}
                    row=dict(layer=layer,expert=expert,call=call,qlen=len(x),input_sha256=digest(x),
                        weight_hashes={k:dict(bf16=digest(v),fp32=digest(v.float()),shape=list(v.shape)) for k,v in w.items()},
                        naturally_routed_rows=(natural_ids==expert).any(1).tolist(),
                        natural_route_weight=natural.tolist(),comparisons=comparisons,
                        production_hashes=hashes,npu_hashes=n_hashes,
                        deterministic=len(set(hashes))==len(set(n_hashes))==1)
                    if a.stage_capture:
                        row['stage_comparisons']={}
                        for name in ['gate','up','mul','down']:
                            visible=stages['mul_visible'] if name=='mul' else stages[name+'_raw'].bfloat16().float()
                            row['stage_comparisons'][name]=dict(
                                production_vs_fp32=metrics(visible,pure[name]),
                                production_vs_npu=metrics(visible,stages['npu_'+name]),
                                npu_vs_fp32=metrics(stages['npu_'+name],pure[name]))
                        row['production_observer_parity']=True
                    row['relative_error_ratio']=comparisons['production_vs_canonical_fp32']['relative_l2']/max(comparisons['npu_vs_canonical_fp32']['relative_l2'],1e-30)
                    row['local_safety_pass']=all(comparisons[k]['relative_l2']<=.01 for k in ['production_vs_canonical_fp32','production_vs_npu'])
                    row['finite']=all(v['finite'] for v in comparisons.values())
                    artifact=a.output/f'L{layer:02d}-E{expert:02d}-C{call:04d}.pt'
                    torch.save(dict(hidden=x,production=prod,npu=refn,canonical_fp32=pure,
                        canonical_bf16_boundaries=bf,natural_route_weight=natural,actual_stages=stages),artifact)
                    row['artifact_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
                    report['rows'].append(row)
                    with (a.output/'rows.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
                    print(json.dumps(dict(layer=layer,expert=expert,call=call,pass_=row['local_safety_pass'],
                        prod_cpu=comparisons['production_vs_canonical_fp32']['relative_l2'],
                        prod_npu=comparisons['production_vs_npu']['relative_l2'])),flush=True)
                    if not row['finite'] or not row['deterministic']:raise ValueError('FINITE_OR_DETERMINISM')
                    if not row['local_safety_pass']:raise ValueError('BLOCKED_CPU_BACKEND_NUMERICAL')
                del npu
            del cpu
            torch.npu.empty_cache()
        report['status']='OUTPUT_SAFETY_SCREEN_PASS_STAGE_AND_BUCKET_COVERAGE_PENDING'
    except BaseException as exc:
        report.update(status='BLOCKED',error=repr(exc));raise
    finally:
        (a.output/'three_way.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
