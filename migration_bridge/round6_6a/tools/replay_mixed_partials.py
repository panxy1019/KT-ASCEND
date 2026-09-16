"""Fresh mixed CPU and resident NPU replays of actual gain/loss operands.

No reference routing, no production modifications, no tolerance refit.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS


def main():
    import torch
    import torch_npu  # noqa:F401
    from safetensors import safe_open
    from kt_kernel import KTMoEWrapper
    from kt_kernel.utils.llamafile import LlamafileMoEWrapper
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast
    p=argparse.ArgumentParser()
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--gguf',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    spec=json.loads(a.inputs.read_text())
    index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    def digest(x):
        return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def metric(x,y):
        d=x.double()-y.double()
        return dict(exact=torch.equal(x,y),max_abs=float(d.abs().max()),
                    relative_l2=float(d.norm()/y.double().norm().clamp_min(1e-30)),
                    finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()))
    def load(layer,src,name,call):
        return torch.load(a.trace/f'{src}_layer{layer:02d}_{name}_call{call:04d}.pt',
                          map_location='cpu',weights_only=False)['tensor']
    report=dict(status='RUNNING',scope='DIAGNOSTIC_NOT_QUALIFICATION',rows=[],
                workers=32,threadpool_count=1,buffer_capacity=2048)
    resident=list(range(0,64,2))
    method=NS(with_bias=False,moe_runner_config=NS(activation='silu'),_kt_return_fp32_contribution=True)
    try:
        for layer in (1,9,17,26):
            LlamafileMoEWrapper._gguf_loader_instance=None
            LlamafileMoEWrapper._gguf_loader_path=None
            mask=torch.tensor([e%2==0 for e in range(64)],dtype=torch.bool)
            cpu=KTMoEWrapper(layer_idx=layer,num_experts=64,num_experts_per_tok=6,hidden_size=2048,
                moe_intermediate_size=1408,gpu_experts_mask=mask,cpuinfer_threads=32,threadpool_count=1,
                weight_path=str(a.gguf),chunked_prefill_size=2048,max_deferred_experts_per_token=0,
                method='LLAMAFILE',numa_nodes=[0])
            cpu.load_weights(torch.arange(64,dtype=torch.int32))
            gateup=[];down=[];weight_hashes=[]
            for expert in resident:
                weights={}
                for name in ('gate','up','down'):
                    key=f'model.layers.{layer}.mlp.experts.{expert}.{name}_proj.weight'
                    with safe_open(a.model/index[key],framework='pt',device='cpu') as sf:
                        weights[name]=sf.get_tensor(key).clone()
                weight_hashes.append(dict(expert=expert,sha256={k:digest(v) for k,v in weights.items()}))
                gateup.append(torch.cat([weights['gate'],weights['up']]))
                down.append(weights['down'])
            full=NS(w13_weight=npu_format_cast(torch.stack(gateup).to('npu').transpose(1,2)),
                    w2_weight=npu_format_cast(torch.stack(down).to('npu').transpose(1,2)),top_k=6)
            del gateup,down,weights
            for i,r in enumerate(spec['requests']):
                call=i*(spec['token_budget']+1)+r['original_first_difference']+1
                x=load(layer,'cpu','input_hidden_after_sync',call).contiguous()
                ids=load(layer,'model','router_topk_ids',call).long().contiguous()
                weights=load(layer,'model','router_weights',call).float().contiguous()
                pc=load(layer,'ktep','cpu_partial',call).float()
                pn=load(layer,'ktep','npu_owned_partial',call).float()
                merged=load(layer,'ktep','routed_merge',call)
                assert 0<len(x)<=2048 and ids.shape==weights.shape==(len(x),6)
                assert x.dtype==torch.bfloat16 and pc.dtype==pn.dtype==torch.float32
                npu_owned=ids.remainder(2)==0
                physical=torch.where(npu_owned,ids//2,0).to('npu',dtype=torch.int32)
                nw=torch.where(npu_owned,weights,0).to('npu')
                xn=x.to('npu')
                ch=[];nh=[]
                for _ in range(10):
                    c=cpu.forward(x,ids,weights).clone().float()
                    n=UnquantizedFusedMoEMethod.forward_npu(method,full,
                        NS(hidden_states=xn,topk_output=(nw,physical,None))).hidden_states.cpu().float()
                    ch.append(digest(c));nh.append(digest(n))
                row=dict(layer=layer,quality_id=r['quality_id'],state=r['state'],call=call,qlen=len(x),
                    input_sha256=digest(x),ids_sha256=digest(ids),weights_sha256=digest(weights),
                    cpu_replay_vs_capture=metric(c,pc),npu_replay_vs_capture=metric(n,pn),
                    replay_merge_vs_capture=metric((c+n).to(merged.dtype),merged),
                    cpu_repeat_hashes=ch,npu_repeat_hashes=nh,
                    deterministic=len(set(ch))==len(set(nh))==1,
                    resident_weight_hashes=weight_hashes)
                report['rows'].append(row)
                (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('resident_weight_hashes','cpu_repeat_hashes','npu_repeat_hashes')}),flush=True)
                if not row['deterministic'] or not all(row[k]['finite'] for k in
                    ('cpu_replay_vs_capture','npu_replay_vs_capture','replay_merge_vs_capture')):
                    raise ValueError('REPLAY_NONDETERMINISM_OR_NONFINITE')
            del cpu,full,xn,physical,nw
            torch.npu.empty_cache()
        exact=all(r[k]['exact'] for r in report['rows'] for k in
                  ('cpu_replay_vs_capture','npu_replay_vs_capture','replay_merge_vs_capture'))
        report['status']='ALL_MIXED_PARTIALS_EXACT' if exact else 'PARTIAL_REPLAY_DIFFERENCE_REQUIRES_REVIEW'
    except BaseException as exc:
        report.update(status='ERROR',error=repr(exc));raise
    finally:
        (a.output/'mixed_partials.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
