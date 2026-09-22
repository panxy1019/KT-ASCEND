"""Same real operands, C32/C64/canonical/NPU; no corpus-driven tuning."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
import torch_npu
from safetensors import safe_open
from kt_kernel import KTMoEWrapper
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.hardware_backend.npu.utils import npu_format_cast

def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def error(x,y):
    return float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-30))

def main():
    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True)
    p.add_argument('--tp',type=int,choices=(1,2),required=True)
    p.add_argument('--numa-nodes',type=int,nargs=2,default=[4,5])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prefill',action='store_true')
    p.add_argument('--operands',type=Path,default=Path('/workspace/results/r65-r67-serial-v5/R16/observer/operands'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    operands=a.operands
    real=[torch.load(operands/f'L{a.layer:02d}-C{c}.pt',weights_only=False)['tensor'] for c in (1,4,7)]
    cases=[(f'real{idx}',x) for idx,x in enumerate(real)]
    if a.prefill:
        # Explicit diagnostic layout: real historical decode rows, not a claim
        # of captured contiguous prefill history. Actual prefill follows in canary.
        cases += [(f'prefill{n}',torch.cat([real[i%3] for i in range(n)])) for n in (2,8,9,10,11,16,32,64)]
    mask=torch.ones(64,dtype=torch.bool);mask[[1,63]]=False
    weight_expert=random.Random(67064+a.layer).choice((1,63))
    os.environ['KT_C64_CAPTURE_WEIGHT_EXPERT']=str(weight_expert)
    weights_dir=a.output/'weights';weights_dir.mkdir()
    os.environ['KT_C64_CAPTURE_DIR']=str(weights_dir)
    cpu=KTMoEWrapper(layer_idx=a.layer,num_experts=64,num_experts_per_tok=6,hidden_size=2048,
        moe_intermediate_size=1408,gpu_experts_mask=mask,cpuinfer_threads=32*a.tp,threadpool_count=a.tp,
        weight_path='/workspace/results/r65-assets-v1/all-moe-f32.gguf',chunked_prefill_size=64,
        max_deferred_experts_per_token=0,method='LLAMAFILE',numa_nodes=a.numa_nodes[:a.tp])
    cpu.load_weights(torch.arange(64,dtype=torch.int32))
    os.environ.pop('KT_C64_CAPTURE_WEIGHT_EXPERT')
    model=Path('/workspace/models/DeepSeek-V2-Lite-604d5664')
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    report=dict(status='RUNNING',layer=a.layer,tp=a.tp,rows=[],weight_identity=None,gate=.01,repeats=10,
                weight_identity_expert=weight_expert,
                prefill_scope='STACKED_REAL_DECODE_OPERANDS_FOR_KERNEL_BOUNDARY')
    def save(): (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
      for expert in (1,63):
        w={}
        for proj in ('gate','up','down'):
          key=f'model.layers.{a.layer}.mlp.experts.{expert}.{proj}_proj.weight'
          with safe_open(model/index[key],framework='pt',device='cpu') as f: w[proj]=f.get_tensor(key).clone()
        if a.tp==2 and expert==weight_expert:
          for proj in w:
            shape=(704,2048) if proj!='down' else (2048,704)
            parts=[torch.from_numpy(np.fromfile(weights_dir/f'tp{s}_{proj}.f32',dtype=np.float32).reshape(shape)) for s in (0,1)]
            full=torch.cat(parts,dim=0 if proj!='down' else 1)
            assert torch.equal(full,w[proj].float()),'NATIVE_WEIGHT_SLICE_IDENTITY'
          report['weight_identity']='EXACT'
        npu=NS(w13_weight=npu_format_cast(torch.cat([w['gate'],w['up']])[None].to('npu').transpose(1,2)),
               w2_weight=npu_format_cast(w['down'][None].to('npu').transpose(1,2)),top_k=6)
        method=NS(with_bias=False,moe_runner_config=NS(activation='silu'),_kt_return_fp32_contribution=True)
        for name,x in cases:
          q=x.shape[0];ids=torch.full((q,6),-1,dtype=torch.int64);ids[:,0]=expert
          route=torch.zeros(q,6);route[:,0]=1
          capture=a.output/f'E{expert}-{name}';capture.mkdir()
          os.environ['KT_C64_CAPTURE_DIR']=str(capture)
          y=None;hashes=[]
          for rep in range(10):
            out=cpu.forward(x,ids,route).cpu().clone();hashes.append(digest(out))
            if y is None:y=out
          assert len(set(hashes))==1,'NONDETERMINISM'
          physical=ids.int();physical[:,0]=0
          nref=UnquantizedFusedMoEMethod.forward_npu(method,npu,
              NS(hidden_states=x.to('npu'),topk_output=(route.to('npu'),physical.to('npu'),None))).hidden_states.cpu()
          gate=(x.float()@w['gate'].float().T).bfloat16().float()
          up=(x.float()@w['up'].float().T).bfloat16().float()
          mul=(torch.nn.functional.silu(gate)*up).bfloat16().float()
          canonical=(mul@w['down'].float().T).bfloat16().float()
          metrics=dict(vs_npu=error(y,nref),vs_canonical=error(y,canonical))
          if a.tp==2:
            c32=torch.load(a.output.parent/f'tp1-L{a.layer:02d}'/f'E{expert}-{name}.pt',weights_only=True)
            metrics['vs_c32']=error(y,c32)
          if a.tp==2:
            read=lambda f:torch.from_numpy(np.fromfile(capture/f,dtype=np.float32).reshape(q,2048))
            offline=read('partial0.f32')+read('partial1.f32')
            assert torch.equal(offline,read('precast.f32')),'PRECAST_MERGE'
            assert torch.equal(offline,read('visible.bin')) and torch.equal(offline,y),'VISIBLE_MERGE'
          assert bool(torch.isfinite(y).all()) and max(metrics.values())<=.01,'LOCAL_OPERATOR_GATE'
          os.environ.pop('KT_C64_CAPTURE_DIR',None)
          nohit=ids.clone();nohit[:,0]=0
          zero=cpu.forward(x,nohit,route).cpu().clone()
          assert torch.equal(zero,torch.zeros_like(zero)),'CPU_NOT_HIT'
          torch.save(y,a.output/f'E{expert}-{name}.pt')
          report['rows'].append(dict(expert=expert,case=name,qlen=q,metrics=metrics,
              input_sha256=digest(x),output_sha256=hashes[0],repeat_exact=True,merge_exact=a.tp==2,
              cpu_not_hit_exact=True))
          save(); print(json.dumps(report['rows'][-1]),flush=True)
      report['status']='PASS'
    except BaseException as e:
      report.update(status='FAIL',error=repr(e));raise
    finally:save()

if __name__=='__main__':main()
