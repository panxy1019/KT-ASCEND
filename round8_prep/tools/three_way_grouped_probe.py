"""Real frozen weights: production LLAMAFILE, production NPU forward, CPU oracle.
Synthetic operands, compact eight-expert NPU layout; not model quality proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import gguf
import torch
import torch_npu
from kt_kernel import KTMoEWrapper
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.hardware_backend.npu.utils import npu_format_cast

IDS=[0,1,31,63,127,191,254,255]

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def metric(a,b):
 a=a.double().cpu();b=b.double().cpu()
 return dict(maxabs=(a-b).abs().max().item(),relative_l2=((a-b).norm()/b.norm().clamp_min(1e-30)).item())

def main():
 p=argparse.ArgumentParser();p.add_argument('--layer',type=int,choices=[0,20,39],required=True)
 a=p.parse_args();root=Path('/workspace/round8');out=root/f'three-way-grouped-L{a.layer}-v1'
 out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.manual_seed(8675309)
 manifest=json.loads((root/'cpu-experts-f32-v1/manifest.json').read_text())
 assert manifest['status']=='FULL_40_LAYER_CONVERSION_IDENTITY_PASS_NOT_MODEL_QUALIFICATION'
 path=root/f'cpu-experts-f32-v1/layer-{a.layer:02}-f32.gguf'
 sha=digest(path)
 row=next(r for r in manifest['layers'] if r['layer']==a.layer)
 # Manifest schema is checked explicitly; no accepting a fresh unbound artifact.
 assert sha==row['sha256']
 reader=gguf.GGUFReader(str(path));weights={}
 for t in reader.tensors:
  name=t.name.split('ffn_')[1].split('_exps')[0]
  weights[name]=torch.from_numpy(t.data[IDS].copy())
 w13=torch.cat([weights['gate'],weights['up']],dim=1).bfloat16().npu()
 w2=weights['down'].bfloat16().npu()
 layer=SimpleNamespace(w13_weight=npu_format_cast(w13.transpose(1,2)),
                       w2_weight=npu_format_cast(w2.transpose(1,2)),top_k=8)
 method=SimpleNamespace(with_bias=False,moe_runner_config=SimpleNamespace(activation='silu'),
                        _kt_return_fp32_contribution=True)
 cpu=KTMoEWrapper(layer_idx=a.layer,num_experts=256,num_experts_per_tok=8,
  hidden_size=2048,moe_intermediate_size=512,gpu_experts_mask=torch.zeros(256,dtype=torch.bool),
  cpuinfer_threads=64,threadpool_count=2,numa_nodes=[4,5],weight_path=str(path),
  chunked_prefill_size=64,max_deferred_experts_per_token=0,method='LLAMAFILE')
 cpu.load_weights(torch.arange(256,dtype=torch.int32))
 report=dict(layer=a.layer,gguf_sha256=sha,logical_ids=IDS,seed=8675309,
  scope='REAL_WEIGHT_SYNTHETIC_OPERAND_PRODUCTION_FORWARD_COMPACT_NPU_LAYOUT',rows=[])
 cases=[(n,None) for n in (1,8,64)]+[(1,j) for j in range(8)]
 for qlen,only in cases:
  x=torch.randn(qlen,2048).bfloat16();ids=torch.tensor(IDS).expand(qlen,-1).contiguous()
  routes=torch.softmax(torch.randn(qlen,8),dim=-1)
  if only is not None:routes.zero_();routes[:,only]=1
  canonical=torch.zeros(qlen,2048)
  for j in range(8):
   g=(x.float()@weights['gate'][j].T).bfloat16().float()
   u=(x.float()@weights['up'][j].T).bfloat16().float()
   y=((torch.nn.functional.silu(g)*u).bfloat16().float()@weights['down'][j].T).bfloat16().float()
   canonical+=y*routes[:,j:j+1].bfloat16().float()
  c=[cpu.forward(x,ids,routes).detach().cpu().clone() for _ in range(10)]
  dispatch=SimpleNamespace(hidden_states=x.npu(),topk_output=(routes.npu(),
             torch.arange(8,dtype=torch.int32).expand(qlen,-1).contiguous().npu(),None))
  n=[UnquantizedFusedMoEMethod.forward_npu(method,layer,dispatch).hidden_states.cpu() for _ in range(3)]
  finite=all(torch.isfinite(y).all().item() for y in c+n+[canonical])
  exact=all(torch.equal(c[0],y) for y in c[1:]) and all(torch.equal(n[0],y) for y in n[1:])
  result=dict(qlen=qlen,only_logical_expert=None if only is None else IDS[only],
    finite=finite,repeat_exact=exact,cpu_vs_npu=metric(c[0],n[0]),
    cpu_vs_canonical=metric(c[0],canonical),npu_vs_canonical=metric(n[0],canonical))
  report['rows'].append(result)
  (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
  assert finite and exact and result['cpu_vs_npu']['relative_l2']<=.01 and result['cpu_vs_canonical']['relative_l2']<=.01, 'LOCAL_OPERATOR_GATE'
  print(a.layer,qlen,only,result['cpu_vs_npu']['relative_l2'],flush=True)
 report['status']='PASS';(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
