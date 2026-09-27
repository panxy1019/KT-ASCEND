"""Real QSA weights, synthetic text activations; local module gate only.

Frozen output rel-L2 gates: FP32 1e-4, BF16 .03. No paged-cache qualification.
"""
import argparse
import ctypes
import hashlib
import json
import math
import pathlib
import sys
import torch
from safetensors import safe_open

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=pathlib.Path,required=True)
p.add_argument('--identity-receipt',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
base=pathlib.Path(__file__).parents[1]
sys.path[:0]=[str(base),str(base/'tools')]
from probe_identity import verify_probe_identity
from runtime.qsa import compress_index_keys,select_tokens,sparse_gqa,text_rope
rev='de4b8e4d43b917e7706784d8bb445c9af86a3540'
identity=verify_probe_identity(model=a.model,receipt=a.identity_receipt,expected_model=pathlib.Path('/models')/rev,
 expected_receipt_sha256='7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239',revision=rev,
 manifest_sha256='62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e')
index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
def load(key):
 with safe_open(str(a.model/index[key]),framework='pt',device='cpu') as f:return f.get_tensor(key)
def error(x,y):
 return float(torch.linalg.vector_norm(x.double()-y.double())/torch.linalg.vector_norm(y.double()).clamp_min(1e-30))
def norm(x,w):
 v=x.float()
 return (v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-6)*(1+w.float())).to(x.dtype)
def run(x,w):
 n=x.shape[0];pos=torch.arange(n,device=x.device)
 projected=x@w['indexer.index_qk_proj.weight'].T
 iq=text_rope(norm(projected[:,:512].reshape(n,4,128),w['indexer.q_layernorm.weight']),pos)
 ck=compress_index_keys(projected[:,512:],w['indexer.k_layernorm.weight'],pos,text_rope)
 selected=select_tokens(iq,ck,pos)
 q,gate=(x@w['q_proj.weight'].T).reshape(n,24,512).chunk(2,-1)
 q=text_rope(norm(q,w['q_norm.weight']),pos)
 k=text_rope(norm((x@w['k_proj.weight'].T).reshape(n,2,256),w['k_norm.weight']),pos)
 v=(x@w['v_proj.weight'].T).reshape(n,2,256)
 out=sparse_gqa(q,k,v,selected)*gate.sigmoid()
 return out.flatten(1)@w['o_proj.weight'].T,selected,iq,ck
def canonical(x,w):
 # Independent float64 equations. At these short lengths all causal blocks fit
 # the index budget, so the exact selected SET is the full causal prefix.
 x=x.double();w={k:v.double() for k,v in w.items()};n=len(x)
 def gn(t,weight):return t*(t.square().mean(-1,keepdim=True)+1e-6).rsqrt()*(1+weight)
 def rope(t):
  angle=torch.arange(n,dtype=torch.float64)[:,None]/(10000000.0**(torch.arange(32,dtype=torch.float64)/32))
  c=angle.cos()[:,None];s=angle.sin()[:,None]
  return torch.cat((t[...,:32]*c-t[...,32:64]*s,t[...,32:64]*c+t[...,:32]*s,t[...,64:]),-1)
 q,g=(x@w['q_proj.weight'].T).reshape(n,24,512).chunk(2,-1)
 q=rope(gn(q,w['q_norm.weight']))
 k=rope(gn((x@w['k_proj.weight'].T).reshape(n,2,256),w['k_norm.weight']))
 v=(x@w['v_proj.weight'].T).reshape(n,2,256)
 output=[]
 for i in range(n):
  heads=[]
  for h in range(24):
   scores=(k[:i+1,h//12]*q[i,h]).sum(-1)/16
   heads.append((scores.softmax(0)[:,None]*v[:i+1,h//12]).sum(0))
  output.append(torch.stack(heads)*g[i].sigmoid())
 return torch.stack(output).flatten(1)@w['o_proj.weight'].T

def canonical_index(x,w):
 # Independently qualify the real index projection, normalization and pooled
 # keys: short causal output alone cannot detect a broken indexer.
 projected=x.double()@w['indexer.index_qk_proj.weight'].double().T
 def transform(v,weight,positions):
  v=v*(v.square().mean(-1,keepdim=True)+1e-6).rsqrt()*(1+weight.double())
  angle=positions.double()[:,None]/(10000000.0**(torch.arange(32,dtype=torch.float64)/32))
  c=angle.cos();s=angle.sin()
  if v.ndim==3:c=c[:,None];s=s[:,None]
  return torch.cat((v[...,:32]*c-v[...,32:64]*s,v[...,32:64]*c+v[...,:32]*s,v[...,64:]),-1)
 n=len(x);groups=n//4
 q=transform(projected[:,:512].reshape(n,4,128),w['indexer.q_layernorm.weight'],torch.arange(n))
 pooled=projected[:groups*4,512:].reshape(groups,4,128).mean(1)
 k=transform(pooled,w['indexer.k_layernorm.weight'],torch.arange(groups)*4)
 return q,k

torch.set_num_threads(2)
import torch_npu
torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
acl=ctypes.CDLL('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so');phy=ctypes.c_int32(-1)
acl.aclrtGetPhyDevIdByLogicDevId.argtypes=[ctypes.c_int32,ctypes.POINTER(ctypes.c_int32)]
assert acl.aclrtGetPhyDevIdByLogicDevId(0,ctypes.byref(phy))==0 and phy.value==15
assert torch.npu.device_count()==1
assert list(pathlib.Path('/dev').glob('davinci[0-9]*'))==[pathlib.Path('/dev/davinci15')]
report={'status':'RUNNING','scope':'QSA_LOCAL_REAL_WEIGHTS_SYNTHETIC_TEXT','source_identity':identity,
 'gates':{'float32':1e-4,'bfloat16':.03},'rows':[],'topk_stress':False}
names=['indexer.index_qk_proj.weight','indexer.q_layernorm.weight','indexer.k_layernorm.weight',
 'q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight','q_norm.weight','k_norm.weight']
try:
 with torch.no_grad():
  for layer in (3,23,47):
   weights={k:load(f'model.language_model.layers.{layer}.self_attn.{k}') for k in names}
   for dtype in (torch.float32,torch.bfloat16):
    cw={k:v.to(dtype) for k,v in weights.items()};dw={k:v.to('npu:0') for k,v in cw.items()}
    for length in (1,3,4,9,17):
     torch.manual_seed(15300+layer+length);x=torch.randn(length,2560).to(dtype);dx=x.to('npu:0')
     want=canonical(x,cw);want_iq,want_ck=canonical_index(x,cw);hashes=[]
     for repeat in range(3):
      out,selected,iq,ck=run(dx,dw);torch.npu.synchronize();cpu=out.cpu()
      assert torch.isfinite(cpu).all()
      hashes.append(hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())
     for i,row in enumerate(selected.cpu().tolist()):
      valid=[j for j in row if j>=0]
      assert sorted(valid)==list(range(i+1)) and row[:len(valid)]==valid
      assert row[len(valid):]==[-1]*(len(row)-len(valid))
     # Tokenwise index selection from successively completed compressed keys.
     for i in range(length):
      one=select_tokens(iq[i:i+1],ck[:(i+1)//4],torch.tensor([i],device='npu:0'))
      assert sorted(j for j in one.cpu()[0].tolist() if j>=0)==list(range(i+1))
     relative=error(cpu,want);gate=1e-4 if dtype==torch.float32 else .03
     iq_error=error(iq.cpu(),want_iq);ck_error=error(ck.cpu(),want_ck) if len(want_ck) else 0.0
     row={'layer':layer,'dtype':str(dtype),'tokens':length,'relative_l2':relative,
          'index_query_relative_l2':iq_error,'compressed_key_relative_l2':ck_error,'repeat_hashes':hashes}
     report['rows'].append(row)
     errors=(relative,iq_error,ck_error)
     assert all(math.isfinite(value) for value in errors) and max(errors)<=gate and len(set(hashes))==1,row
     del dx,out,selected,iq,ck,one,cpu
    del dw;torch.npu.synchronize();torch.npu.empty_cache()
  # Controlled unique increasing scores: only the highest 512 of 520 groups
  # survive, plus the 3 uncompressed tail tokens. No dense fallback can pass.
  q=torch.ones(1,4,128,device='npu:0')
  k=torch.arange(1,521,device='npu:0',dtype=torch.float32)[:,None].expand(520,128).contiguous()
  selected=select_tokens(q,k,torch.tensor([2082],device='npu:0')).cpu()[0].tolist()
  assert set(selected)==set(range(32,2083)) and len(selected)==2051
  report['topk_stress']=True
  del q,k
  # Signed heads distinguish sum(relu(dot)) from relu(sum(dot)).
  q=torch.tensor([[[1.,0.],[0.,1.]]],device='npu:0')
  k=torch.tensor([[4.,-3.],[2.,1.]],device='npu:0')
  selected=select_tokens(q,k,torch.tensor([7],device='npu:0'),topk_blocks=1).cpu()[0].tolist()
  assert selected==[0,1,2,3,-1,-1,-1]
  report['signed_multihead_ranking']=True
  del q,k
  torch.npu.synchronize();torch.npu.empty_cache()
 report.update(status='PASS_QSA_LOCAL_ONLY',peak_allocated=torch.npu.max_memory_allocated(),after_allocated=torch.npu.memory_allocated(),after_reserved=torch.npu.memory_reserved())
 assert report['after_allocated']==report['after_reserved']==0
except BaseException as exc:
 report.update(status='FAIL',error=repr(exc));raise
finally:a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({'status':report['status'],'rows':len(report['rows']),'peak_allocated':report['peak_allocated']}))
