"""Real original10-expert CPU MoE gate; no NPU or full-model qualification."""
import argparse,ctypes,hashlib,json,pathlib,sys,time
import numpy as np
import torch
import kt_kernel_ext as native
from safetensors import safe_open
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]))
from runtime.lowbit_numpy import encode_blocks,decode_blocks
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]/'tools'))
from probe_identity import verify_probe_identity
p=argparse.ArgumentParser();p.add_argument('--model',type=pathlib.Path,required=True);p.add_argument('--receipt',type=pathlib.Path,required=True)
p.add_argument('--storage',choices=['Q8_0','MXFP4'],required=True);p.add_argument('--pools',type=int,choices=[2,4],required=True);p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--compare',type=pathlib.Path)
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2);torch.manual_seed(15901)
rev='de4b8e4d43b917e7706784d8bb445c9af86a3540'
identity=verify_probe_identity(model=a.model,receipt=a.receipt,expected_model=pathlib.Path('/models')/rev,expected_receipt_sha256='7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239',revision=rev,manifest_sha256='62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e')
assert native.llamafile_block32_tail_supported()
lib=ctypes.CDLL(native.__file__)
for symbol in ('quantize_row_q8_0','dequantize_row_q8_0'):
 getattr(lib,symbol).argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64]
def q8_activation(x):
 x=x.float().contiguous();raw=torch.empty(x.numel()//32*34,dtype=torch.uint8);result=torch.empty_like(x)
 lib.quantize_row_q8_0(x.data_ptr(),raw.data_ptr(),x.numel())
 lib.dequantize_row_q8_0(raw.data_ptr(),result.data_ptr(),x.numel())
 return result
index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map'];pre='model.language_model.layers.0.mlp.experts.'
original_ids=[0,1,2,3,4,507,508,509,510,511];parts=[]
for name in ('gate_up_proj','down_proj'):
 with safe_open(str(a.model/index[pre+name]),framework='pt',device='cpu') as f:
  view=f.get_slice(pre+name);parts.append(torch.stack([view[e] for e in original_ids]))
gate,up=parts[0].chunk(2,dim=1);original=[gate.contiguous(),up.contiguous(),parts[1]]
encoded=[];reference_weights=[]
for weight in original:
 raw=encode_blocks(weight.float().numpy(),a.storage)
 encoded.append(torch.from_numpy(raw))
 reference_weights.append(torch.from_numpy(decode_blocks(raw,a.storage)).reshape(weight.shape))
report=dict(status='RUNNING',storage=a.storage,pools=a.pools,cpu_workers=32*a.pools,source_identity=identity,
 source_experts=original_ids,source_projection_sha256=[hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest() for w in original],
 encoded_sha256=[hashlib.sha256(w.numpy().tobytes()).hexdigest() for w in encoded],native_sha256=hashlib.sha256(pathlib.Path(native.__file__).read_bytes()).hexdigest(),
 scope='REAL10_EXPERT_CPU_ONLY_QUANTIZED_WEIGHT_AND_ACTIVATION_REFERENCE',rel_l2_gate=.01,rows=[])
outputs={}
comparison=np.load(a.compare) if a.compare else None
try:
 pool=native.WorkerPoolConfig();pool.subpool_count=a.pools;pool.subpool_numa_map=[6,7] if a.pools==2 else [2,3,6,7];pool.subpool_thread_count=[32]*a.pools
 cpu=native.CPUInfer(pool);cfg=native.moe.MOEConfig(10,10,2560,640)
 mask=torch.zeros(10,dtype=torch.bool);cfg.gpu_experts_mask=mask.data_ptr();cfg.pool=cpu.backend_;cfg.layer_idx=0
 cfg.gate_type=cfg.up_type=cfg.down_type={'Q8_0':8,'MXFP4':39}[a.storage];cfg.hidden_type=30;cfg.output_type=0;cfg.bf16_numerical_compat=True
 cfg.m_block=32;cfg.group_min_len=10;cfg.group_max_len=32;cfg.max_len=32
 cfg.gate_proj,cfg.up_proj,cfg.down_proj=[w.data_ptr() for w in encoded]
 moe=native.moe.MOE(cfg);moe.load_weights()
 for q in (1,2,9,10,11,32):
  x=torch.randn(q,2560).bfloat16();qx=q8_activation(x);ids=torch.arange(10).repeat(q,1);routes=torch.softmax(torch.randn(q,10),dim=1).float();qlen=torch.tensor([q],dtype=torch.int32)
  expected=torch.zeros(q,2560)
  for e in range(10):
   g=(qx@reference_weights[0][e].T).bfloat16().float();u=(qx@reference_weights[1][e].T).bfloat16().float()
   hidden=(torch.nn.functional.silu(g)*u).bfloat16().float();qh=q8_activation(hidden)
   expected+=(qh@reference_weights[2][e].T)*routes[:,e,None].bfloat16().float()
  hashes=[]
  for repeat in range(3):
   y=torch.full_like(expected,float('nan'));moe.forward(qlen.data_ptr(),10,ids.data_ptr(),routes.data_ptr(),x.data_ptr(),y.data_ptr(),False)
   assert torch.isfinite(y).all();hashes.append(hashlib.sha256(y.numpy().tobytes()).hexdigest())
  outputs[str(q)]=y.numpy().copy()
  if comparison is not None:
   other=torch.from_numpy(comparison[str(q)]);cross_error=float((y-other).norm()/other.norm().clamp_min(1e-30))
   assert cross_error<=1e-5,('C64/C128',q,cross_error)
  error=float((y-expected).norm()/expected.norm().clamp_min(1e-30));assert error<=.01 and len(set(hashes))==1,(q,error)
  mask.fill_(True);y.fill_(float('nan'));moe.forward(qlen.data_ptr(),10,ids.data_ptr(),routes.data_ptr(),x.data_ptr(),y.data_ptr(),False);assert torch.count_nonzero(y)==0
  mask.zero_();moe.forward(qlen.data_ptr(),10,ids.data_ptr(),routes.data_ptr(),x.data_ptr(),y.data_ptr(),False);assert hashlib.sha256(y.numpy().tobytes()).hexdigest()==hashes[0]
  row=dict(tokens=q,relative_l2=error,cross_pool_relative_l2=cross_error if comparison is not None else None,repeat_hashes=hashes,ownership_flip_exact=True);report['rows'].append(row);print(json.dumps(row),flush=True)
 report['status']='PASS_REAL10_EXPERT_LOWBIT_CPU_MOE'
except BaseException as exc:report.update(status='FAIL',error=repr(exc));raise
finally:
 np.savez(a.output.with_suffix('.npz'),**outputs)
 a.output.write_text(json.dumps(report,indent=2)+'\n')
print(report['status'],flush=True)
