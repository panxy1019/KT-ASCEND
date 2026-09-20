"""Actual KTEP local merge with 64 original BF16 NPU experts and C128 CPU bank.

Module gate only: synthetic activations, no scheduler or whole model qualification.
"""
import argparse,ctypes,hashlib,json,pathlib,sys,faulthandler
faulthandler.enable();faulthandler.dump_traceback_later(60,repeat=True)
import torch
torch.set_num_threads(2)
import torch_npu
from safetensors import safe_open
from kt_kernel import kt_kernel_ext,get_current_device_stream_handle
from kt_kernel.utils.llamafile import LlamafileMoEWrapper
from kt_kernel.experts_base import BaseMoEWrapper,KExpertsCPUBuffer
p=argparse.ArgumentParser(description=__doc__)
for arg in ('model','identity-receipt','bank','output'):p.add_argument('--'+arg,type=pathlib.Path,required=True)
p.add_argument('--dynamic',action='store_true')
p.add_argument('--storage',choices=['BF16','Q8_0','MXFP4'],default='BF16')
p.add_argument('--bank-manifest',type=pathlib.Path)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]/'tools'))
from probe_identity import verify_probe_identity
rev='de4b8e4d43b917e7706784d8bb445c9af86a3540'
identity=verify_probe_identity(model=a.model,receipt=a.identity_receipt,expected_model=pathlib.Path('/models')/rev,
 expected_receipt_sha256='7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239',revision=rev,
 manifest_sha256='62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e')
h=hashlib.sha256()
with a.bank.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
if a.storage=='BF16':
 assert h.hexdigest()=='731679cb66e83c7c9a2b4b2ae29af68581729e194d8d495530f2aec1342bd97b'
else:
 manifest=json.loads(a.bank_manifest.read_text())
 assert manifest['sha256']==h.hexdigest() and manifest['file']==a.bank.name
 assert manifest['storage']==a.storage and manifest['layer']==0
 assert manifest['complete_cpu_backing'] and manifest['logical_ids']==list(range(512))
 assert manifest['geometry']==dict(experts=512,hidden=2560,intermediate=640)
 assert manifest['source_identity']['receipt_sha256']==identity['receipt_sha256']
 assert len(manifest['projection_identities'])==1536
 assert kt_kernel_ext.llamafile_block32_tail_supported()
assert kt_kernel_ext.llamafile_bf16_f32_tail_supported()
torch.npu.set_device(0)
torch.npu.set_per_process_memory_fraction(0.04,0)
acl=ctypes.CDLL('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so');phy=ctypes.c_int32(-1)
acl.aclrtGetPhyDevIdByLogicDevId.argtypes=[ctypes.c_int32,ctypes.POINTER(ctypes.c_int32)]
assert acl.aclrtGetPhyDevIdByLogicDevId(0,ctypes.byref(phy))==0 and phy.value==15 and torch.npu.device_count()==1
assert list(pathlib.Path('/dev').glob('davinci[0-9]*'))==[pathlib.Path('/dev/davinci15')]
from types import SimpleNamespace as NS
from unittest.mock import patch
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe import kt_ep_wrapper as ep
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder,set_global_expert_distribution_recorder
mask=torch.arange(512).remainder(8)==0
config=ep.KTConfig(layer_idx=0,gpu_experts_mask=mask,cpuinfer_threads=128,
 threadpool_count=4,numa_nodes=[2,3,6,7],weight_path=str(a.bank),chunked_prefill_size=32,
 max_deferred_experts_per_token=0,method='LLAMAFILE',num_layers=48)
# No distributed process group in this bounded module fixture: its TP rank is0.
# All weight creation, post-load, CPU submission, NPU kernels and merging are real.
with patch.object(ep,'get_tensor_model_parallel_rank',return_value=0):
 method=ep.KTEPWrapperMethod(UnquantizedFusedMoEMethod(),config)
layer=torch.nn.Module();layer.top_k=10;layer.num_experts=512
layer.intermediate_size_per_partition=640;layer.moe_tp_size=1
runner=MoeRunnerConfig(num_experts=512,num_local_experts=64,hidden_size=2560,
 intermediate_size_per_partition=640,layer_id=0,top_k=10,params_dtype=torch.bfloat16)
layer.moe_runner_config=runner
with torch.device('npu:0'):
 method.create_weights(layer,512,2560,640,torch.bfloat16)
index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
pre='model.language_model.layers.0.mlp.experts.'
def expert(e):
 with safe_open(str(a.model/index[pre+'gate_up_proj']),framework='pt',device='cpu') as f:gu=f.get_slice(pre+'gate_up_proj')[e].clone()
 with safe_open(str(a.model/index[pre+'down_proj']),framework='pt',device='cpu') as f:down=f.get_slice(pre+'down_proj')[e].clone()
 return gu,down
with torch.no_grad():
 for local,e in enumerate(torch.where(mask)[0].tolist()):
  gu,down=expert(e)
  layer.w13_weight[local].copy_(gu.to('npu:0'));layer.w2_weight[local].copy_(down.to('npu:0'))
 del gu,down
method.process_weights_after_loading(layer)
method.create_moe_runner(layer,runner)
weight_formats={name:int(torch_npu.get_npu_format(getattr(layer,name))) for name in ('w13_weight','w2_weight')}
print('ACTUAL_NPU_WEIGHT_FORMATS',weight_formats,flush=True)
# Verify every original resident expert after native ND->NZ conversion and transpose.
w13_nd=torch_npu.npu_format_cast(layer.w13_weight,2).cpu()
w2_nd=torch_npu.npu_format_cast(layer.w2_weight,2).cpu()
for local,e in enumerate(torch.where(mask)[0].tolist()):
 gu,down=expert(e)
 assert torch.equal(w13_nd[local].T,gu)
 assert torch.equal(w2_nd[local].T,down)
del gu,down,w13_nd,w2_nd
print('KTEP_REAL64_PROCESSED_WEIGHT_IDENTITY_PASS',flush=True)
dynamic_evidence=[]
if a.dynamic:
 sys.path.insert(0,str(pathlib.Path(__file__).parents[1]))
 from runtime.expert_source import OriginalExpertSource
 from runtime.dynamic_adapter import TorchKTEPAdapter,bind_controller
 source=OriginalExpertSource(a.model)
 adapter=TorchKTEPAdapter({0:layer},{0:method},source,
  complete_cpu_backing_ids=range(512),host_staging_limit=64*1024**2,
  allow_unqualified_npu_probe=True)
 controller=bind_controller(adapter)
 pinned_pointer=method.wrapper.gpu_experts_mask.data_ptr()
 for step,incoming in enumerate((511,0,511)):
  rid=f'bounded-migration-{step}';generation=controller.begin(rid)
  controller.observe(rid,generation,0,list(range(8,512,8))*2+[incoming]*100)
  try:controller.plan()
  except RuntimeError:pass
  else:raise AssertionError('migration admitted with live consumer')
  adapter.synchronize();controller.finish(rid,consumers_drained=True)
  plans=controller.plan()
  assert len(plans)==1 and len(plans[0].moves)==1
  controller.commit(plans)
  assert method.wrapper.gpu_experts_mask.data_ptr()==pinned_pointer
  assert controller.generation==step+1
  slots=method.gpu_index_to_logical.cpu().tolist()
  nd13=torch_npu.npu_format_cast(layer.w13_weight,2).cpu()
  nd2=torch_npu.npu_format_cast(layer.w2_weight,2).cpu()
  for slot,e in enumerate(slots):
   gu,down=expert(e)
   assert torch.equal(nd13[slot].T,gu) and torch.equal(nd2[slot].T,down),(step,slot,e)
  del nd13,nd2,gu,down
  dynamic_evidence.append(dict(generation=controller.generation,moves=plans[0].moves,
   all64_processed_experts_exact=True,pinned_mask_pointer_stable=True,copied_bytes=adapter.copied_bytes))
 mask=method.gpu_experts_mask.cpu().clone()
 assert bool(mask[511]) and not bool(mask[0])
 print('DYNAMIC_REAL_DEVICE_PROMOTE_DEMOTE_PROMOTE_PASS',flush=True)
weights={}
for e in sorted(set([1,7,63,127,511]+list(range(0,80,8))+[128,504])):
 gu,down=expert(e);gate,up=gu.float().chunk(2,0);weights[e]=(gate,up,down.float())
del gu,down
cpu_weights=weights
if a.storage!='BF16':
 import gguf
 from gguf.quants import dequantize
 reader=gguf.GGUFReader(str(a.bank));tensors={t.name:t for t in reader.tensors}
 cpu_weights={}
 for e in weights:
  pieces=[]
  for name in ('gate','up','down'):
   t=tensors[f'blk.0.ffn_{name}_exps.weight']
   assert int(t.tensor_type)=={'Q8_0':8,'MXFP4':39}[a.storage]
   raw=t.data.reshape(512,-1)[e]
   shape=(2560,640) if name=='down' else (640,2560)
   pieces.append(torch.from_numpy(dequantize(raw,t.tensor_type).copy()).reshape(shape))
  cpu_weights[e]=tuple(pieces)
 lib=ctypes.CDLL(kt_kernel_ext.__file__)
 for symbol in ('quantize_row_q8_0','dequantize_row_q8_0'):
  getattr(lib,symbol).argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64]
 def q8_activation(value):
  value=value.float().contiguous();raw=torch.empty(value.numel()//32*34,dtype=torch.uint8);result=torch.empty_like(value)
  lib.quantize_row_q8_0(value.data_ptr(),raw.data_ptr(),value.numel())
  lib.dequantize_row_q8_0(raw.data_ptr(),result.data_ptr(),value.numel())
  return result
captured={}
def capture(stage,tensor,**kwargs):
 if stage in ('cpu_partial','npu_owned_partial','routed_merge'):captured[stage]=tensor.detach().cpu().clone()
class Recorder:
 def on_gpu_expert_mask(self,idx,observed):assert idx==0 and torch.equal(observed.cpu(),mask)
previous=get_global_expert_distribution_recorder();set_global_expert_distribution_recorder(Recorder())
report={'status':'RUNNING','scope':'REAL_KTEP_LOCAL_MERGE_NOT_FULL_MODEL','source_identity':identity,
 'npu_weight_formats':weight_formats,'storage':a.storage,'bank_sha256':h.hexdigest(),'resident_experts':64,'resident_projections_exact':128,
 'cpu_workers':128,'numa_nodes':[2,3,6,7],'dynamic_migrations':dynamic_evidence,'rows':[],
 'native_sha256':hashlib.sha256(pathlib.Path(kt_kernel_ext.__file__).read_bytes()).hexdigest()}
try:
 with patch.object(ep,'_r61_capture',side_effect=capture):
  for q in (1,10,32):
   for kind,route_ids in (('mixed',[0,1,7,8,63,64,127,128,504,511]),('resident_only',list(range(0,80,8))),('repeated_cpu',[511]*10)):
    torch.manual_seed(15900+q);x=torch.randn(q,2560).bfloat16();r=torch.softmax(torch.randn(q,10),-1).float()
    ids=torch.tensor(route_ids,dtype=torch.long).repeat(q,1)
    expected_cpu=torch.zeros(q,2560);expected_npu=torch.zeros_like(expected_cpu)
    for column,e in enumerate(route_ids):
     quantized_cpu=a.storage!='BF16' and not bool(mask[e])
     gate,up,down=(cpu_weights if quantized_cpu else weights)[e]
     reference_x=q8_activation(x) if quantized_cpu else x.float()
     g=(reference_x@gate.T).bfloat16().float();u=(reference_x@up.T).bfloat16().float()
     intermediate=(torch.nn.functional.silu(g)*u).bfloat16().float()
     if quantized_cpu:intermediate=q8_activation(intermediate)
     y=intermediate@down.T
     if mask[e]:y=y.bfloat16().float()
     target=expected_npu if mask[e] else expected_cpu
     target.add_(y*r[:,column,None].bfloat16().float())
    dispatch=StandardDispatchOutput(hidden_states=x.to('npu:0'),hidden_states_scale=None,
     topk_output=StandardTopKOutput(topk_weights=r.to('npu:0'),topk_ids=ids.to('npu:0'),router_logits=None))
    hashes=[];errors={}
    for repeat in range(3):
     out=method.apply(layer,dispatch).hidden_states;torch.npu.synchronize()
     assert out.dtype==torch.bfloat16 and torch.isfinite(out).all()
     hashes.append(hashlib.sha256(out.cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
    expected=(expected_cpu+expected_npu).bfloat16().float()
    for name,want in (('cpu_partial',expected_cpu),('npu_owned_partial',expected_npu),('routed_merge',expected)):
     actual=captured[name].float();assert torch.isfinite(actual).all()
     rel=float((actual-want).norm()/want.norm().clamp_min(1e-30));errors[name]=rel
     assert rel<=.01,(kind,q,name,rel)
     if torch.count_nonzero(want)==0:assert torch.count_nonzero(actual)==0
    assert len(set(hashes))==1
    assert captured['cpu_partial'].dtype==captured['npu_owned_partial'].dtype==torch.float32
    merge=(captured['cpu_partial']+captured['npu_owned_partial']).bfloat16()
    assert torch.equal(captured['routed_merge'],merge)
    row={'tokens':q,'kind':kind,'errors':errors,'hashes':hashes};report['rows'].append(row);print(json.dumps(row),flush=True)
    del out,dispatch
 report['status']='PASS_REAL_KTEP_LOCAL_MERGE'
 report['peak_allocated']=torch.npu.max_memory_allocated()
except BaseException as exc:
 report.update(status='FAIL',error=repr(exc));raise
finally:
 set_global_expert_distribution_recorder(previous)
 kt_kernel_ext.shutdown_ascend_callback_worker()
 a.output.write_text(json.dumps(report,indent=2)+'\n')
print(report['status'],flush=True)
