"""Actual KTEP local merge with 64 original BF16 NPU experts and C64 CPU bank.

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
assert h.hexdigest()=='731679cb66e83c7c9a2b4b2ae29af68581729e194d8d495530f2aec1342bd97b'
assert kt_kernel_ext.llamafile_bf16_f32_tail_supported()
torch.npu.set_device(0)
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
config=ep.KTConfig(layer_idx=0,gpu_experts_mask=mask,cpuinfer_threads=64,
 threadpool_count=2,numa_nodes=[6,7],weight_path=str(a.bank),chunked_prefill_size=32,
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
# Verify every original resident expert after native ND->NZ conversion and transpose.
w13_nd=torch_npu.npu_format_cast(layer.w13_weight,2).cpu()
w2_nd=torch_npu.npu_format_cast(layer.w2_weight,2).cpu()
for local,e in enumerate(torch.where(mask)[0].tolist()):
 gu,down=expert(e)
 assert torch.equal(w13_nd[local].T,gu)
 assert torch.equal(w2_nd[local].T,down)
del gu,down,w13_nd,w2_nd
print('KTEP_REAL64_NZ_IDENTITY_PASS',flush=True)
weights={}
for e in sorted(set([1,7,63,127,511]+list(range(0,80,8))+[128,504])):
 gu,down=expert(e);gate,up=gu.float().chunk(2,0);weights[e]=(gate,up,down.float())
del gu,down
captured={}
def capture(stage,tensor,**kwargs):
 if stage in ('cpu_partial','npu_owned_partial','routed_merge'):captured[stage]=tensor.detach().cpu().clone()
class Recorder:
 def on_gpu_expert_mask(self,idx,observed):assert idx==0 and torch.equal(observed.cpu(),mask)
previous=get_global_expert_distribution_recorder();set_global_expert_distribution_recorder(Recorder())
report={'status':'RUNNING','scope':'REAL_KTEP_LOCAL_MERGE_NOT_FULL_MODEL','source_identity':identity,
 'bank_sha256':h.hexdigest(),'resident_experts':64,'resident_projections_exact':128,
 'cpu_workers':64,'numa_nodes':[6,7],'rows':[],
 'native_sha256':hashlib.sha256(pathlib.Path(kt_kernel_ext.__file__).read_bytes()).hexdigest()}
try:
 with patch.object(ep,'_r61_capture',side_effect=capture):
  for q in (1,10,32):
   for kind,route_ids in (('mixed',[0,1,7,8,63,64,127,128,504,511]),('resident_only',list(range(0,80,8))),('repeated_cpu',[511]*10)):
    torch.manual_seed(15900+q);x=torch.randn(q,2560).bfloat16();r=torch.softmax(torch.randn(q,10),-1).float()
    ids=torch.tensor(route_ids,dtype=torch.long).repeat(q,1)
    expected_cpu=torch.zeros(q,2560);expected_npu=torch.zeros_like(expected_cpu)
    for column,e in enumerate(route_ids):
     gate,up,down=weights[e]
     g=(x.float()@gate.T).bfloat16().float();u=(x.float()@up.T).bfloat16().float()
     intermediate=(torch.nn.functional.silu(g)*u).bfloat16().float()
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
