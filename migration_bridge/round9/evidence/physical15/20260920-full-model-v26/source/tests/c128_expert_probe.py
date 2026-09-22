"""Actual C128 wrapper, original BF16 bank and framework NPU boundary.

CPU routed partial only. No NPU expert computation or full KTEP merge claim.
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
torch.npu.set_per_process_memory_fraction(0.04,0)
acl=ctypes.CDLL('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so');phy=ctypes.c_int32(-1)
acl.aclrtGetPhyDevIdByLogicDevId.argtypes=[ctypes.c_int32,ctypes.POINTER(ctypes.c_int32)]
assert acl.aclrtGetPhyDevIdByLogicDevId(0,ctypes.byref(phy))==0 and phy.value==15 and torch.npu.device_count()==1
assert list(pathlib.Path('/dev').glob('davinci[0-9]*'))==[pathlib.Path('/dev/davinci15')]
mask=torch.arange(512).remainder(8)==0
wrapper=LlamafileMoEWrapper(layer_idx=0,num_experts=512,num_experts_per_tok=10,
 hidden_size=2560,moe_intermediate_size=640,gpu_experts_mask=mask,cpuinfer_threads=128,
 threadpool_count=4,weight_path=str(a.bank),chunked_prefill_size=32,numa_nodes=[2,3,6,7],max_deferred_experts_per_token=0)
wrapper.load_weights()
# Record actual worker affinities and NUMA pages; capacity is not proof of binding.
thread_evidence={}
for task in pathlib.Path('/proc/self/task').iterdir():
 status=task.joinpath('status').read_text()
 thread_evidence[task.name]=[line for line in status.splitlines() if line.startswith(('Name:','Cpus_allowed_list:','Mems_allowed_list:'))]
a.output.with_suffix('.threads.json').write_text(json.dumps(thread_evidence,indent=2)+'\n')
a.output.with_suffix('.numa_maps.txt').write_text(pathlib.Path('/proc/self/numa_maps').read_text())
print('WRAPPER_BANK_LOADED',flush=True)
index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
weights={}
for e in (1,7,63,127,511):
 pre='model.language_model.layers.0.mlp.experts.'
 with safe_open(str(a.model/index[pre+'gate_up_proj']),framework='pt',device='cpu') as f:gu=f.get_slice(pre+'gate_up_proj')[e].float()
 with safe_open(str(a.model/index[pre+'down_proj']),framework='pt',device='cpu') as f:down=f.get_slice(pre+'down_proj')[e].float()
 weights[e]=(*gu.chunk(2,dim=0),down)
report={'status':'RUNNING','scope':'C128_CPU_WRAPPER_PARTIAL_NPU_TRANSFER_ONLY','source_identity':identity,
 'bank_sha256':h.hexdigest(),'cpu_workers':128,'numa_nodes':[2,3,6,7],'resident_mask':'logical_id_mod8_eq0',
 'bf16_numerical_compat':True,'rel_l2_gate':.01,'rows':[],'native_sha256':hashlib.sha256(pathlib.Path(kt_kernel_ext.__file__).read_bytes()).hexdigest()}
try:
 for q in (1,10,32):
  for kind,route_ids in (('mixed',[0,1,7,8,63,64,127,128,504,511]),('resident_only',list(range(0,80,8))),('repeated_cpu',[511]*10)):
   torch.manual_seed(15900+q);x=torch.randn(q,2560).bfloat16();r=torch.softmax(torch.randn(q,10),-1).float()
   ids=torch.tensor(route_ids,dtype=torch.long).repeat(q,1);expected=torch.zeros(q,2560)
   for column,e in enumerate(route_ids):
    if mask[e]:continue
    gate,up,down=weights[e]
    g=(x.float()@gate.T).bfloat16().float();u=(x.float()@up.T).bfloat16().float()
    intermediate=(torch.nn.functional.silu(g)*u).bfloat16().float()
    expected+=(intermediate@down.T)*r[:,column,None].bfloat16().float()
   dx=x.to('npu:0');di=ids.to('npu:0');dr=r.to('npu:0');hashes=[]
   for repeat in range(3):
    out=wrapper.forward(dx,di,dr,get_current_device_stream_handle('npu'))
    torch.npu.synchronize();actual=out.cpu();assert out.dtype==torch.float32 and torch.isfinite(actual).all()
    hashes.append(hashlib.sha256(actual.numpy().tobytes()).hexdigest())
   relative=float((actual-expected).norm()/expected.norm().clamp_min(1e-30))
   row={'tokens':q,'kind':kind,'relative_l2':relative,'hashes':hashes};report['rows'].append(row)
   assert relative<=.01 and len(set(hashes))==1,row
   if kind=='resident_only':assert torch.count_nonzero(actual)==0
   print(json.dumps(row),flush=True)
   del dx,di,dr,out,actual
 wrapper.clear_buffer_cache();torch.npu.synchronize();torch.npu.empty_cache()
 report.update(status='PASS_C128_WRAPPER_CPU_PARTIAL_ONLY',peak_allocated=torch.npu.max_memory_allocated(),after_allocated=torch.npu.memory_allocated(),after_reserved=torch.npu.memory_reserved())
 assert report['after_allocated']==report['after_reserved']==0
except BaseException as exc:
 report.update(status='FAIL',error=repr(exc));raise
finally:
 kt_kernel_ext.shutdown_ascend_callback_worker()
 a.output.write_text(json.dumps(report,indent=2)+'\n')
print(report['status'],flush=True)
