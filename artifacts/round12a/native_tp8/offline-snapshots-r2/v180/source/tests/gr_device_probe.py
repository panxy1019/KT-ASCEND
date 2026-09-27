"""Opt-in real-weight GR probe; synthetic activations, not model qualification.

Frozen relative-L2 gates: F32 1e-4; BF16 0.02 against independent double
precision equations using exactly the stored BF16 checkpoint weights.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import torch
from safetensors import safe_open

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--identity-receipt',type=pathlib.Path,required=True)
a=p.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]/'tools'))
from probe_identity import verify_probe_identity
revision='de4b8e4d43b917e7706784d8bb445c9af86a3540'
identity=verify_probe_identity(model=a.model,receipt=a.identity_receipt,
    expected_model=pathlib.Path('/models')/revision,
    expected_receipt_sha256='7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239',
    revision=revision,manifest_sha256='62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e')
import torch_npu
import ctypes as c
acl=c.CDLL('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so')
# torch owns the ACL lifetime; query mapping after its initialization.
torch.npu.set_device(0)
phy=c.c_int32(-1)
acl.aclrtGetPhyDevIdByLogicDevId.argtypes=[c.c_int32,c.POINTER(c.c_int32)]
assert acl.aclrtGetPhyDevIdByLogicDevId(0,c.byref(phy))==0 and phy.value==15
assert torch.npu.device_count()==1
assert list(pathlib.Path('/dev').glob('davinci[0-9]*'))==[pathlib.Path('/dev/davinci15')]
torch.npu.matmul.allow_hf32=False
torch.set_num_threads(2)
path=pathlib.Path(__file__).parents[1]/'runtime/hyperconnection.py'
spec=importlib.util.spec_from_file_location('round9_gr',path)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
report={'status':'RUNNING','scope':'REAL_WEIGHTS_SYNTHETIC_ACTIVATIONS_GR_ONLY','physical_id':phy.value,'source_revision':identity['revision'],'source_identity':identity,'implementation_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'gates':{'float32':1e-4,'bfloat16':0.02},'rows':[]}
def save(): a.output.write_text(json.dumps(report,indent=2))
def error(x,y):
    return float(torch.linalg.vector_norm(x.double()-y.double())/torch.linalg.vector_norm(y.double()).clamp_min(1e-30))
try:
    for layer in (0,23,24,47):
        for role in ('attn','mlp'):
            prefix=f'model.language_model.layers.{layer}.{role}_hyper_connection.'
            weights={}
            for suffix in ('hc_norm.weight','input_mix_weight_down.weight','input_mix_weight_up.weight','block_inject_weight.weight'):
                key=prefix+suffix
                with safe_open(str(a.model/index[key]),framework='pt',device='cpu') as f: weights[suffix]=f.get_tensor(key)
            for dtype in (torch.float32,torch.bfloat16):
                cfg=mod.HyperConnectionConfig(params_dtype=dtype)
                cpu=mod.GatedResidual(cfg,device='cpu')
                cpu.load_state_dict(weights,strict=True)
                model=copy.deepcopy(cpu).to('npu:0')
                for count in (1,3,32):
                    torch.manual_seed(9015+count+layer)
                    x=torch.randn(count,10240).to(dtype); b=torch.randn(count,2560).to(dtype)
                    v=x.double().reshape(count,4,2560)
                    norm=(v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-6)).flatten(-2)*(1+weights['hc_norm.weight'].double())
                    low=norm@weights['input_mix_weight_down.weight'].double().T/4
                    gate=torch.sigmoid((low*torch.sigmoid(low))@weights['input_mix_weight_up.weight'].double().T)
                    want_mix=(gate*norm).reshape(count,4,2560).mean(1)
                    inject=2*torch.sigmoid(norm@weights['block_inject_weight.weight'].double().T/4)
                    want_out=(v+b.double().unsqueeze(1)*inject.unsqueeze(-1)).flatten(-2)
                    dx=x.to('npu:0'); db=b.to('npu:0'); hashes=[]
                    with torch.no_grad():
                        for repeat in range(3):
                            mix,state=model.mix(dx); out=model.combine(db,state)
                            torch.npu.synchronize()
                            cm=mix.cpu(); co=out.cpu()
                            assert torch.isfinite(cm).all() and torch.isfinite(co).all()
                            hashes.append(hashlib.sha256(cm.view(torch.uint8).numpy().tobytes()+co.view(torch.uint8).numpy().tobytes()).hexdigest())
                        em=error(cm,want_mix);eo=error(co,want_out)
                        gate_limit=1e-4 if dtype==torch.float32 else 0.02
                        row={'layer':layer,'role':role,'dtype':str(dtype),'tokens':count,'mix_rel_l2':em,'combine_rel_l2':eo,'repeat_hashes':hashes}
                        report['rows'].append(row);save()
                        assert len(set(hashes))==1 and max(em,eo)<=gate_limit,row
                    del dx,db,mix,state,out,cm,co
                del cpu,model
                torch.npu.synchronize();torch.npu.empty_cache()
    report['peak_allocated']=torch.npu.max_memory_allocated()
    report['after_allocated']=torch.npu.memory_allocated()
    report['after_reserved']=torch.npu.memory_reserved()
    assert report['after_allocated']==0 and report['after_reserved']==0
    report['status']='PASS_GR_LOCAL_ONLY'
except BaseException as e:
    report['status']='FAIL';report['error']=repr(e);raise
finally: save()
print(json.dumps({'status':report['status'],'rows':len(report['rows']),'peak_allocated':report.get('peak_allocated')}))
