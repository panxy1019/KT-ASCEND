"""Original-row PLE probe. Frozen rel-L2 gates FP32=1e-4, BF16=0.03.

Real checkpoint weights/rows, synthetic hidden states; no full-model claim.
"""
import argparse,ast,ctypes,hashlib,json,pathlib,sys,types
import torch
from safetensors import safe_open
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=pathlib.Path,required=True)
p.add_argument('--identity-receipt',type=pathlib.Path,required=True)
p.add_argument('--reference-source',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
base=pathlib.Path(__file__).parents[1];sys.path.insert(0,str(base));sys.path.insert(0,str(base/'tools'))
from probe_identity import verify_probe_identity
from runtime.ple_hash import NGramHasher
from runtime.ple_rows import ShardedBF16Rows,PREFIX
from runtime.ple_sync import SynchronousPLE
rev='de4b8e4d43b917e7706784d8bb445c9af86a3540'
identity=verify_probe_identity(model=a.model,receipt=a.identity_receipt,expected_model=pathlib.Path('/models')/rev,
 expected_receipt_sha256='7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239',revision=rev,
 manifest_sha256='62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e')
index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
config=json.loads((a.model/'config.json').read_text())['text_config']
assert config['ple_layer_ids']==[2]
pre='model.language_model.layers.1.ple.'
def load(key):
 with safe_open(str(a.model/index[key]),framework='pt',device='cpu') as f:return f.get_tensor(key)
mult=load(pre+'ple_embedding.layer_multipliers').tolist()
sizes=load(pre+'ple_embedding.ngram_heads_vocab_sizes').tolist()
offsets=load(pre+'ple_embedding.ngram_heads_offsets').tolist()
hasher=NGramHasher(ngram_size=config['ngram_size'],heads_per_ngram=config['heads_per_ngram'],
 vocab_size=config['vocab_size'],eos_token_id=config['eos_token_id'],multipliers=mult,head_sizes=sizes,head_offsets=offsets)
# Invoke only the two exact frozen upstream hash methods, without CUDA imports.
source_bytes=a.reference_source.read_bytes()
if hashlib.sha256(source_bytes).hexdigest()!='a3981e148bc9ede4b84dea700f40831f86211cea29b817e8cc8feca1066e9f72':
 raise ValueError('PLE reference must match frozen upstream 4ccff141 source')
source=source_bytes.decode();tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen4ExpNGramEmbedding')
cls.bases=[];cls.decorator_list=[];cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_hash_contexts','_shift_right_ignore_eos')]
ns={'torch':torch,'get_req_to_token_pool':lambda:types.SimpleNamespace(ple_window_cache=None)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'frozen-upstream-hash','exec'),ns)
ref=ns['Qwen4ExpNGramEmbedding']();ref.ngram_size=config['ngram_size'];ref.heads_per_ngram=config['heads_per_ngram'];ref.eos_token_id=config['eos_token_id'];ref.enable_ple_fusion=False
ref.layer_multipliers=torch.tensor(mult);ref.ngram_heads_vocab_sizes=torch.tensor(sizes);ref.ngram_heads_offsets=torch.tensor(offsets)
tokens=[1,42,248043,248044,7,7,0,248319,13,5,6,7,8,9,10,11,12]
ids,history=hasher.hash_tokens(tokens)
contexts=torch.tensor([config['eos_token_id']]*(config['ngram_size']-1)+tokens).unfold(0,config['ngram_size'],1)
assert torch.equal(torch.tensor(ids),ref._hash_contexts(contexts))
report={'status':'RUNNING','scope':'PLE_LOCAL_REAL_ROWS_SYNTHETIC_HIDDEN','source_identity':identity,
 'upstream_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'hash_exact':True,'gates':{'float32':1e-4,'bfloat16':0.03},'rows':[]}
with ShardedBF16Rows(a.model) as table:
 boundary=[]
 for i,(path,start,base_row) in enumerate(table.parts):boundary.extend([base_row,table.ends[i]-1])
 raw=table.gather_bytes(boundary)
 expected=[]
 for row in boundary:
  part=next(i for i,end in enumerate(table.ends) if row<end);name=f'{PREFIX}shard_{part}.weight'
  with safe_open(str(a.model/index[name]),framework='pt',device='cpu') as f:
   one=f.get_slice(name)[row-table.parts[part][2]:row-table.parts[part][2]+1]
   expected.append(one.view(torch.uint8).numpy().tobytes())
 assert raw==b''.join(expected)
 report['boundary_rows_exact']=len(boundary)
 raw=table.gather_bytes([x for row in ids for x in row])
 embeddings=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(len(tokens),2560).clone()
 report['selected_rows_sha256']=hashlib.sha256(raw).hexdigest()
weights={k:load(pre+k) for k in ('key_proj.weight','value_proj.weight','norm_key.weight','norm_query.weight','norm_conv.weight','conv1d.weight')}
torch.set_num_threads(2)
import torch_npu
torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False
torch.npu.conv.allow_hf32=False
acl=ctypes.CDLL('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so');phy=ctypes.c_int32(-1)
acl.aclrtGetPhyDevIdByLogicDevId.argtypes=[ctypes.c_int32,ctypes.POINTER(ctypes.c_int32)]
assert acl.aclrtGetPhyDevIdByLogicDevId(0,ctypes.byref(phy))==0 and phy.value==15
assert torch.npu.device_count()==1
assert list(pathlib.Path('/dev').glob('davinci[0-9]*'))==[pathlib.Path('/dev/davinci15')]
def err(x,y):return float(torch.linalg.vector_norm(x.double()-y.double())/torch.linalg.vector_norm(y.double()).clamp_min(1e-30))
def canonical(x,e):
 t=x.shape[0]
 def norm(v,name):
  v=v.double().reshape(t,4,2560)
  return v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-6)*(1+weights[name].double().reshape(4,2560))
 key=e.double()@weights['key_proj.weight'].double().T;value=e.double()@weights['value_proj.weight'].double().T
 score=(norm(key,'norm_key.weight')*norm(x,'norm_query.weight')).sum(-1,keepdim=True)/(2560**0.5)
 gate=torch.sigmoid(torch.sign(score)*torch.sqrt(torch.abs(score).clamp_min(1e-6)))
 gated=(gate*value.unsqueeze(1)).flatten(1);normalized=norm(gated,'norm_conv.weight').flatten(1)
 conv=[]
 for i in range(t):
  v=torch.zeros(10240,dtype=torch.float64)
  for k in range(4):
   j=i-9+3*k
   if j>=0:v+=normalized[j]*weights['conv1d.weight'][:,0,k].double()
  conv.append(v*torch.sigmoid(v))
 return gated+torch.stack(conv)
try:
 for dtype in (torch.float32,torch.bfloat16):
  model=SynchronousPLE(dilation=hasher.ngram_size,dtype=dtype,device='cpu');model.load_state_dict(weights,strict=True);model=model.to('npu:0')
  for length in (1,3,17):
   torch.manual_seed(9315+length);x=torch.randn(length,10240).to(dtype);e=embeddings[:length].to(dtype)
   want=canonical(x,e);dx=x.to('npu:0');de=e.to('npu:0');hashes=[]
   with torch.no_grad():
    for repeat in range(3):
     out,state=model(dx,de);torch.npu.synchronize();cpu=out.cpu()
     assert torch.isfinite(cpu).all()
     hashes.append(hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())
    relative=err(cpu,want);gate=1e-4 if dtype==torch.float32 else 0.03
    pieces=[];partial=None
    for i in range(length):
     piece,partial=model(dx[i:i+1],de[i:i+1],partial);pieces.append(piece)
    chunk_error=err(torch.cat(pieces).cpu(),cpu);state_error=err(partial.cpu(),state.cpu())
    row={'dtype':str(dtype),'tokens':length,'relative_l2':relative,'decode_relative_l2':chunk_error,'state_relative_l2':state_error,'repeat_hashes':hashes}
    report['rows'].append(row)
    assert len(set(hashes))==1 and max(relative,chunk_error,state_error)<=gate,row
   del out,state,cpu,dx,de,pieces,partial,piece
  del model;torch.npu.synchronize();torch.npu.empty_cache()
 report.update(status='PASS_PLE_LOCAL_ONLY',peak_allocated=torch.npu.max_memory_allocated(),after_allocated=torch.npu.memory_allocated(),after_reserved=torch.npu.memory_reserved())
 assert report['after_allocated']==report['after_reserved']==0
except BaseException as exc:
 report['status']='FAIL';report['error']=repr(exc);raise
finally:a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({'status':report['status'],'rows':len(report['rows']),'peak_allocated':report['peak_allocated']}))
