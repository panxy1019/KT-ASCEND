"""Stream one original routed-expert layer into lossless BF16 GGUF.

Only one expert projection is copied at a time. No full F32 expansion.
Output directory must be new; incomplete outputs never acquire PASS status.
"""
import argparse
import gc
import hashlib
import json
import pathlib
import numpy as np
import torch
import gguf
from safetensors import safe_open
from probe_identity import verify_probe_identity

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def write_bank(output,layer,source,experts=512,hidden=2560,intermediate=640):
 """source(name,expert) returns the original BF16 2-D projection."""
 output.mkdir(parents=True,exist_ok=False)
 staging=[];arrays={};path=output/f'layer-{layer:02d}-bf16.gguf'
 for name,shape in (('gate',(experts,intermediate,hidden)),('up',(experts,intermediate,hidden)),('down',(experts,hidden,intermediate))):
  stage=output/f'{name}.staging';staging.append(stage)
  array=np.memmap(stage,mode='w+',dtype=np.uint16,shape=shape)
  for expert in range(experts):
   original=source(name,expert)
   if original.dtype!=torch.bfloat16 or tuple(original.shape)!=shape[1:]:
    raise ValueError(f'source shape/dtype mismatch {name} expert{expert}')
   if not bool(torch.isfinite(original).all()):raise ValueError('nonfinite source weight')
   array[expert]=original.contiguous().view(torch.uint16).numpy()
  array.flush();arrays[name]=array
 writer=gguf.GGUFWriter(str(path),'llama')
 for name,array in arrays.items():
  writer.add_tensor(f'blk.{layer}.ffn_{name}_exps.weight',array,raw_dtype=gguf.GGMLQuantizationType.BF16)
 writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
 del writer,array,arrays;gc.collect()
 # Independent decoder/header and every expert/projection reopened from source.
 reader=gguf.GGUFReader(str(path));count=0
 if len(reader.tensors)!=3:raise ValueError('unexpected GGUF tensor count')
 for tensor in reader.tensors:
  if int(tensor.tensor_type)!=30:raise ValueError('bank storage is not BF16')
  name=tensor.name.split('ffn_')[1].split('_exps')[0]
  shape=(experts,hidden,intermediate) if name=='down' else (experts,intermediate,hidden)
  if tuple(reversed(tensor.shape.tolist()))!=shape:raise ValueError('GGUF logical shape mismatch')
  actual=tensor.data.view(np.uint16).reshape(shape)
  for expert in range(experts):
   expected=source(name,expert).contiguous().view(torch.uint16).numpy()
   if not np.array_equal(actual[expert],expected):raise ValueError(f'nonexact {name} {expert}')
   count+=1
 del reader,tensor,actual;gc.collect()
 for stage in staging:stage.unlink()
 return {'file':path.name,'bytes':path.stat().st_size,'sha256':sha(path),
         'exact_projection_instances':count,'storage':'BF16_ORIGINAL_BYTES','layer':layer}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',type=pathlib.Path,required=True)
 p.add_argument('--identity-receipt',type=pathlib.Path,required=True)
 p.add_argument('--output',type=pathlib.Path,required=True)
 p.add_argument('--layer',type=int,choices=range(48),required=True)
 a=p.parse_args();torch.set_num_threads(2)
 rev='de4b8e4d43b917e7706784d8bb445c9af86a3540'
 identity=verify_probe_identity(model=a.model,receipt=a.identity_receipt,expected_model=pathlib.Path('/models')/rev,
  expected_receipt_sha256='7d6cb3db19497303bdd8ceb24efae626e1926a610ab7fbd4406b47d05db4a239',revision=rev,
  manifest_sha256='62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e')
 index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
 prefix=f'model.language_model.layers.{a.layer}.mlp.experts.'
 def source(name,expert):
  key=prefix+('down_proj' if name=='down' else 'gate_up_proj')
  with safe_open(str(a.model/index[key]),framework='pt',device='cpu') as f:
   value=f.get_slice(key)[expert]
   if name=='gate':value=value[:640]
   elif name=='up':value=value[640:]
   return value
 row=write_bank(a.output,a.layer,source)
 report={'status':'PASS_SINGLE_LAYER_BF16_BANK_IDENTITY_ONLY','source_identity':identity,**row}
 (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True)

if __name__=='__main__':main()
