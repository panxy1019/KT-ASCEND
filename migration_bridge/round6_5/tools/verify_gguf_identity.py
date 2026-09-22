"""Verify every exported F32 expert converts exactly to source BF16 bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    import numpy as np
    import torch
    from gguf import GGUFReader
    p=argparse.ArgumentParser()
    p.add_argument('--gguf',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('REFUSE_OVERWRITE')
    m=json.loads(a.gguf.with_suffix('.gguf.manifest.json').read_text())
    if m['layers']!=list(range(1,27)) or m['model_revision']!='604d5664dddd88a0433dbae533b7fe9472482de0':
        raise ValueError('SOURCE_SCOPE')
    torch.set_num_threads(1)
    reader=GGUFReader(str(a.gguf))
    rows=[]
    for t in reader.tensors:
        parts=t.name.split('.')
        layer=int(parts[1]); short=parts[2].split('_')[1]
        if list(t.data.shape)!=m['exported_shapes'][t.name]: raise ValueError('SHAPE')
        for expert,key in enumerate(m['source_tensor_keys'][str(layer)][short]):
            f32=torch.from_numpy(np.array(t.data[expert],copy=True))
            bf16=f32.bfloat16()
            if not torch.equal(f32,bf16.float()): raise ValueError('NOT_BF16_VALUED:'+key)
            sha=hashlib.sha256(bf16.view(torch.uint8).numpy().tobytes()).hexdigest()
            if sha!=m['source_tensor_sha256'][key]: raise ValueError('WEIGHT_IDENTITY:'+key)
            rows.append(dict(key=key,sha256=sha,exact=True))
        print(json.dumps(dict(tensor=t.name,verified_experts=len(rows))),flush=True)
    if len(rows)!=4992: raise ValueError('TENSOR_COUNT')
    a.output.write_text(json.dumps(dict(status='ALL_EXPERT_WEIGHT_IDENTITY_PASS',ended_unix=time.time(),
        count=len(rows),gguf_sha256=m['gguf_sha256'],rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
