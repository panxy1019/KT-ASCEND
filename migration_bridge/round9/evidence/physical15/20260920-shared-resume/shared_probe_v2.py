import runpy
import sys
import json
import time
import subprocess
subprocess.run([sys.executable,'/probe/acl_mapping.py'],check=True)

import torch
import torch_npu
assert torch.npu.device_count() == 1
budget_mib=int(sys.argv[1])
assert budget_mib in (0,256,1024,2048)
start=time.time()
torch.npu.set_device(0)
x=torch.arange(1024,dtype=torch.float32).to('npu:0')
y=x*2+1
torch.npu.synchronize()
assert torch.equal(y.cpu(),torch.arange(1024,dtype=torch.float32)*2+1)
block=None
if budget_mib:
    block=torch.empty(budget_mib*1024*1024,dtype=torch.uint8,device='npu:0')
    block.fill_(7)
    torch.npu.synchronize()
    assert block[0].item()==7 and block[-1].item()==7
print(json.dumps({'phase':'allocated','budget_mib':budget_mib,'allocated':torch.npu.memory_allocated(),'reserved':torch.npu.memory_reserved(),'peak_allocated':torch.npu.max_memory_allocated(),'device':str(torch.npu.get_device_name(0))}),flush=True)
time.sleep(8)
del block,x,y
torch.npu.synchronize()
torch.npu.empty_cache()
print(json.dumps({'status':'PASS_BOUNDED_RUNTIME_ONLY','budget_mib':budget_mib,'after_allocated':torch.npu.memory_allocated(),'after_reserved':torch.npu.memory_reserved(),'elapsed_s':time.time()-start}),flush=True)
