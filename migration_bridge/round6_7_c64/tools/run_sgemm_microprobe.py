"""Pre-registered F32 tail gate: rel-L2 <=1e-4, 10 byte-exact repeats."""
import ctypes
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
torch.manual_seed(67064)
root=Path('/workspace/c64-dev')
lib=ctypes.CDLL(str(root/'sgemm_microprobe.so'))
call=lib.c64_sgemm
call.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*3
call.restype=ctypes.c_int
results=[]
for k in (2048,704):
 for m in (32,64,128,192,224,256):
  for n in (1,2,8,16,32):
   a=torch.randn(m,k,dtype=torch.float32)*.1
   b=torch.randn(n,k,dtype=torch.float32)*.1
   ref=b@a.T
   anchor=None
   exact=True
   statuses=[]
   for repeat in range(10):
    y=torch.full((n,m),float('nan'))
    statuses.append(call(m,n,k,a.data_ptr(),b.data_ptr(),y.data_ptr()))
    if anchor is None: anchor=y.clone()
    else: exact &= torch.equal(y,anchor)
   rel=float(torch.linalg.vector_norm(y-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-30))
   finite=bool(torch.isfinite(y).all())
   row=dict(m=m,n=n,k=k,rel_l2=rel,finite=finite,repeat_exact=exact,
            kernel_returns=statuses,input_sha256=hashlib.sha256(a.numpy().tobytes()+b.numpy().tobytes()).hexdigest(),
            pass_gate=finite and exact and all(statuses) and rel<=1e-4)
   results.append(row)
   print(json.dumps(row),flush=True)
report=dict(gate=1e-4,seed=67064,repeats=10,cases=results,
            binary_sha256=hashlib.sha256((root/'sgemm_microprobe.so').read_bytes()).hexdigest(),
            status='PASS' if all(r['pass_gate'] for r in results) else 'FAIL')
(root/'sgemm_tail_microprobe.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if report['status']=='PASS' else 1)
