"""Real NPU regression: transposed NZ source, repeated identity changes, CPU oracle.

Restoring per-expert NZ slice copy makes this fail on the offset warning.
The function is extracted from the deployed source, not reimplemented here.
"""
import argparse
import ast
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--child',action='store_true');a=p.parse_args()
if not a.child:
    result=subprocess.run([sys.executable,__file__,'--source',str(a.source),'--child'],capture_output=True,text=True)
    print(result.stdout);print(result.stderr,file=sys.stderr)
    assert result.returncode==0,'operand/weight oracle failed'
    assert 'Check input storage_offset' not in result.stderr,'UNSAFE_NZ_SLICE_COPY_WARNING'
    print('NZ_TRANSFER_REGRESSION_PASS');sys.exit(0)

import torch
import torch_npu
from types import SimpleNamespace
torch.npu.config.allow_internal_format=True
tree=ast.parse(a.source.read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='copy_experts_weights_bf16')
namespace={'torch':torch}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(a.source),'exec'),namespace)
copy=namespace['copy_experts_weights_bf16'];torch.manual_seed(19)
host={};src=SimpleNamespace();dst=SimpleNamespace()
for name,shape in [('w13_weight',(8,1024,2048)),('w2_weight',(8,2048,512))]:
    base=torch.randn(*shape,dtype=torch.bfloat16)
    host[name]=base.transpose(1,2)
    setattr(src,name,torch_npu.npu_format_cast(base.npu().transpose(1,2),29))
    setattr(dst,name,torch_npu.npu_format_cast(torch.zeros((3,*shape[1:]),dtype=torch.bfloat16).npu().transpose(1,2),29))
for selection in ([6,2,5],[1,7,0],[6,2,5]):
    copy(src,dst,torch.tensor(selection,device='npu',dtype=torch.int64))
    torch.npu.synchronize()
    for name in host:
        actual=torch_npu.npu_format_cast(getattr(dst,name),2).cpu()
        assert torch.equal(actual,host[name][selection]),name
print('WEIGHT_ORACLE_3_UPDATES_PASS')
