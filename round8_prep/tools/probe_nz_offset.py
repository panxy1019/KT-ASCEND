"""Diagnostic only: isolate NZ sliced equality/copy versus host oracle."""
import torch
import torch_npu
import sys
torch.npu.config.allow_internal_format = True
torch.manual_seed(19)
host=torch.randn(8,2048,1024,dtype=torch.bfloat16)
source=host.npu()
destination=torch.zeros(3,2048,1024,dtype=torch.bfloat16).npu()
if '--transpose' in sys.argv:
    source=source.transpose(1,2)
    destination=destination.transpose(1,2)
    host=host.transpose(1,2)
src=torch_npu.npu_format_cast(source,29)
dst=torch_npu.npu_format_cast(destination,29)
print('OPERANDS',src.shape,src.stride(),dst.shape,dst.stride(),flush=True)
assert torch_npu.get_npu_format(src) == 29
assert torch_npu.get_npu_format(dst) == 29
print('COPY_BEGIN',flush=True)
for slot,logical in enumerate([6,2,5]):dst[slot].copy_(src[logical])
torch.npu.synchronize()
print('COPY_END_EQUAL_BEGIN',flush=True)
for slot,logical in enumerate([6,2,5]):assert torch.equal(dst[slot],src[logical])
print('EQUAL_END_HOST_ORACLE_BEGIN',flush=True)
actual=torch_npu.npu_format_cast(dst,2).cpu()
assert torch.equal(actual,host[[6,2,5]])
print('HOST_ORACLE_PASS',flush=True)
print('SLICED_HOST_COPY_BEGIN',flush=True)
for logical in [6,2,5]:
    assert torch.equal(src[logical].detach().cpu(),host[logical])
print('SLICED_HOST_COPY_PASS',flush=True)
