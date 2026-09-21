"""Check a process-local NPU allocator limit without a large allocation."""
import argparse
import ctypes
import json
from pathlib import Path
import torch
import torch_npu

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
torch.npu.set_device(0)
acl = ctypes.CDLL('/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so')
phy = ctypes.c_int32(-1)
assert acl.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(phy)) == 0 and phy.value == 15
assert torch.npu.device_count() == 1
assert list(Path('/dev').glob('davinci[0-9]*')) == [Path('/dev/davinci15')]
free, total = torch.npu.mem_get_info()
report = {'status': 'RUNNING', 'scope': 'PROCESS_LOCAL_ALLOCATOR_CAP_ONLY',
          'physical_id': 15, 'visible_total_bytes': total, 'free_before_bytes': free}
try:
    torch.npu.empty_cache()
    torch.npu.set_per_process_memory_fraction((1024**2)/total, 0)
    try:
        forbidden = torch.empty(1024**2, dtype=torch.float32, device='npu:0')
    except RuntimeError as exc:
        if 'out of memory' not in str(exc).lower():
            raise
        report['expected_cap_rejection'] = str(exc)
    else:
        del forbidden
        raise AssertionError('4MiB allocation escaped the1MiB process cap')
    torch.npu.set_per_process_memory_fraction((41*1024**3)/total, 0)
    small = torch.ones(1024, device='npu:0')
    torch.npu.synchronize()
    assert float(small.sum().cpu()) == 1024
    del small
    torch.npu.empty_cache()
    report.update(status='PASS_PROCESS_ALLOCATOR_CAP', proposed_owned_cap_bytes=41*1024**3,
                  allocated_after=torch.npu.memory_allocated(), reserved_after=torch.npu.memory_reserved())
except BaseException as exc:
    report.update(status='FAIL', error=repr(exc))
    raise
finally:
    a.output.write_text(json.dumps(report, indent=2)+'\n')
print(report['status'], flush=True)
