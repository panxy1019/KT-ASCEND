"""Create an isolated C64 development container and freeze deployed provenance."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

base=Path('/home/admin/kt-r64-migration/payload/kt-artifacts/round6-deferred')
root=base/'r65-r67-c64-development-v1'
root.mkdir(exist_ok=False)
source=json.loads(subprocess.check_output(['docker','inspect','kt-r65-static-offload']))[0]
memory={r.split(':')[0]:int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines() if r.endswith('kB')}
assert memory['SwapTotal']==memory['SwapFree'] and memory['MemAvailable']>.2*memory['MemTotal']
smi=subprocess.check_output(['npu-smi','info','-t','proc-mem','-i','2'],text=True)
assert 'No process in device.' in smi
assert Path('/sys/bus/pci/devices/0000:81:00.0/numa_node').read_text().strip()=='4'
cmd=['docker','run','-d','--name','kt-r67-c64-development-v1','--label','kt.owner=round6.7-c64-development',
     '--entrypoint','/bin/sleep','--network','none','--cap-add','SYS_NICE','--cpuset-cpus','128-191',
     '--cpuset-mems','4,5','--memory','192g','--memory-swap','192g','--shm-size','16g',
     '-e','ASCEND_RT_VISIBLE_DEVICES=0','-e','PYTHONDONTWRITEBYTECODE=1']
for dev in ('davinci2','davinci_manager','devmm_svm','hisi_hdc'): cmd+=['--device','/dev/'+dev]
for m in source['Mounts']: cmd+=['-v',f"{m['Source']}:{m['Destination']}:ro"]
cmd+=['-v',f'{root}:/workspace/c64-dev:rw','kt-r67-frozen-runtime:20260917','infinity']
cid=subprocess.check_output(cmd,text=True).strip()
record=dict(time_unix=time.time(),container=cid,name='kt-r67-c64-development-v1',command=cmd,
            host_memory=memory,npu_preflight=smi,physical_npu=2,logical_npu=0,numa=[4,5],cpu_set='128-191',
            protected_containers=['kt-r65-static-offload','kt-r67-r8-b1-c32-v2'])
(root/'container-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
probe=r'''import hashlib,json,subprocess,importlib.util
from pathlib import Path
paths=['kt-kernel/cpu_backend/worker_pool.cpp','kt-kernel/cpu_backend/worker_pool.h',
'kt-kernel/python/experts_base.py','kt-kernel/python/utils/llamafile.py',
'kt-kernel/operators/llamafile/moe.hpp','kt-kernel/operators/moe-tp.hpp',
'third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py']
out={'files':{},'git':{},'imports':{}}
for rel in paths:
 p=Path('/workspace/kt-src')/rel
 out['files'][str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
native=next(Path('/workspace/results/venv-r6/lib/python3.11/site-packages/kt_kernel').glob('kt_kernel_ext*.so'))
out['native']={'path':str(native),'sha256':hashlib.sha256(native.read_bytes()).hexdigest()}
for repo in ['/workspace/kt-src','/workspace/kt-src/third_party/sglang']:
 out['git'][repo]=subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()
out['imports']['kt_kernel']=str(importlib.util.find_spec('kt_kernel').origin)
out['compiler']=subprocess.check_output(['g++','--version'],text=True)
out['flags']=Path('/workspace/results/migration-cpu-build-v1/build/CMakeFiles/kt_kernel_ext.dir/flags.make').read_text()
Path('/workspace/c64-dev/provenance.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
'''
subprocess.run(['docker','exec','kt-r67-c64-development-v1','/workspace/results/venv-r6/bin/python','-c',probe],check=True)
print(json.dumps(record,indent=2))
