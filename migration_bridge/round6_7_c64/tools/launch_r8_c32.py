"""Host-only isolated R8 C32 launch. Never mutates existing container/workload."""
import json
from pathlib import Path
import subprocess
import time

base = Path('/home/admin/kt-r64-migration/payload/kt-artifacts/round6-deferred')
root = base/'r65-r67-c64-dev-v1/r8-c32-v2'
inside = '/workspace/results/r65-r67-r8-c32-v2'
name = 'kt-r67-r8-b1-c32-v2'
source = json.loads(subprocess.check_output(['docker','inspect','kt-r65-static-offload']))[0]
mem = {r.split(':')[0]:int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines() if r.endswith('kB')}
assert mem['SwapTotal']==mem['SwapFree'] and mem['MemAvailable']>=max(64*2**30,.2*mem['MemTotal'])
inventory = subprocess.check_output(['npu-smi','info'],text=True)
# Explicit target-specific admission; do not assume idle from unused ID alone.
assert 'No running processes found in NPU 6' in inventory
assert Path('/sys/bus/pci/devices/0000:41:00.0/numa_node').read_text().strip()=='2'
build = json.loads((root/'build-provenance.json').read_text())
assert build['relink_control_exact'] and build['tp_count']==1 and build['workers']==32
binary = 'kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'
native = '/workspace/results/venv-r6/lib/python3.11/site-packages/kt_kernel/'+binary
cmd = ['docker','run','-d','--name',name,'--label','kt.owner=round6.7-r8-c32-v2',
       '--entrypoint','/bin/sleep','--network','none','--cap-add','SYS_NICE',
       '--cpuset-cpus','64-95','--cpuset-mems','2','--memory','192g','--memory-swap','192g','--shm-size','16g']
for device in ('davinci6','davinci_manager','devmm_svm','hisi_hdc'):
    cmd += ['--device','/dev/'+device]
for key,value in dict(KT_R67_CPU_SET='64-95',KT_R67_NUMA_NODE='2',KT_R67_NPU_ID='6',
                      ASCEND_RT_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',
                      KT_R67_RUNTIME_MANIFEST=inside+'/runtime-manifest.json').items():
    cmd += ['-e',key+'='+value]
for mount in source['Mounts']:
    cmd += ['-v',f"{mount['Source']}:{mount['Destination']}:ro"]
cmd += ['-v',f'{root}:{inside}:rw','-v',f'{root/binary}:{native}:ro']
for f in ('worker_pool.cpp','worker_pool.h'):
    cmd += ['-v',f'{base}/r65-r67-c64-dev-v1/v4/kt-kernel/cpu_backend/{f}:/workspace/kt-src/kt-kernel/cpu_backend/{f}:ro']
cmd += ['kt-r67-frozen-runtime:20260917','infinity']
# Nested writable bind target must exist under the read-only parent bind.
(base/'r65-r67-r8-c32-v2').mkdir(exist_ok=True)
existing = subprocess.run(['docker','inspect',name],capture_output=True,text=True)
if existing.returncode == 0:
    info = json.loads(existing.stdout)[0]
    assert info['Config']['Labels'].get('kt.owner')=='round6.7-r8-c32-v2'
    assert info['HostConfig']['CpusetCpus']=='64-95' and info['HostConfig']['CpusetMems']=='2'
    assert sorted(info['HostConfig']['Binds'])==sorted(cmd[i+1] for i,x in enumerate(cmd) if x=='-v')
    assert not (root/'study-device0').exists(), 'DO_NOT_RESTART_EXISTING_STUDY'
    cid=info['Id']
    if not info['State']['Running']: subprocess.run(['docker','start',name],check=True)
else:
    cid = subprocess.check_output(cmd,text=True).strip()
(root/'container-provenance.json').write_text(json.dumps(dict(container=cid,name=name,time_unix=time.time(),
    command=cmd,host_memory=mem,npu_preflight=inventory,original_container=source['Id'],
    npu=6,numa=2,cpus='64-95',workers=32,tp_count=1,batch=1,
    performance='CONTENDED_NOT_FORMAL',build=build),indent=2)+'\n')
bootstrap = f'''import json,hashlib
from pathlib import Path
p=Path('/workspace/results/r65-r66-runtime-source-final-b0.json')
m=json.loads(p.read_text())
for r in m['files']:
    if r['path'].endswith(('/cpu_backend/worker_pool.cpp','/cpu_backend/worker_pool.h','/{binary}')):
        r['sha256']=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
m['amendment']='SPARSE_NUMA_C32_ONLY'
Path('{inside}/runtime-manifest.json').write_text(json.dumps(m,indent=2)+'\\n')
'''
subprocess.run(['docker','exec',name,'/workspace/results/venv-r6/bin/python','-c',bootstrap],check=True)
tools = inside+'/migration_bridge/round6_7/tools'
study = inside+'/study-device0'
shell = f'''set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH="/workspace/kt-src/third_party/sglang/python:/workspace/kt-src:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /workspace/results/venv-r6/bin/python /workspace/r65-tools/migration_bridge/round6_5/tools/owned_guard.py --output {inside}/guard-device0 --timeout-seconds 86400 -- /workspace/results/venv-r6/bin/python {tools}/farm_worker.py --base /workspace/results/r65-r67-serial-v5 --study {study} --profile R8 >{inside}/controller-device0.log 2>&1
'''
# The container exposes one physical device (host6), enumerated as logical0.
subprocess.run(['docker','exec','-d','-e','KT_R67_NPU_ID=0','-e','ASCEND_RT_VISIBLE_DEVICES=0',name,'bash','-c',shell],check=True)
print(json.dumps(dict(container=name,id=cid,study=study,status='LAUNCHED_SYSTEM_SCREEN_NOT_YET_PASS')))
