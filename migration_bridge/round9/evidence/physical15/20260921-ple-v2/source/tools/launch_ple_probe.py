"""Exclusive physical15 PLE-only probe: 4 CPUs, 8GiB host, guarded4GiB HBM.

This launcher has no shared mode and never stops unrelated workloads. It is a
component test, not full-model admission. The component itself caps the NPU
allocator to1percent; the process guard also accounts for driver memory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def command(args):return subprocess.check_output(args,text=True,timeout=30)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admission():
    if '110.123.0.3' not in command(['hostname','-I']).split():raise RuntimeError('wrong host')
    cpus=(160,162,164,166)
    if not set(cpus)<=os.sched_getaffinity(0):raise RuntimeError('PLE CPU affinity unavailable')
    cores=set()
    for cpu in cpus:
        path=Path(f'/sys/devices/system/cpu/cpu{cpu}')
        if not (path/'node2').exists():raise RuntimeError('PLE NUMA topology changed')
        cores.add(((path/'topology/physical_package_id').read_text(),(path/'topology/core_id').read_text()))
    if len(cores)!=4:raise RuntimeError('PLE physical cores alias')
    def cpu_stats():
        return {int(v[0][3:]):list(map(int,v[1:])) for line in Path('/proc/stat').read_text().splitlines()
                if (v:=line.split())[0] in {f'cpu{x}' for x in cpus}}
    before=cpu_stats();time.sleep(1);after=cpu_stats()
    idle={c:(after[c][3]-before[c][3])/max(1,sum(after[c][:8])-sum(before[c][:8])) for c in cpus}
    if sum(idle.values())/4<=.8:raise RuntimeError('PLE selected CPUs busy')
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(mem['MemAvailable'].split()[0])<16*1024**2:raise RuntimeError('PLE host reserve unavailable')
    numa=Path('/sys/devices/system/node/node2/meminfo').read_text()
    free=next(int(line.split()[-2]) for line in numa.splitlines() if 'MemFree:' in line)
    if free<2*1024**2:raise RuntimeError('PLE NUMA free reserve unavailable')
    npu=command(['npu-smi','info'])
    if re.search(r'\|\s*7\s+1\s*\|\s*\d+\s*\|',npu):raise RuntimeError('physical15 not idle; exclusive PLE denied')
    row=next(line for line in npu.splitlines() if '0000:83:00.0' in line)
    used,total=map(int,re.findall(r'(\d+)\s*/\s*(\d+)',row)[-1])
    if total!=65536 or used+4096+8192>total:raise RuntimeError('PLE HBM reserve unavailable')
    return dict(status='ADMITTED_PLE_COMPONENT_PENDING',time=time.time(),host='110.123.0.3',
        physical=15,resource_mode='exclusive',cpus=list(cpus),numa_nodes=[2],cpu_quota=4,
        idle_fraction=idle,host_memory_limit_gib=8,owned_hbm_limit_mib=4096,
        allocator_fraction=.01,npu_smi=npu,host_mem_available_kib=int(mem['MemAvailable'].split()[0]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',type=int,required=True)
    parser.add_argument('--probe',choices=('ple','joint'),default='ple')
    parser.add_argument('--ple-gate',type=Path)
    args=parser.parse_args()
    if args.version<1:raise ValueError('positive version required')
    root=Path('/mnt/data1/kt-round9');source=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(source))
    if args.probe=='joint':
        from runtime.joint_budget_gate import SOURCE_FILES
        if args.ple_gate is None or not args.ple_gate.resolve(strict=True).is_relative_to(root):
            raise ValueError('joint probe requires owned standalone PLE receipt')
    else:
        from runtime.ple_gate import SOURCE_FILES
        if args.ple_gate is not None:raise ValueError('standalone PLE probe does not accept a prior gate')
    out=root/'evidence'/f'20260921-ple-v{args.version}';out.mkdir(exist_ok=False)
    sample=admission()
    for sub in ('runtime','tests','tools'):
        shutil.copytree(source/sub,out/'source'/sub,ignore=shutil.ignore_patterns('__pycache__'))
    frozen=out/'source'
    all_sha={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()}
    (out/'source-sha256.json').write_text(json.dumps(all_sha,sort_keys=True,indent=2)+'\n')
    probe_file='joint_budget_device_probe.py' if args.probe=='joint' else 'ple_async_device_probe.py'
    bound=dict(source_sha256={name:all_sha['runtime/'+name] for name in SOURCE_FILES},
               probe_sha256=all_sha['tests/'+probe_file],probe_kind=args.probe)
    extra=''
    if args.probe=='joint':
        (out/'ple-component-gate.json').write_bytes(args.ple_gate.read_bytes())
        bound.update(status='ADMITTED_JOINT_BUDGET_PENDING',ple_gate_sha256=sha(out/'ple-component-gate.json'))
        extra=f' --ple-gate /out/ple-component-gate.json --ple-gate-sha256 {bound["ple_gate_sha256"]}'
    sample.update(bound)
    (out/'admission.json').write_text(json.dumps(sample,indent=2)+'\n')
    rev='de4b8e4d43b917e7706784d8bb445c9af86a3540'
    name=f'kt-round9-ple-v{args.version}'
    script=('source /usr/local/Ascend/ascend-toolkit/set_env.sh; ulimit -c 0; '
        f'exec timeout -k 10 300 python3 -u /out/source/tests/{probe_file} '
        f'--model /models/{rev} --identity-receipt /models/{rev}.round9-support/destination-verification/verification-1789886190714568317.json '
        '--admission /out/admission.json --output /out/result.json'+extra+' > /out/probe.log 2>&1')
    cmd=['docker','create','--name',name,'--network','none','--cpuset-cpus',','.join(map(str,sample['cpus'])),
         '--cpuset-mems','2','--cpus','4','--memory','8g','--memory-swap','8g','--pids-limit','1024','--shm-size','1g']
    for dev in ('davinci15','davinci_manager','devmm_svm','hisi_hdc'):cmd+=['--device','/dev/'+dev]
    for mount in (f'{out}:/out','/home/admin/models/Qwen3.8-Flash-Next:/models:ro',
                  '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro'):cmd+=['-v',mount]
    for env in ('ASCEND_RT_VISIBLE_DEVICES=0','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2'):
        cmd+=['-e',env]
    cmd+=['--entrypoint','bash','601e76d17e9f','-c',script];command(cmd)
    inspection=json.loads(command(['docker','inspect',name]))[0]
    (out/'container-created.json').write_text(json.dumps(inspection,indent=2)+'\n')
    with (out/'guard.log').open('w') as stream:
        guard=subprocess.Popen(['python3',str(frozen/'tools/monitor_owned_server.py'),
            '--container',name,'--resource-mode','exclusive','--output',str(out/'monitor.jsonl'),
            '--ready-file',str(out/'guard-ready.json'),'--seconds','330','--owned-limit-mib','4096'],
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        deadline=time.monotonic()+40
        while not (out/'guard-ready.json').exists():
            if guard.poll() is not None or time.monotonic()>deadline:raise RuntimeError('PLE guard readiness failed')
            time.sleep(.2)
        fresh=admission();fresh.update(bound)
        (out/'admission.json').write_text(json.dumps(fresh,indent=2)+'\n')
        if guard.poll() is not None:raise RuntimeError('PLE guard exited before launch')
        if any(sha(frozen/rel)!=value for rel,value in all_sha.items()):raise RuntimeError('frozen PLE probe source changed')
        command(['docker','start',name]);print('STARTED_MONITORED_PLE',name,flush=True)
    except BaseException:
        subprocess.run(['docker','stop','--time','10',name],check=False,timeout=30)
        if guard.poll() is None:
            guard.terminate()
            try:guard.wait(timeout=10)
            except subprocess.TimeoutExpired:
                guard.kill();guard.wait(timeout=10)
        raise


if __name__=='__main__':main()
