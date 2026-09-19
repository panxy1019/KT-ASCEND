"""Fail-closed admission for the isolated Q32 stability container on NPU2."""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT=Path('/home/admin/kt-round8-qwen35')
CONTAINER='kt-round8-qwen35-stability'


def command(*args): return subprocess.check_output(args,text=True,timeout=45)
def swap_counters():
    rows=dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return int(rows['pswpin']),int(rows['pswpout'])


def main():
    before=swap_counters();time.sleep(30);after=swap_counters()
    inspect=json.loads(command('docker','inspect',CONTAINER))[0]
    host=inspect['HostConfig'];reasons=[]
    if inspect['State']['Running']:reasons.append('OWNED_CONTAINER_ALREADY_RUNNING')
    if host['CpusetCpus']!='128-191' or host['CpusetMems']!='4,5':reasons.append('CPU_NUMA_IDENTITY')
    if host['Memory']!=384*2**30 or host['MemorySwap']!=384*2**30:reasons.append('CGROUP_MEMORY_IDENTITY')
    devices=[x['PathOnHost'] for x in host['Devices']]
    if '/dev/davinci2' not in devices:reasons.append('NPU_IDENTITY')
    proc=command('npu-smi','info','-t','proc-mem','-i','2','-c','0')
    if 'No process in device.' not in proc:reasons.append('NPU2_BUSY')
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable']<384*2**30:reasons.append('HOST_MEMORY_RESERVE')
    if after!=before:reasons.append('HOST_SWAP_ACTIVITY')
    if shutil.disk_usage(ROOT).free<64*2**30:reasons.append('DISK_RESERVE')
    result=dict(timestamp=time.time(),admitted=not reasons,reasons=reasons,
        execution_host='110.129.0.16',container=CONTAINER,npu=2,cpus='128-191',mems='4,5',
        available_bytes=mem['MemAvailable'],swap_before=before,swap_after=after,
        npu_proc_raw=proc)
    (ROOT/'q32-stability-admission.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(bool(reasons))


if __name__=='__main__':main()
