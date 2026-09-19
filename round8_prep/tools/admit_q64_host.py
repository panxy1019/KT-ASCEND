"""Host-only measured Q64 admission. Never kills or changes any process."""
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from round8_resource_guard import reject_reasons

ROOT=Path('/home/admin/kt-round8-qwen35')
CONTAINER='kt-round8-qwen35-prep'


def run(*args):
    return subprocess.check_output(args,text=True,timeout=45)


def counters():
    return {k:int(v) for k,v in (l.split() for l in Path('/proc/vmstat').read_text().splitlines())
            if k in ('pswpin','pswpout')}


def main():
    before=counters();t0=time.monotonic()
    time.sleep(30)
    after=counters()
    old=json.loads(run('docker','inspect','kt-r67-c64-development-v1'))[0]
    current=json.loads(run('docker','inspect',CONTAINER))[0]
    assert current['HostConfig']['CpusetCpus']=='128-191'
    assert current['HostConfig']['CpusetMems']=='4,5'
    assert current['HostConfig']['Memory']==384*2**30
    assert current['HostConfig']['MemorySwap']==384*2**30
    devices=[d['PathOnHost'] for d in current['HostConfig']['Devices']]
    assert '/dev/davinci2' in devices
    assert not any(re.fullmatch(r'/dev/davinci[0-9]+',d) and d!='/dev/davinci2' for d in devices)
    proc=run('npu-smi','info','-t','proc-mem','-i','2','-c','0')
    usages=run('npu-smi','info','-t','usages','-i','2','-c','0')
    capacity=int(re.search(r'HBM Capacity\(MB\)\s*:\s*(\d+)',usages)[1])*2**20
    percent=int(re.search(r'HBM Usage Rate\(%\)\s*:\s*(\d+)',usages)[1])
    topology=run('lscpu','-p=CPU,CORE,SOCKET,NODE')
    rows=[l.split(',') for l in topology.splitlines() if not l.startswith('#')]
    owned=[r for r in rows if 128<=int(r[0])<=191]
    assert {r[3] for r in owned}=={'4','5'}
    mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
    swap=int(run('docker','exec',CONTAINER,'cat','/sys/fs/cgroup/memory.swap.current'))
    x=dict(inventory_age_seconds=0,round7_perf_active=old['State']['Running'],
        round7_npus=[2] if old['State']['Running'] else [],round7_cpus=[],
        requested_npu=2,requested_cpus=list(range(128,192)),
        host_available_bytes=mem['MemAvailable'],host_total_bytes=mem['MemTotal'],
        predicted_private_bytes=384*2**30,swap_used_bytes=mem['SwapTotal']-mem['SwapFree'],
        npu_free_bytes=int(capacity*(99-percent)/100),predicted_npu_peak_bytes=44*2**30,
        disk_free_bytes=shutil.disk_usage(ROOT).free,download_conversion_budget_bytes=20*2**30,
        other_npu_processes=[] if 'No process in device.' in proc else [proc],
        same_host_inventory=True,physical_core_count=len({(r[1],r[2]) for r in owned}),
        swap_observation_seconds=time.monotonic()-t0,
        host_swapin_delta=after['pswpin']-before['pswpin'],
        host_swapout_delta=after['pswpout']-before['pswpout'],cgroup_swap_current=swap,
        host_page_size_bytes=os.sysconf('SC_PAGE_SIZE'))
    reasons=reject_reasons(x)
    result=dict(timestamp=time.time(),admitted=not reasons,reasons=reasons,inventory=x,
                npu_proc_raw=proc,npu_usage_raw=usages,physical_topology=owned)
    (ROOT/'q64-admission-v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(bool(reasons))


if __name__=='__main__':main()
