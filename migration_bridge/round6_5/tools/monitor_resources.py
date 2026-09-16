"""Read-only host sampler; only R6.5 container processes and NPU0 are attributed.

RSS sums can double-count shared pages. Host cache is system-wide, NOT charged
to this workload. Missing counters remain null, never interpreted as zero.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def read(command):
    return subprocess.check_output(command, text=True, timeout=15)


def kb_fields(text):
    return {key: int(value)*1024 for key, value in re.findall(
        r'^([A-Za-z_()]+):\s+(\d+) kB$', text, re.M)}


def process(pid):
    root=Path('/proc')/str(pid)
    status=kb_fields((root/'status').read_text())
    stat=(root/'stat').read_text().rsplit(')',1)[1].split()
    rollup=root/'smaps_rollup'
    pss=kb_fields(rollup.read_text()).get('Pss') if rollup.exists() else None
    cgroup=None
    for line in (root/'cgroup').read_text().splitlines():
        if line.startswith('0::'):
            cg=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
            cgroup={name:(cg/name).read_text() for name in
                    ('memory.current','memory.max','memory.stat','memory.events','memory.pressure')
                    if (cg/name).exists()}
    return dict(pid=pid, pss_bytes=pss, cgroup=cgroup, memory={k:status.get(k) for k in
        ('VmRSS','VmHWM','RssAnon','RssFile','RssShmem','VmSwap')},
        minor_faults=int(stat[7]),major_faults=int(stat[9]),
        cpu_ticks=int(stat[11])+int(stat[12]),
        numa_maps=(root/'numa_maps').read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--container',required=True)
    p.add_argument('--container-id',required=True)
    p.add_argument('--duration',type=int,default=10800)
    p.add_argument('--interval',type=int,default=10)
    args=p.parse_args()
    if args.container!='kt-r65-static-offload' or not 1<=args.duration<=172800 or not 1<=args.interval<=60:
        raise ValueError('MONITOR_SCOPE')
    identity=read(['docker','inspect','--format','{{.Id}}',args.container]).strip()
    if identity!=args.container_id:raise ValueError('CONTAINER_ID_DRIFT')
    deadline=time.monotonic()+args.duration
    while time.monotonic()<deadline:
        item=dict(time_unix=time.time(),container_id=identity,npu=0)
        try:
            output=read(['docker','top',identity,'-eo','pid,ppid,args'])
            entries=[line.split(None,2) for line in output.splitlines()[1:]]
            procs=[]
            for pid,ppid,command in entries:
                try:
                    row=process(int(pid));row.update(ppid=int(ppid),command=command)
                    procs.append(row)
                except (FileNotFoundError,ProcessLookupError):pass
            item.update(status='SAMPLED',processes=procs,
                host_memory=kb_fields(Path('/proc/meminfo').read_text()),
                cpu_stat=Path('/proc/stat').read_text(),
                npu0_process_memory=read(['npu-smi','info','-t','proc-mem','-i','0','-c','0']),
                npu0_usage=read(['npu-smi','info','-t','usages','-i','0','-c','0']))
        except Exception as exc:item.update(status='MONITOR_ERROR',exception=repr(exc))
        print(json.dumps(item),flush=True)
        time.sleep(args.interval)


if __name__=='__main__':main()
