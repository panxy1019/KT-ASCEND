"""Read-only attribution to the existing container; write only round artifacts."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def read(args):
    return subprocess.check_output(args, text=True, timeout=20)


def fields(text):
    return {k:int(v)*1024 for k,v in re.findall(r'^([A-Za-z_()]+):\s+(\d+) kB$',text,re.M)}


def npu0_hbm(text):
    lines = text.splitlines()
    for i,line in enumerate(lines):
        if re.search(r'\|\s*0\s+\S+\s+\|', line) and i+1 < len(lines):
            match = re.search(r'(\d+)\s*/\s*(65536)\s*\|',lines[i+1])
            if match: return dict(used_mib=int(match[1]), capacity_mib=int(match[2]))
    return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--container-id',required=True)
    a=p.parse_args()
    allowed=Path('/home/admin/kt-r64-migration/payload/kt-artifacts/round6-deferred')
    if a.study.parent!=allowed or not a.study.name.startswith('r65-r67-'):
        raise ValueError('MONITOR_PATH_SCOPE')
    if read(['docker','inspect','--format','{{.Id}}','kt-r65-static-offload']).strip()!=a.container_id:
        raise ValueError('CONTAINER_ID_DRIFT')
    started=time.monotonic()
    with (a.study/'host-telemetry-v2.jsonl').open('x') as output:
        while time.monotonic()-started<172800:
            result=dict(time_unix=time.time(),container_id=a.container_id,npu=0)
            try:
                procs=[]
                for line in read(['docker','top',a.container_id,'-eo','pid,ppid,args']).splitlines()[1:]:
                    pid,ppid,command=line.split(None,2)
                    root=Path('/proc')/pid
                    try:
                        status=fields((root/'status').read_text())
                        stat=(root/'stat').read_text().rsplit(')',1)[1].split()
                        procs.append(dict(pid=int(pid),ppid=int(ppid),command=command,
                            memory={k:status.get(k) for k in ('VmRSS','VmHWM','RssAnon','RssFile','VmSwap')},
                            minor_faults=int(stat[7]),major_faults=int(stat[9]),cpu_ticks=int(stat[11])+int(stat[12])))
                    except (FileNotFoundError,ProcessLookupError): pass
                full=read(['npu-smi','info'])
                result.update(status='SAMPLED',processes=procs,host_memory=fields(Path('/proc/meminfo').read_text()),
                    cpu_stat=Path('/proc/stat').read_text(),npu0_hbm=npu0_hbm(full),
                    npu0_usage=read(['npu-smi','info','-t','usages','-i','0','-c','0']),
                    npu0_process_memory=read(['npu-smi','info','-t','proc-mem','-i','0','-c','0']))
                cgroup=read(['docker','inspect','--format','{{.State.Pid}}',a.container_id]).strip()
                for line in (Path('/proc')/cgroup/'cgroup').read_text().splitlines():
                    if line.startswith('0::'):
                        cg=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
                        result['cgroup']={n:(cg/n).read_text() for n in ('memory.current','memory.stat','memory.events','memory.pressure') if (cg/n).exists()}
            except Exception as exc: result.update(status='MONITOR_ERROR',error=repr(exc))
            output.write(json.dumps(result)+'\n');output.flush()
            if (a.study/'summary.json').exists():break
            time.sleep(10)


if __name__=='__main__':main()
