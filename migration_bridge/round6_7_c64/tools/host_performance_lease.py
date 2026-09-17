"""Read-only host admission. Never stops a container or any other task.

Run only when the operator and canary gates have passed. The lease goes WAIT
while protected controllers are alive, even between their model processes.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--hours',type=float,default=24);a=p.parse_args()
assert str(a.output).startswith('/home/admin/kt-r64-migration/payload/kt-artifacts/round6-deferred/r65-r67-c64-development-v1/')
name='kt-r67-c64-development-v1'
heavy=re.compile(r'sglang|run_serial_sweep|farm_worker|operator_oracle|run_operator_matrix|launch_server')
hz=os.sysconf('SC_CLK_TCK')
def snapshot():
    data={}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            raw=(path/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
            command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            data[int(path.name)]=(int(tail[11])+int(tail[12]),command)
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return data
def write(value):
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(a.output)
deadline=time.monotonic()+a.hours*3600
idle_since=None;previous=snapshot();previous_time=time.monotonic()
while time.monotonic()<deadline:
    time.sleep(5)
    blockers=[]
    try:
        top=subprocess.check_output(['docker','top',name,'-eo','pid'],text=True,timeout=10)
        owned={int(x) for x in top.splitlines()[1:]};owned.add(os.getpid())
        current=snapshot();now=time.monotonic();elapsed=now-previous_time
        outside=[]
        for pid,(ticks,command) in current.items():
            if pid in owned:continue
            used=max(0,ticks-previous.get(pid,(ticks,''))[0])/hz/elapsed
            if heavy.search(command):blockers.append(dict(pid=pid,reason='OTHER_MODEL_OR_CONTROLLER',command=command))
            if used:outside.append((pid,used))
        if sum(v for _,v in outside)>2 or any(v>1 for _,v in outside):
            blockers.append(dict(reason='HOST_CPU_BUSY',cores=sum(v for _,v in outside)))
        previous=current;previous_time=now
        for device in (0,1,3,4,5,6,7):
            result=subprocess.check_output(['npu-smi','info','-t','proc-mem','-i',str(device)],text=True,timeout=5)
            if 'No process in device.' not in result:blockers.append(dict(reason='OTHER_NPU_PROCESS',device=device))
        mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.endswith('kB')}
        if mem['SwapTotal']!=mem['SwapFree'] or mem['MemAvailable']<max(64*2**30,.2*mem['MemTotal']):
            blockers.append(dict(reason='HOST_MEMORY_ADMISSION'))
    except Exception as exc:blockers.append(dict(reason='OBSERVATION_UNAVAILABLE',error=repr(exc)))
    if blockers:idle_since=None
    elif idle_since is None:idle_since=time.monotonic()
    exclusive=idle_since is not None and time.monotonic()-idle_since>=60
    write(dict(status='EXCLUSIVE' if exclusive else 'WAIT',container=name,time_unix=time.time(),
               blockers=blockers,criterion='NO_OTHER_MODEL_CONTROLLER_OR_NPU; OUTSIDE_CPU_TOTAL_LE2_MAX_LE1; 60S_IDLE',
               no_other_task_modified=True))
write(dict(status='EXPIRED',container=name,time_unix=time.time()))
