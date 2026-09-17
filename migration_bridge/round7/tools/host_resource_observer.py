"""Host-only read-only contention observer, never stops or changes workloads."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def process_ids(text):
    return [int(x) for x in re.findall(r'Process id:\s*(\d+)',text)]


def memory_blockers(mem,cgroup_swap,current_vm,initial_vm):
    blockers=[]
    if cgroup_swap:blockers.append(dict(reason='CONTAINER_SWAP_USED',bytes=cgroup_swap))
    if current_vm!=initial_vm:blockers.append(dict(reason='ACTIVE_HOST_SWAPPING'))
    if mem['MemAvailable']<max(64*2**30,.2*mem['MemTotal']):
        blockers.append(dict(reason='HOST_MEMORY_RESERVE'))
    return blockers


def assess_window(samples,start,end,max_gap=20):
    relevant=sorted((s for s in samples if start-max_gap<=s['time_unix']<=end+max_gap),
                    key=lambda x:x['time_unix'])
    before=[s for s in relevant if s['time_unix']<=start]
    after=[s for s in relevant if s['time_unix']>=end]
    if not before or not after:return dict(status='UNOBSERVED',formal_eligible=False)
    bounded=[s for s in relevant if before[-1]['time_unix']<=s['time_unix']<=after[0]['time_unix']]
    if any(b['time_unix']-a['time_unix']>max_gap for a,b in zip(bounded,bounded[1:])):
        return dict(status='STALE',formal_eligible=False)
    clean=all(s['status']=='EXCLUSIVE' for s in bounded)
    return dict(status='EXCLUSIVE' if clean else 'CONTENDED_OR_NOT_YET_QUIET',
        formal_eligible=clean,samples=len(bounded),
        criterion='NO_OTHER_MODEL_CONTROLLER_OR_NPU; OUTSIDE_CPU_TOTAL_LE2_MAX_LE1; 60S_QUIET; CGROUP_ZERO_SWAP_HOST_NO_ACTIVE_SWAP_V2',
        scope='READ_ONLY_HOST_CONTENTION_PROXY_NOT_DIRECT_DRAM_BANDWIDTH_MEASUREMENT')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hours',type=float,default=24);a=p.parse_args()
    root='/home/admin/kt-r64-migration/payload/kt-artifacts/round6-deferred/r65-r67-c64-development-v1/round7/'
    assert a.output.resolve().is_relative_to(Path(root).resolve()),'OWNED_OUTPUT_SCOPE'
    a.output.mkdir(parents=True,exist_ok=False)
    hz=os.sysconf('SC_CLK_TCK')
    def swap_counters():
        return {k:int(v) for k,v in (line.split() for line in Path('/proc/vmstat').read_text().splitlines())
                if k in ('pswpin','pswpout')}
    initial_vm=swap_counters()
    init_pid=int(subprocess.check_output(['docker','inspect','--format','{{.State.Pid}}',
        'kt-r67-c64-development-v1'],text=True,timeout=5).strip())
    assert init_pid>0,'OWNED_CONTAINER_NOT_RUNNING'
    group=next(line[3:] for line in Path(f'/proc/{init_pid}/cgroup').read_text().splitlines() if line.startswith('0::'))
    swap_file=Path('/sys/fs/cgroup')/group.lstrip('/')/'memory.swap.current'
    def snapshot():
        data={}
        for path in Path('/proc').glob('[0-9]*'):
            try:
                raw=(path/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
                command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                data[int(path.name)]=(int(tail[11])+int(tail[12]),command)
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        return data
    def write(state):
        with (a.output/'samples.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        temporary=a.output/'latest.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(a.output/'latest.json')
    previous=snapshot();previous_time=time.monotonic();quiet_since=None
    deadline=time.monotonic()+a.hours*3600
    heavy=re.compile(r'sglang|launch_server|run_serial_sweep|farm_worker|operator_oracle|run_operator_matrix')
    while time.monotonic()<deadline:
        time.sleep(5);blockers=[];outside=[]
        try:
            top=subprocess.check_output(['docker','top','kt-r67-c64-development-v1','-eo','pid'],text=True,timeout=5)
            owned={int(x) for x in top.splitlines()[1:]};owned.add(os.getpid())
            current=snapshot();now=time.monotonic();elapsed=now-previous_time
            for pid,(ticks,command) in current.items():
                if pid in owned:continue
                used=max(0,ticks-previous.get(pid,(ticks,''))[0])/hz/elapsed
                if used:outside.append(used)
                if heavy.search(command):blockers.append(dict(reason='OTHER_MODEL_OR_CONTROLLER',pid=pid,
                    executable=Path(command.split()[0]).name if command.split() else ''))
            previous=current;previous_time=now
            if sum(outside)>2 or max(outside,default=0)>1:
                blockers.append(dict(reason='OUTSIDE_CPU_BUSY',cores=sum(outside)))
            for device in range(8):
                info=subprocess.check_output(['npu-smi','info','-t','proc-mem','-i',str(device)],text=True,timeout=5)
                others=[pid for pid in process_ids(info) if pid not in owned]
                if others:blockers.append(dict(reason='OTHER_NPU_PROCESS',device=device,pids=others))
            mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                 if line.endswith('kB')}
            current_vm=swap_counters();cgroup_swap=int(swap_file.read_text())
            blockers+=memory_blockers(mem,cgroup_swap,current_vm,initial_vm)
        except Exception as error:blockers.append(dict(reason='OBSERVATION_UNAVAILABLE',error=repr(error)))
        if blockers:quiet_since=None
        elif quiet_since is None:quiet_since=time.monotonic()
        exclusive=quiet_since is not None and time.monotonic()-quiet_since>=60
        write(dict(status='EXCLUSIVE' if exclusive else 'WAIT',time_unix=time.time(),blockers=blockers,
            outside_cpu_cores=sum(outside),no_other_task_modified=True,
            initial_swap_counters=initial_vm,memory_policy='CGROUP_ZERO_SWAP_HOST_NO_ACTIVE_SWAP_V2'))


if __name__=='__main__':main()
