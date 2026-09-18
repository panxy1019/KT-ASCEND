"""Host-only read-only contention observer, never stops or changes workloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def process_ids(text):
    return [int(x) for x in re.findall(r'Process id:\s*(\d+)',text)]


def memory_blockers(mem,cgroup_swap,current_vm,previous_vm):
    blockers=[]
    if cgroup_swap:blockers.append(dict(reason='CONTAINER_SWAP_USED',bytes=cgroup_swap))
    if current_vm!=previous_vm:blockers.append(dict(reason='ACTIVE_HOST_SWAPPING'))
    if mem['MemAvailable']<max(64*2**30,.2*mem['MemTotal']):
        blockers.append(dict(reason='HOST_MEMORY_RESERVE'))
    return blockers


def unified_group(text,allow_root=False):
    matches=[line[3:] for line in text.splitlines() if line.startswith('0::')]
    if len(matches)!=1 or (matches[0]=='/' and not allow_root):raise ValueError('OWNERSHIP_CGROUP_UNAVAILABLE')
    return matches[0]


def in_owned_group(candidate,owned_group):
    return candidate==owned_group or candidate.startswith(owned_group+'/')


def outside_usage(current,previous,owned_group,observer_pid,hz,elapsed):
    """Attribute each snapshot by actual cgroup, not an earlier docker-top list."""
    if elapsed<=0:raise ValueError('NONPOSITIVE_OBSERVER_INTERVAL')
    result=[]
    for pid,row in current.items():
        if pid==observer_pid or in_owned_group(row['cgroup'],owned_group):continue
        prior=previous.get(pid)
        # A reusedPID is not the previous process's CPU counter/lifetime.
        ticks=prior['ticks'] if prior and prior['start_ticks']==row['start_ticks'] else row['ticks']
        used=max(0,row['ticks']-ticks)/hz/elapsed
        if used:result.append(dict(pid=pid,cores=used,cgroup=row['cgroup'],
            start_ticks=row['start_ticks'],executable=Path(row['command'].split()[0]).name
            if row['command'].split() else ''))
    return sorted(result,key=lambda x:x['cores'],reverse=True)


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


def fresh_exclusive(state,now,max_age=20):
    return (state.get('status')=='EXCLUSIVE' and not state.get('blockers')
            and 0<=now-state.get('time_unix',0)<=max_age)


def wait_for_exclusive(samples_path,timeout=1800):
    """Wait outside timing; never substitute entrance admission for window audit."""
    deadline=time.monotonic()+timeout
    latest=Path(samples_path).with_name('latest.json')
    while time.monotonic()<deadline:
        try:
            state=json.loads(latest.read_text())
            if fresh_exclusive(state,time.time()):return state
        except (OSError,ValueError):pass
        time.sleep(5)
    raise TimeoutError('HOST_NOT_FRESH_EXCLUSIVE_BEFORE_MEASUREMENT')


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
    initial_vm=swap_counters();previous_vm=initial_vm
    init_pid=int(subprocess.check_output(['docker','inspect','--format','{{.State.Pid}}',
        'kt-r67-c64-development-v1'],text=True,timeout=5).strip())
    assert init_pid>0,'OWNED_CONTAINER_NOT_RUNNING'
    group=unified_group(Path(f'/proc/{init_pid}/cgroup').read_text())
    swap_file=Path('/sys/fs/cgroup')/group.lstrip('/')/'memory.swap.current'
    def snapshot():
        data={}
        for path in Path('/proc').glob('[0-9]*'):
            try:
                raw=(path/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
                command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                process_group=unified_group((path/'cgroup').read_text(),allow_root=True)
                data[int(path.name)]=dict(ticks=int(tail[11])+int(tail[12]),command=command,
                    start_ticks=int(tail[19]),cgroup=process_group)
            except (FileNotFoundError,ProcessLookupError):pass
        return data
    def write(state):
        with (a.output/'samples.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        temporary=a.output/'latest.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(a.output/'latest.json')
    previous=snapshot();previous_time=time.monotonic();quiet_since=None
    deadline=time.monotonic()+a.hours*3600
    heavy=re.compile(r'sglang|launch_server|run_serial_sweep|farm_worker|operator_oracle|run_operator_matrix')
    while time.monotonic()<deadline:
        time.sleep(5);blockers=[];outside=[];details=[];memory_observation={}
        try:
            current=snapshot();now=time.monotonic();elapsed=now-previous_time
            owned={pid for pid,row in current.items() if in_owned_group(row['cgroup'],group)}
            owned.add(os.getpid())
            details=outside_usage(current,previous,group,os.getpid(),hz,elapsed)
            outside=[row['cores'] for row in details]
            for pid,row in current.items():
                if pid in owned:continue
                command=row['command']
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
            memory_observation=dict(host_mem_available=mem['MemAvailable'],host_mem_total=mem['MemTotal'],
                host_swap_used=mem['SwapTotal']-mem['SwapFree'],cgroup_swap_bytes=cgroup_swap,
                current_swap_counters=current_vm)
            # VM counters are cumulative. Activity is the interval delta;
            # one old swap-in must not latch WAIT forever. Any new activity
            # still resets the full60s quiet period and invalidates its window.
            memory_observation['previous_swap_counters']=previous_vm
            blockers+=memory_blockers(mem,cgroup_swap,current_vm,previous_vm)
            previous_vm=current_vm
        except Exception as error:blockers.append(dict(reason='OBSERVATION_UNAVAILABLE',error=repr(error)))
        if blockers:quiet_since=None
        elif quiet_since is None:quiet_since=time.monotonic()
        exclusive=quiet_since is not None and time.monotonic()-quiet_since>=60
        write(dict(status='EXCLUSIVE' if exclusive else 'WAIT',time_unix=time.time(),blockers=blockers,
            outside_cpu_cores=sum(outside),no_other_task_modified=True,
            initial_swap_counters=initial_vm,memory=memory_observation,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            memory_policy='CGROUP_ZERO_SWAP_INTERVAL_ACTIVITY_WITH_60S_QUIET_V3',
            ownership_method='V3_ACTUAL_SNAPSHOT_CGROUP_AND_PROCESS_START_ID',
            owned_cgroup=group,outside_cpu_processes=details))


if __name__=='__main__':main()
