"""Live guard for one explicitly named Round9 container; never touches training."""
import argparse
import json
import pathlib
import re
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--container', required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
p.add_argument('--training-pids', nargs='+', type=int, required=True)
p.add_argument('--ready-file', type=pathlib.Path, required=True)
p.add_argument('--seconds', type=int, default=3600)
a = p.parse_args()
if not re.fullmatch(r'kt-round9-full-model-v[0-9]+', a.container):
    raise ValueError('guard may only control an owned full-model container')
if a.output.exists() or a.ready_file.exists():
    raise FileExistsError(a.output)
started = time.monotonic()
system_baseline_mib = None
baseline_samples = []


def command(args):
    return subprocess.check_output(args, text=True, timeout=20)


def stop_owned(reason):
    try:
        with a.output.open('a') as f:
            f.write(json.dumps({'time':time.time(), 'action':'STOP_OWNED', 'reason':reason})+'\n')
    finally:
        subprocess.run(['docker','stop','--time','10',a.container], check=False, timeout=30)


def parse_physical15(raw):
    line = next((line for line in raw.splitlines() if '0000:83:00.0' in line), None)
    if line is None:
        raise RuntimeError('physical15 telemetry row missing')
    hbm = re.findall(r'(\d+)\s*/\s*(\d+)',line)
    if not hbm or int(hbm[-1][1]) != 65536:
        raise RuntimeError('physical15 telemetry capacity malformed')
    used = int(hbm[-1][0])
    processes = {}
    for line in raw.splitlines():
        m = re.match(r'\|\s*7\s+1\s*\|\s*(\d+)\s*\|[^|]*\|\s*(\d+)\s*\|',line)
        if m:
            processes[int(m[1])] = int(m[2])
    return used, processes


def hbm_budget_exceeded(used, owned, training, system_baseline):
    #npu-smi board and process counters are not atomic. Do not infer a training
    #spike by subtracting an asynchronously updated owned counter from board use.
    #Keep absolute board/owned bounds and the pre-start system+training bound.
    return used > 56832 or owned > 41472 or training + system_baseline > 15360


def stable_system_baseline(samples):
    recent = samples[-3:]
    if len(recent) != 3 or max(recent)-min(recent) > 64:
        return None
    return max(recent)


while True:
    try:
        state = json.loads(command(['docker','inspect',a.container]))[0]
        waiting = state['State']['Status'] == 'created'
        if waiting and time.monotonic()-started > 60:
            raise RuntimeError('container was not started within60s of guard')
        if not state['State']['Running'] and not waiting:
            with a.output.open('a') as f:
                f.write(json.dumps({'time':time.time(),'action':'OWNED_EXITED','state':state['State']})+'\n')
            break
        raw = command(['npu-smi','info'])
        used, processes = parse_physical15(raw)
        top = '' if waiting else command(['docker','top',a.container,'-eo','pid'])
        owned_pids = {int(s.strip()) for s in top.splitlines()[1:] if s.strip().isdigit()}
        outsiders = set(processes)-owned_pids-set(a.training_pids)
        if outsiders:
            raise RuntimeError(f'unbound physical15 process identity: {sorted(outsiders)}')
        training = sum(processes.get(pid,0) for pid in a.training_pids)
        owned = sum(value for pid,value in processes.items() if pid in owned_pids)
        if system_baseline_mib is None:
            if not waiting or owned != 0:
                raise RuntimeError('guard must establish baseline before container start')
            candidate = used - training
            if not 0 <= candidate < 15360:
                raise RuntimeError('invalid pre-start system HBM baseline')
            baseline_samples.append(candidate)
            system_baseline_mib = stable_system_baseline(baseline_samples)
        item = {'time':time.time(),'hbm_used_mib':used,'training_process_mib':training,
                'owned_process_mib':owned,'nonowned_board_mib':used-owned,
                'system_baseline_mib':system_baseline_mib,
                'baseline_samples_mib':baseline_samples,
                'estimated_nonowned_board_mib':None if system_baseline_mib is None else training+system_baseline_mib,
                'container_status':state['State']['Status'],'processes':processes,'raw_npu_smi':raw}
        meminfo = dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
        item['host_mem_available_kib'] = int(meminfo['MemAvailable'].split()[0])
        item['host_swap_free_kib'] = int(meminfo['SwapFree'].split()[0])
        pressure = pathlib.Path('/proc/pressure/memory')
        item['memory_pressure'] = pressure.read_text() if pressure.exists() else 'UNAVAILABLE_KERNEL_PSI'
        pid = state['State']['Pid']
        cg = [] if waiting else pathlib.Path(f'/proc/{pid}/cgroup').read_text().splitlines()
        unified = next((line.split(':',2)[2] for line in cg if line.startswith('0::')), None)
        if unified and (pathlib.Path('/sys/fs/cgroup') / unified.lstrip('/') / 'memory.current').exists():
            cgroot = pathlib.Path('/sys/fs/cgroup') / unified.lstrip('/')
            item['owned_cgroup'] = {name:(cgroot/name).read_text().strip()
                for name in ('memory.current','memory.peak','memory.swap.current','memory.events')
                if (cgroot/name).exists()}
        else:
            legacy = next((line.split(':',2)[2] for line in cg
                           if 'memory' in line.split(':',2)[1].split(',')), None)
            if legacy:
                cgroot = pathlib.Path('/sys/fs/cgroup/memory') / legacy.lstrip('/')
                item['owned_cgroup'] = {name:(cgroot/name).read_text().strip()
                    for name in ('memory.usage_in_bytes','memory.max_usage_in_bytes',
                                 'memory.memsw.usage_in_bytes','memory.failcnt')
                    if (cgroot/name).exists()}
        with a.output.open('a') as f:
            f.write(json.dumps(item)+'\n')
        if item['host_mem_available_kib'] < 64*1024**2:
            stop_owned('host available memory fell below64GiB reserve')
            break
        #Physical-board domain; 15GiB nonowned envelope includes system baseline.
        #Stop0.5GiB before41GiB owned/56GiB total budget, not at the boundary.
        if (waiting and used > 15360) or hbm_budget_exceeded(used, owned, training, system_baseline_mib or 0):
            stop_owned('shared HBM budget exceeded')
            break
        if waiting and system_baseline_mib is not None and not a.ready_file.exists():
            a.ready_file.write_text(json.dumps({'time':time.time(),'status':'GUARD_ARMED'})+'\n')
        if time.monotonic()-started > a.seconds:
            stop_owned('bounded live-monitor window ended')
            break
    except Exception as exc:
        stop_owned(f'telemetry/guard failure: {exc!r}')
        raise
    time.sleep(1)
