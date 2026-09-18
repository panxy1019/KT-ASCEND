"""Owned host coordinator: join system probes, then isolated DEV screen only."""
import json
from pathlib import Path
import subprocess
import time
from collect_parallel_probes import collect


def main():
    mainroot=Path('/home/admin/kt-r64-migration/payload/kt-artifacts/round6-deferred/r65-r67-c64-development-v1/round7')
    workerroot=Path('/home/admin/kt-r64-migration/round7-parallel')
    sources=[(256,mainroot/'cache-runtime-v8-probes/threshold-256',mainroot/'runtime-v8/provenance.json')]
    for label,threshold in [('a',512),('b',1024)]:
        root=workerroot/f'kt-r7-parallel-{label}-v1'
        sources.append((threshold,root/f'threshold-{threshold}',root/'runtime-v8/provenance.json'))
    deadline=time.monotonic()+6*3600
    while not all((r/'summary.json').exists() for _,r,_ in sources):
        for _,root,_ in sources:
            for path in root.glob('*/lifecycle.json'):
                report=json.loads(path.read_text())
                if report['status']=='FAIL':raise RuntimeError(f'PROBE_FAILED:{path}')
        for label in ('a','b'):
            status=workerroot/f'kt-r7-parallel-{label}-v1/chain-exit-code.txt'
            if status.exists() and status.read_text().strip()!='0':
                raise RuntimeError(f'WORKER_CHAIN_FAILED:{status}')
        if time.monotonic()>deadline:raise TimeoutError('PARALLEL_PROBE_JOIN_TIMEOUT')
        time.sleep(30)
    # Successful lifecycle reports prove model processes reaped, but check no
    # new scheduler is running on these owned workers before DEV timing.
    for c in ('kt-r67-c64-development-v1','kt-r7-parallel-a-v1','kt-r7-parallel-b-v1'):
        result=subprocess.run(['docker','exec',c,'pgrep','-f','^sglang::scheduler'],capture_output=True,text=True)
        if result.returncode!=1:raise RuntimeError(f'WORKER_NOT_IDLE:{c}:{result.stdout}')
    output=mainroot/'parallel-cache-v8-probes-collected-v1'
    collect(sources,output,mainroot/'runtime-v8/provenance.json')
    # Old parent alone was SIGSTOP'ed after launching256; its healthy child was
    # left running. Retire this exact dispatcher only after its gate completed.
    cmd=subprocess.run(['docker','exec','kt-r67-c64-development-v1','ps','-p','191211','-o','args='],capture_output=True,text=True)
    if cmd.returncode==0:
        if 'cache-runtime-v8-qualification-v1.lock' not in cmd.stdout:
            raise RuntimeError('OLD_DISPATCHER_PID_IDENTITY_CHANGED')
        subprocess.run(['docker','exec','kt-r67-c64-development-v1','kill','-KILL','191211'],check=True)
    tools='/workspace/c64-dev/round7/tools-v22/migration_bridge/round7/tools'
    prefix='/workspace/c64-dev/round7'
    print('ALL_PROBES_PASS_STARTING_SERIAL_DEV',flush=True)
    subprocess.run(['docker','exec','kt-r67-c64-development-v1','taskset','-c','128-191',
        'bash',tools+'/with_full_cpu_backing_env.sh',tools+'/threshold_sweep.py',
        '--base',prefix+'/runtime-v8','--dev',prefix+'/corpora-v2/SCHED_DEV.jsonl',
        '--profile','uniform','--probes',prefix+'/parallel-cache-v8-probes-collected-v1',
        '--output',prefix+'/dev-threshold-screen-v2'],check=True)
    print('DEV_COMPLETE_AWAITING_LOCAL_CANDIDATE_COMMIT_AND_ACK_NO_HELDOUT_STARTED',flush=True)


if __name__=='__main__':main()
