"""One untimed Dynamic DEV threshold on a pre-probed .22 lane."""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT=Path('/home/admin/kt-round8-qwen35')
PY='/workspace/round8/venv/bin/python'
def cmd(*args):return subprocess.check_output(args,text=True,timeout=120)

def main():
    p=argparse.ArgumentParser();p.add_argument('--threshold',type=int,choices=(256,512,1024),required=True)
    a=p.parse_args();lane='b' if a.threshold==512 else 'a'
    container=f'kt-round8-qwen35-dynamic-{lane}'
    name=f'q32-dynamic-{a.threshold}-v1';out=ROOT/name
    state_path=out/'lane-state.json'
    assert not state_path.exists()
    state=dict(status='RUNNING',container=container,threshold=a.threshold,timed=False)
    state_path.write_text(json.dumps(state,indent=2))
    try:
        mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
        assert mem['MemAvailable']>256*2**30,'host reserve insufficient'
        config=json.loads((out/'resolved-command.json').read_text())
        inspect=json.loads(cmd('docker','inspect',container))[0]
        assert inspect['HostConfig']['CpusetMems']==('2,3' if lane=='b' else '0,1')
        admission=out/'admission.json'
        admission.write_text(json.dumps(dict(admitted=True,timestamp=time.time(),
            host='110.129.0.22',container=container,available_bytes=mem['MemAvailable'],
            purpose='UNTIMED_DEV',parallel=True)))
        cmd('docker','exec','-d',container,'bash','-c',
            f'exec {PY} /workspace/round8/tools/start_q64.py --candidate {name} '
            f'--admission /workspace/round8/{name}/admission.json '
            f'> /workspace/round8/{name}/server.log 2>&1')
        deadline=time.monotonic()+1200
        while True:
            r=subprocess.run(['docker','exec',container,PY,'-c',
                'import requests; assert requests.get("http://127.0.0.1:18400/health",timeout=3).status_code==200'],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
            if r.returncode==0:break
            assert time.monotonic()<deadline,'readiness timeout'
            time.sleep(5)
        with (out/'client.log').open('x') as log:
            subprocess.run(['docker','exec',container,PY,'/workspace/round8/tools/qwen_dynamic_dev_client.py',
                '--threshold',str(a.threshold),'--dev','/workspace/round8/qwen-scheduler-v1/SCHED_DEV.jsonl',
                '--audit-root',f'/workspace/round8/{name}'],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
        state['status']='DEV_COMPLETE_PENDING_AUDIT'
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:
        state_path.write_text(json.dumps(state,indent=2))
        subprocess.run(['docker','stop','-t','60',container],capture_output=True,timeout=150)

if __name__=='__main__':main()
