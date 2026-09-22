"""Run three pre-registered Dynamic thresholds on DEV only, sequentially."""
import json
import subprocess
import time
from pathlib import Path

ROOT=Path('/home/admin/kt-round8-qwen35');CONTAINER='kt-round8-qwen35-stability'
PYTHON='/workspace/round8/venv/bin/python';TOOLS='/workspace/round8/tools'
OUT=ROOT/'qwen-dynamic-dev-v1'


def command(*args,timeout=180):return subprocess.check_output(args,text=True,timeout=timeout)
def stop_owned():
    subprocess.run(['docker','stop','-t','60',CONTAINER],capture_output=True,text=True,timeout=120)
    deadline=time.monotonic()+90
    while True:
        state=json.loads(command('docker','inspect',CONTAINER))[0]['State']
        if not state['Running']:assert not state['OOMKilled'];return state
        assert time.monotonic()<deadline,'delayed container exit timeout';time.sleep(2)
def sample():return dict(timestamp=time.time(),npu=command('npu-smi','info'),
    stats=json.loads(command('docker','stats','--no-stream','--format','{{json .}}',CONTAINER)))


def main():
    OUT.mkdir(exist_ok=False);state=dict(status='RUNNING',completed=[],started=time.time())
    try:
        for threshold in (256,512,1024):
            name=f'q32-dynamic-{threshold}-v1';state['current']=name
            (OUT/'controller-state.json').write_text(json.dumps(state,indent=2)+'\n')
            subprocess.run(['python3',str(ROOT/'tools/admit_qwen_q32_stability.py')],check=True,
                stdout=(OUT/f'{name}-admission.log').open('w'),timeout=90)
            command('docker','start',CONTAINER)
            command('docker','exec','-d',CONTAINER,'bash','-c',
                f'exec {PYTHON} {TOOLS}/start_q64.py --candidate {name} '
                f'--admission /workspace/round8/q32-stability-admission.json '
                f'> /workspace/round8/{name}/server.log 2>&1')
            deadline=time.monotonic()+1200
            while True:
                ready=subprocess.run(['docker','exec',CONTAINER,PYTHON,'-c',
                    'import requests; assert requests.get("http://127.0.0.1:18400/health",timeout=3).status_code==200'],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
                if ready.returncode==0:break
                assert time.monotonic()<deadline,'server readiness timeout';time.sleep(5)
            with (OUT/f'{name}-client.log').open('w') as log:
                client=subprocess.Popen(['docker','exec',CONTAINER,PYTHON,
                    f'{TOOLS}/qwen_dynamic_dev_client.py','--threshold',str(threshold),
                    '--dev','/workspace/round8/qwen-scheduler-v1/SCHED_DEV.jsonl',
                    '--audit-root',f'/workspace/round8/{name}'],stdout=log,stderr=subprocess.STDOUT)
                with (OUT/f'{name}-resources.jsonl').open('x') as resources:
                    while client.poll() is None:
                        resources.write(json.dumps(sample())+'\n');resources.flush();time.sleep(10)
                assert client.wait()==0,'DEV client failed'
            summary=json.loads((ROOT/name/'dynamic-dev-summary.json').read_text())
            assert summary['status'] in ('PASS','WORKLOAD_SHIFT_NOT_OBSERVED')
            state['completed'].append(dict(candidate=name,status=summary['status']))
            stop_owned()
        state['status']='DEV_SWEEP_COMPLETE_SELECTION_PENDING'
    except BaseException as error:
        state.update(status='FAILED',error=repr(error))
        try:
            state['owned_container_after_failure']=stop_owned()
        except BaseException as cleanup_error:
            state['cleanup_error']=repr(cleanup_error)
        raise
    finally:
        state['updated']=time.time();(OUT/'controller-state.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
