"""Host controller for two clean-process Q32 stability campaigns."""
import json
import argparse
import subprocess
import time
from pathlib import Path

ROOT=Path('/home/admin/kt-round8-qwen35')
CONTAINER='kt-round8-qwen35-prep'
PYTHON='/workspace/round8/venv/bin/python'
TOOLS='/workspace/round8/tools'
OUT=ROOT/'qwen-q32-stability-22-v2'


def command(*args,timeout=180):
    return subprocess.check_output(args,text=True,timeout=timeout)


def stop_owned():
    subprocess.run(['docker','stop','-t','60',CONTAINER],capture_output=True,text=True,timeout=120)
    deadline=time.monotonic()+90
    state=json.loads(command('docker','inspect',CONTAINER))[0]['State']
    while state['Running'] and time.monotonic()<deadline:
        time.sleep(2)
        state=json.loads(command('docker','inspect',CONTAINER))[0]['State']
    assert not state['Running'] and not state['OOMKilled'],state


def sample():
    inspect=json.loads(command('docker','inspect',CONTAINER))[0]
    return dict(timestamp=time.time(),state=inspect['State'],
        stats=json.loads(command('docker','stats','--no-stream','--format','{{json .}}',CONTAINER)),
        npu=command('npu-smi','info'))


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-campaign-a',type=Path)
    args=parser.parse_args()
    prior=None
    campaigns=('A','B')
    if args.resume_campaign_a:
        completed=json.loads((args.resume_campaign_a/'summary.json').read_text())
        assert completed['status']=='CAMPAIGN_PASS' and completed['generated_tokens']==1152
        prior=completed['reference_tokens'];campaigns=('B',)
        OUT=ROOT/'qwen-q32-stability-22-v3-resume'
    OUT.mkdir(exist_ok=False)
    state=dict(status='RUNNING',campaigns=[],started=time.time())
    if args.resume_campaign_a:state['prior_campaign_a']=str(args.resume_campaign_a)
    try:
        for campaign in campaigns:
            stop_owned()
            command('docker','start',CONTAINER)
            subprocess.run(['python3',str(ROOT/'tools/admit_q64_host.py')],
                           check=True,stdout=(OUT/f'admission-{campaign}.log').open('w'),timeout=90)
            command('docker','exec','-d',CONTAINER,'bash','-c',
                f'exec {PYTHON} {TOOLS}/start_q64.py --candidate q32-perf-launch-v2 '
                f'--admission /workspace/round8/q64-admission-v1.json '
                f'> /workspace/round8/{OUT.name}/server-{campaign}.log 2>&1')
            deadline=time.monotonic()+1200
            while True:
                ready=subprocess.run(['docker','exec',CONTAINER,PYTHON,'-c',
                    'import requests; assert requests.get("http://127.0.0.1:18400/health",timeout=3).status_code==200'],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
                if ready.returncode==0:break
                assert time.monotonic()<deadline,'server readiness timeout'
                time.sleep(5)
            with (OUT/f'client-{campaign}.log').open('w') as log:
                client=subprocess.Popen(['docker','exec',CONTAINER,PYTHON,
                    f'{TOOLS}/qwen_stability_client.py','--campaign',campaign,
                    '--output',f'/workspace/round8/{OUT.name}/campaign-{campaign}'],
                    stdout=log,stderr=subprocess.STDOUT)
                with (OUT/f'resources-{campaign}.jsonl').open('x') as resources:
                    while client.poll() is None:
                        resources.write(json.dumps(sample())+'\n');resources.flush();time.sleep(10)
                assert client.wait()==0,'stability client failed'
            summary=json.loads((OUT/f'campaign-{campaign}/summary.json').read_text())
            assert summary['status']=='CAMPAIGN_PASS' and summary['generated_tokens']==1152
            current=summary['reference_tokens']
            if prior is not None: assert current==prior,'clean restart trajectory mismatch'
            prior=current;state['campaigns'].append(campaign)
            (OUT/'controller-state.json').write_text(json.dumps(state,indent=2)+'\n')
            stop_owned()
        state.update(status='TRAJECTORY_PASS_MEMORY_AUDIT_PENDING',generated_tokens=2304)
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:
        state['updated']=time.time()
        (OUT/'controller-state.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
