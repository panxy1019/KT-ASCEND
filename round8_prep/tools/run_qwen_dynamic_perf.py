"""Fresh same-NPU ABBA; refuses active parallel lanes and unapproved results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from run_qwen_abba import snapshot, ensure_stopped

ROOT=Path('/home/admin/kt-round8-qwen35');PY='/workspace/round8/venv/bin/python'
CONTAINER='kt-round8-qwen35-prep';NAME='qwen-dynamic-perf-v2'
def command(*args):return subprocess.check_output(args,text=True,timeout=120)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--serve',choices=['static','dynamic']);a=p.parse_args()
    if a.serve:
        root=Path('/workspace/round8');cfg=json.loads((root/'qwen-dynamic-candidate-v2'/f'perf-{a.serve}.json').read_text())
        assert '--forward-hooks' not in cfg['argv']
        for path,digest in cfg['code_hashes'].items():assert sha(Path(path))==digest,path
        env=os.environ.copy();env.update(cfg['env']);os.execvpe(cfg['argv'][0],cfg['argv'],env)
    frozen=ROOT/'qwen-dynamic-candidate-v2'
    ack=json.loads((frozen/'freeze-ack.json').read_text())
    assert ack['candidate_sha256']==sha(frozen/'candidate.json')
    contract=json.loads((frozen/'candidate.json').read_text())
    for tool,digest in contract['tools_sha256'].items():assert sha(ROOT/'tools'/tool)==digest,tool
    release=json.loads((frozen/'qualification-release.json').read_text())
    assert release['heldout']=='PASS' and release['cmmlu']=='PASS'
    for mode in ('heldout','cmmlu'):
        assert release[mode+'_summary_sha256']==sha(ROOT/f'qwen-dynamic-{mode}-v2/summary.json')
    out=ROOT/NAME;out.mkdir(exist_ok=False)
    state=dict(status='RUNNING',completed=[])
    def save():(out/'state.json').write_text(json.dumps(state,indent=2))
    save();owns_process=False
    try:
        for block in range(3):
            for position,profile in enumerate(('static','dynamic','dynamic','static')):
                for lane in ('a','b'):
                    info=json.loads(command('docker','inspect',f'kt-round8-qwen35-dynamic-{lane}'))[0]
                    assert not info['State']['Running'],'PARALLEL_LANE_ACTIVE'
                info=json.loads(command('docker','inspect',CONTAINER))[0]
                assert not info['State']['Running'],'OWNED_PREP_BUSY'
                assert info['HostConfig']['CpusetCpus']=='128-191' and info['HostConfig']['CpusetMems']=='4,5'
                cfg=frozen/f'perf-{profile}.json'
                assert sha(cfg)==json.loads((frozen/'candidate.json').read_text())['configs_sha256']['perf-'+profile]
                cell=f'{profile}-{block}-{position}'
                command('docker','start',CONTAINER);owns_process=True
                with (out/f'{cell}-admission.log').open('x') as log:
                    subprocess.run(['python3',str(ROOT/'tools/admit_q64_host.py')],check=True,stdout=log,timeout=90)
                command('docker','exec','-d',CONTAINER,'bash','-c',
                    f'{PY} /workspace/round8/tools/run_qwen_dynamic_perf.py --serve {profile} > /workspace/round8/{NAME}/{cell}-server.log 2>&1')
                deadline=time.monotonic()+1200
                while True:
                    r=subprocess.run(['docker','exec',CONTAINER,PY,'-c','import requests; assert requests.get("http://127.0.0.1:18400/health",timeout=3).status_code==200'],capture_output=True,timeout=15)
                    if r.returncode==0:break
                    assert time.monotonic()<deadline,'readiness timeout';time.sleep(5)
                with (out/f'{cell}-client.log').open('x') as log, (out/f'{cell}-resources.jsonl').open('x') as resources:
                    client=subprocess.Popen(['docker','exec',CONTAINER,PY,'/workspace/round8/tools/qwen_dynamic_perf.py',
                        '--profile',profile,'--block',str(block),'--position',str(position)],stdout=log,stderr=subprocess.STDOUT)
                    while client.poll() is None:
                        row=snapshot()
                        row['host_process_cpu']=command('ps','-eo','pid,pcpu,comm','--sort=-pcpu')
                        resources.write(json.dumps(row)+'\n');resources.flush()
                        for line in row['containers'].splitlines():
                            stat=json.loads(line)
                            if stat['Name']!=CONTAINER:assert float(stat['CPUPerc'].strip('%'))<=10,'FOREIGN_CPU_CONTENTION'
                        device_processes=re.findall(r'^\|\s+([0-7])\s+\d+\s+\|\s+(\d+)\s+\|',row['npu'],re.MULTILINE)
                        assert all(device=='2' for device,pid in device_processes),'FOREIGN_NPU_CONTENTION'
                        assert 'Check input storage_offset' not in (out/f'{cell}-server.log').read_text()
                        time.sleep(10)
                    assert client.wait()==0,'performance client failed'
                state['completed'].append(cell);save();ensure_stopped();owns_process=False
        state['status']='COLLECTED_PENDING_RESOURCE_AUDIT'
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:
        save()
        if owns_process:ensure_stopped()

if __name__=='__main__':main()
