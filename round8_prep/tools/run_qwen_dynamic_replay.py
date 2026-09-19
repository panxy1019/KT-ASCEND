"""Wait for audited DEV collection, then fresh hook-free replay on owned lane A."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

NAME='q32-dynamic-restart-v2';CONTAINER='kt-round8-qwen35-dynamic-a'
PY='/workspace/round8/venv/bin/python'
if '--serve' in sys.argv:
    root=Path('/workspace/round8');cfg=json.loads((root/NAME/'resolved-command.json').read_text())
    for path,digest in cfg['code_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    assert '--forward-hooks' not in cfg['argv']
    env=os.environ.copy();env.update(cfg['env']);os.execvpe(cfg['argv'][0],cfg['argv'],env)

root=Path('/home/admin/kt-round8-qwen35');out=root/NAME;out.mkdir(exist_ok=False)
state={'status':'WAITING_FOR_DEV_TRANSFER_V2'}
def save():(out/'state.json').write_text(json.dumps(state,indent=2))
save();started=False
try:
    deadline=time.monotonic()+7200
    while True:
        path=root/'q32-dynamic-transfer-v2/state.json'
        if path.exists():
            prior=json.loads(path.read_text())
            assert prior['status']!='FAILED',prior
            if prior['status']=='COLLECTED_PENDING_WARNING_AUDIT':break
        assert time.monotonic()<deadline,'predecessor timeout'
        time.sleep(10)
    predecessor=root/'q32-dynamic-transfer-v2'
    assert json.loads((predecessor/'dynamic-dev-summary.json').read_text())['status']=='PASS'
    assert 'Check input storage_offset' not in (predecessor/'server.log').read_text()
    cfg=json.loads((predecessor/'resolved-command.json').read_text())
    idx=cfg['argv'].index('--forward-hooks');del cfg['argv'][idx:idx+2]
    idx=cfg['argv'].index('--kt-numa-nodes');cfg['argv'][idx+1:idx+3]=['0','1']
    cfg['numa_nodes']=[0,1];cfg['purpose']='DEV_CLEAN_RESTART_OBSERVER_PARITY'
    (out/'resolved-command.json').write_text(json.dumps(cfg,indent=2))
    while json.loads(subprocess.check_output(['docker','inspect',CONTAINER]))[0]['State']['Running']:
        assert time.monotonic()<deadline,'owned lane busy; never stop healthy predecessor'
        time.sleep(10)
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
    assert mem['MemAvailable']>256*2**30
    subprocess.run(['docker','start',CONTAINER],check=True);started=True
    state['status']='RUNNING';save()
    subprocess.run(['docker','exec','-d',CONTAINER,'bash','-c',
        f'{PY} /workspace/round8/tools/run_qwen_dynamic_replay.py --serve > /workspace/round8/{NAME}/server.log 2>&1'],check=True)
    deadline=time.monotonic()+1200
    while True:
        r=subprocess.run(['docker','exec',CONTAINER,PY,'-c',
            'import requests; assert requests.get("http://127.0.0.1:18400/health",timeout=3).status_code==200'],capture_output=True,timeout=15)
        if r.returncode==0:break
        assert time.monotonic()<deadline,'readiness timeout'
        time.sleep(5)
    with (out/'client.log').open('x') as log:
        subprocess.run(['docker','exec',CONTAINER,PY,'/workspace/round8/tools/qwen_dynamic_replay.py'],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
    assert 'Check input storage_offset' not in (out/'server.log').read_text()
    state['status']='REPLAY_PASS_PENDING_INDEPENDENT_AUDIT'
except BaseException as error:
    state.update(status='FAILED',error=repr(error));raise
finally:
    save()
    if started:subprocess.run(['docker','stop','-t','60',CONTAINER],capture_output=True,timeout=150)
