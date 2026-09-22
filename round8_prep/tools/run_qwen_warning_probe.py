"""Fresh DEV-only warning localization, never held-out or timed evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

FIXED='--fixed' in sys.argv
NAME='q32-dynamic-transfer-v2' if FIXED else 'q32-dynamic-warning-v1'
CONTAINER='kt-round8-qwen35-dynamic-b' if FIXED else 'kt-round8-qwen35-dynamic-a'
PY='/workspace/round8/venv/bin/python'

if '--serve' in sys.argv:
    root=Path('/workspace/round8')
    cfg=json.loads((root/NAME/'resolved-command.json').read_text())
    for path,digest in cfg['code_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    env=os.environ.copy();env.update(cfg['env'])
    os.execvpe(cfg['argv'][0],cfg['argv'],env)

root=Path('/home/admin/kt-round8-qwen35');out=root/NAME
out.mkdir(exist_ok=False)
cfg=json.loads((root/f'q32-dynamic-{512 if FIXED else 256}-v1/resolved-command.json').read_text())
if FIXED:
    old='/workspace/round8/sglang-python';new=old+'-dynamic-v2'
    cfg['argv']=[s.replace(old,new) for s in cfg['argv']]
    cfg['env']={k:v.replace(old,new) for k,v in cfg['env'].items()}
    cfg['code_hashes']={k.replace(old,new):v for k,v in cfg['code_hashes'].items()}
    changed=f'{new}/sglang/srt/layers/moe/kt_ep_wrapper.py'
    cfg['code_hashes'][changed]=hashlib.sha256((root/'sglang-python-dynamic-v2/sglang/srt/layers/moe/kt_ep_wrapper.py').read_bytes()).hexdigest()
    cfg['argv'][cfg['argv'].index('--kt-gpu-prefill-token-threshold')+1]='256'
    cfg['dynamic_threshold']=256
idx=cfg['argv'].index('--forward-hooks')+1
hooks=json.loads(cfg['argv'][idx]);hooks[0]['hook_factory']='qwen_dynamic_warning:make_structure'
hooks[0]['config']['output']=f'/workspace/round8/{NAME}'
cfg['argv'][idx]=json.dumps(hooks)
cfg['code_hashes']['/workspace/round8/tools/qwen_dynamic_warning.py']=hashlib.sha256((root/'tools/qwen_dynamic_warning.py').read_bytes()).hexdigest()
(out/'resolved-command.json').write_text(json.dumps(cfg,indent=2))
state=dict(status='RUNNING',purpose='DEV_WARNING_LOCALIZATION',heldout_used=False)
(out/'state.json').write_text(json.dumps(state))
try:
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}
    assert mem['MemAvailable']>256*2**30
    inspect=json.loads(subprocess.check_output(['docker','inspect',CONTAINER]))[0]
    assert not inspect['State']['Running'],'do not interrupt healthy lane'
    assert inspect['HostConfig']['CpusetCpus']==('64-127' if FIXED else '0-63')
    subprocess.run(['docker','start',CONTAINER],check=True)
    subprocess.run(['docker','exec','-d',CONTAINER,'bash','-c',
        f'{PY} /workspace/round8/tools/run_qwen_warning_probe.py --serve {"--fixed" if FIXED else ""} > /workspace/round8/{NAME}/server.log 2>&1'],check=True)
    deadline=time.monotonic()+1200
    while True:
        check=subprocess.run(['docker','exec',CONTAINER,PY,'-c',
            'import requests; assert requests.get("http://127.0.0.1:18400/health",timeout=3).status_code==200'],capture_output=True,timeout=15)
        if check.returncode==0:break
        assert time.monotonic()<deadline,'readiness timeout'
        time.sleep(5)
    with (out/'client.log').open('x') as log:
        subprocess.run(['docker','exec',CONTAINER,PY,'/workspace/round8/tools/qwen_dynamic_dev_client.py',
            '--threshold','256','--dev','/workspace/round8/qwen-scheduler-v1/SCHED_DEV.jsonl',
            '--audit-root',f'/workspace/round8/{NAME}'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
    state['status']='COLLECTED_PENDING_WARNING_AUDIT'
except BaseException as error:
    state.update(status='FAILED',error=repr(error));raise
finally:
    (out/'state.json').write_text(json.dumps(state,indent=2))
    subprocess.run(['docker','stop','-t','60',CONTAINER],capture_output=True,timeout=150)
