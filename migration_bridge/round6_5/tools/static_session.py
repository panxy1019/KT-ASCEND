"""Owned single model lifecycle for isolated capture and resource diagnostics."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


@contextmanager
def session(root,profile,*,capture=False,force=None,workers=32):
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(('127.0.0.1',31000))
    env=dict(os.environ)
    env.pop('KT_R65_FORCE',None);env['KT_R65_CAPTURE']=str(int(capture))
    if force: env['KT_R65_FORCE']=str(force)
    tools=Path(__file__).parent
    cmd=[sys.executable,str(tools/'owned_guard.py'),'--output',str(root/'guard'),
         '--timeout-seconds','10800','--','taskset','-c',f'0-{workers-1}','bash',
         str(tools/'launch_static.sh'),str(root),str(profile),str(workers)]
    result=dict(started_unix=time.time(),command=cmd,status='STARTING')
    child=None
    with (root/'launcher.log').open('x') as log:
        try:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            deadline=time.monotonic()+1200
            while True:
                if child.poll() is not None:raise RuntimeError('MODEL_START_FAILED')
                if time.monotonic()>deadline:raise TimeoutError('READY_TIMEOUT')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:31000/health',timeout=2) as r:
                        if r.status==200:break
                except OSError:pass
                time.sleep(2)
            if capture:
                if not (root/'index.jsonl').exists() or (root/'index.jsonl').read_text().strip():
                    raise ValueError('STARTUP_CAPTURE_CONTAMINATION')
                if list((root/'tensors').rglob('*.pt')):raise ValueError('STARTUP_TENSORS')
            if force:
                (root/'armed.sha256').write_text(hashlib.sha256(Path(force).read_bytes()).hexdigest()+'\n')
            result['status']='READY'
            yield root
            result['status']='COMPLETE'
        except BaseException as exc:
            result.update(status='FAILED',exception=repr(exc));raise
        finally:
            if child is not None:
                if child.poll() is None:child.send_signal(signal.SIGTERM)
                child.wait(timeout=60)
                reap=root/'guard/reaping.json'
                result['reaping']=json.loads(reap.read_text()) if reap.exists() else {'complete':False}
                if not result['reaping']['complete']:result['status']='REAPING_FAILED'
            result['ended_unix']=time.time()
            (root/'lifecycle.json').write_text(json.dumps(result,indent=2)+'\n')
            if result['status']=='REAPING_FAILED':raise RuntimeError('OWNED_PROCESS_REAPING_FAILED')
