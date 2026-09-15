"""Two frozen failed histories, production then private oracle; NOT qualification."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main():
    root = Path('/workspace/results')
    here = Path(__file__).resolve().parent
    old = root/'round6_2-prequalification/tools'
    output = root/'r63-rejected-replay-v1'
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1',31000)) != 0, 'EXISTING_SERVER'
    output.mkdir(exist_ok=False)
    source = root/'round6_3-formal-inputs/qd3-force.json'
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '35d45fdd9fa8cc453f1b61ee464e3f2261710ea0ebecf2ce7fbca7ed4e79beae'
    requests = json.loads(raw)
    requests['requests'] = [r for r in requests['requests'] if r['rid'] in ('qd3-math-11','qd3-math-19')]
    assert len(requests['requests']) == 2
    requests['session_id'] = 'r63-rejected-replay-diagnostic'
    manifest = output/'force.json'
    manifest.write_text(json.dumps(requests,indent=2)+'\n')
    result = dict(status='STARTED',qualification=False,started=time.time(),
        parent_history_sha256=hashlib.sha256(raw).hexdigest(),
        history_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        tool_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),here/'probe_explicit_diagnostic.py',here/'run_stateful_probe.py',here/'launch_stateful_smoke.sh')})
    def cancel(signum, frame):
        raise InterruptedError('OWNED_DIAGNOSTIC_CANCELLED')
    signal.signal(signal.SIGTERM,cancel)
    try:
        for mode in ('production','explicit'):
            server_root = root/f'r63-rejected-replay-v1-{mode}-server'
            session = output/mode
            session.mkdir()
            env = dict(os.environ,KT_R63_COMPACT_CAPTURE='0',KT_R63_OBSERVER_PROFILE='full',
                       PYTHONPATH=str(here)+':'+str(old))
            env.pop('KT_R61_EXPLICIT_REPLAY',None)
            if mode == 'explicit':
                env['KT_R61_EXPLICIT_REPLAY']='1'
            with (session/'process.log').open('x') as log:
                def call(script,*args):
                    subprocess.run([sys.executable,str(script),*map(str,args)],env=env,
                        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
                server = subprocess.Popen(['bash',str(here/'launch_stateful_smoke.sh'),'D1',str(server_root),str(manifest)],env=env,stdout=log,stderr=subprocess.STDOUT)
                try:
                    call(old/'wait_capture_ready.py','--log',server_root/'guard/process.log','--captures',server_root/'tensors','--output',server_root/'readiness.json')
                    call(here/'prepare_stateful_smoke.py','--arm','--baseline-result',manifest,'--output',server_root/'armed.sha256')
                    probe = 'run_stateful_probe.py' if mode=='production' else 'probe_explicit_diagnostic.py'
                    call(here/probe,'--requests',manifest,'--index',server_root/'index.jsonl','--trace',server_root/'forcing.jsonl','--output',session/'capture')
                finally:
                    if server.poll() is None:
                        server.send_signal(signal.SIGTERM)
                        server.wait(timeout=60)
                call(here/'audit_stateful_index.py','--server-root',server_root,'--result',session/'capture/result.json','--output',session/'final-integrity.json')
        result['status']='DIAGNOSTIC_CAPTURE_COMPLETE_PENDING_COMPARISON'
    except BaseException as exc:
        result.update(status='FAILED',exception=repr(exc))
        raise
    finally:
        result['ended']=time.time()
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
