"""Reuse the proven owned guard/admission; change only Round7 launch config."""
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.append('/workspace/c64-dev/tools-v16/migration_bridge/round6_7_c64/tools')
from serial_session import memory, check_port_available
from cgroup_memory import admission


@contextmanager
def session(root, config):
    root = Path(root); root.mkdir(parents=True,exist_ok=False)
    from worker_topology import topology, output_root
    output_root(root)
    cpu_set, _ = topology(json.loads(Path(config).read_text()))
    check_port_available()
    report = dict(status='STARTING',started_unix=time.time(),admission=admission(),memory_before=memory())
    tools = Path(__file__).parent
    cmd = [sys.executable,str(tools/'owned_guard.py'),
           '--output',str(root/'guard'),'--timeout-seconds','86400','--',
           'taskset','-c',cpu_set,'bash',str(tools/'launch_round7.sh'),str(root),str(config)]
    child = None
    with (root/'launcher.log').open('x') as log:
        try:
            child = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=root)
            deadline = time.monotonic()+1200
            while True:
                if child.poll() is not None: raise RuntimeError('SERVER_START_FAILED')
                if time.monotonic()>deadline: raise TimeoutError('SERVER_READY_TIMEOUT')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:31000/health',timeout=2) as r:
                        if r.status==200: break
                except OSError: pass
                time.sleep(2)
            report.update(status='READY',memory_after_load=memory())
            (root/'ready.json').write_text(json.dumps(report,indent=2))
            yield root
            report.update(status='COMPLETE',memory_after_work=memory())
        except BaseException as e:
            report.update(status='FAIL',error=repr(e)); raise
        finally:
            if child is not None:
                if child.poll() is None: child.send_signal(signal.SIGTERM)
                child.wait(timeout=60)
                reaping=json.loads((root/'guard/reaping.json').read_text())
                report['reaping']=reaping
                if not reaping['complete']: raise RuntimeError('OWNED_REAP_INCOMPLETE')
            report.update(ended_unix=time.time(),memory_after_stop=memory())
            (root/'lifecycle.json').write_text(json.dumps(report,indent=2))
