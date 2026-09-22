"""Serial bounded route-control gate. No automatic scale promotion."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--determinism',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-controls',type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    tools=Path(__file__).parent;state=dict(status='STARTING',completed=[])
    def save():(a.output/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        if a.wait_controls:
            deadline=time.monotonic()+7200
            while True:
                data=json.loads(a.wait_controls.read_text())
                if data['status']=='FAILED':raise ValueError('LOCAL_CONTROL_PREDECESSOR_FAILED')
                if data['status']=='LOCAL_REAL_WEIGHT_CONTROLS_PASS':break
                if time.monotonic()>deadline:raise TimeoutError('LOCAL_CONTROL_PREDECESSOR_TIMEOUT')
                time.sleep(5)
        commands=[('capture',[tools/'run_static_route_capture.py','--profile',a.profile,
            '--determinism',a.determinism,'--output',a.output/'capture']),
            ('analyze',[tools/'analyze_static_routes.py','--profile',a.profile,
            '--capture',a.output/'capture','--output',a.output/'routes.json']),
            ('lifecycle',[tools/'audit_static_lifecycle.py','--root',a.output/'capture',
            '--output',a.output/'lifecycle-audit.json'])]
        for name,command in commands:
            state.update(status='RUNNING',active=name);save()
            with (a.output/(name+'.log')).open('x') as f:
                r=subprocess.run([sys.executable,*map(str,command)],stdout=f,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError(f'{name}:{r.returncode}')
            state['completed'].append(name);save()
        state['status']='STATIC_ROUTE_CONTROL_PASS'
    except BaseException as exc:state.update(status='FAILED',exception=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
