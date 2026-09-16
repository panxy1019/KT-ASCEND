"""Bounded entry queue: oracle complete -> oracle logits -> single R32 gates.

Does not promote four/all layers before local controls and expert checks close.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path('/workspace/results'); tools=Path(__file__).parent
    inputs=tools.parent/'fixtures/numerical.json'
    state=dict(status='WAIT_R64_DETERMINISM',completed=[],started_unix=time.time())
    def save(): (a.output/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        deadline=time.monotonic()+3600
        reference=root/'r65-r64-det-v3/report.json'
        while not reference.exists():
            if time.monotonic()>deadline:raise TimeoutError('R64_PREDECESSOR')
            time.sleep(10)
        if json.loads(reference.read_text())['status']!='STATIC_COMPACT_DETERMINISM_PASS':raise ValueError('R64_GATE_FAILED')
        identity=json.loads((root/'r65-assets-v1/identity.json').read_text())
        if identity['status']!='ALL_EXPERT_WEIGHT_IDENTITY_PASS':raise ValueError('WEIGHT_IDENTITY_GATE')
        plan=[('oracle_capture',[tools/'capture_static_logits.py','--inputs',inputs,
            '--profile',root/'r65-profiles-v1/single-R64.json','--output',root/'r65-numeric-r64-v1']),
            ('single_R32_determinism',[tools/'run_static_determinism.py','--corpus',root/'corpora-v2/s6.json',
            '--profile',root/'r65-profiles-v1/single-R32.json','--output',root/'r65-single-R32-det-v1']),
            ('single_R32_numeric',[tools/'capture_static_logits.py','--inputs',inputs,
            '--profile',root/'r65-profiles-v1/single-R32.json','--force',root/'r65-numeric-r64-v1/force.json',
            '--output',root/'r65-numeric-single-R32-v1']),
            ('single_R32_evaluate',[tools/'evaluate_static_pairwise.py','--baseline',root/'r65-numeric-r64-v1',
            '--hybrid',root/'r65-numeric-single-R32-v1','--output',a.output/'single-R32-pairwise.json'])]
        for name,args in plan:
            state.update(status='RUNNING',active=name,command=list(map(str,args)));save()
            with (a.output/(name+'.log')).open('x') as log:
                result=subprocess.run([sys.executable,*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{name}:exit={result.returncode}')
            state['completed'].append(name);save()
        state['status']='SINGLE_R32_ENTRY_COMPLETE_LOCAL_CONTROLS_PENDING'
    except BaseException as exc:
        state.update(status='FAILED',exception=repr(exc));raise
    finally:
        state['ended_unix']=time.time();save()


if __name__=='__main__':main()
