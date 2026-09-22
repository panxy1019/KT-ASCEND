"""After failed qualification, reproduce and locate first layer; never promote."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    tools=Path(__file__).parent;results=Path('/workspace/results')
    state=dict(status='WAIT_FAILURE_REPLAY',scope='DEBUG_ONLY_NOT_QUALIFICATION',completed=[])
    def save():(a.output/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
    def run(name,cmd,env=None):
        state.update(status='RUNNING',active=name);save()
        with (a.output/(name+'.log')).open('x') as f:
            code=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,env=env).returncode
        if code not in (0,2) or (code==2 and name!='repeat-pairwise'):raise RuntimeError(f'{name}:{code}')
        state['completed'].append(name);save()
    save()
    try:
        deadline=time.monotonic()+2400
        while not (a.replay/'capture-result.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('FAILURE_REPLAY_TIMEOUT')
            time.sleep(10)
        replay=json.loads((a.replay/'capture-result.json').read_text())
        if replay['status']!='CAPTURE_INTEGRITY_PASS':raise ValueError('FAILURE_REPLAY_INTEGRITY')
        run('repeat-pairwise',[sys.executable,tools/'evaluate_static_pairwise.py',
            '--baseline',results/'r65-numeric-r64-v1','--hybrid',a.replay,
            '--output',a.output/'repeat-pairwise.json'])
        original=json.loads((results/'r65-scale-matrix-v2/four-R32/pairwise.json').read_text())
        rid=original['first_failure']['rid']
        inputs=json.loads((tools.parent/'fixtures/numerical.json').read_text())
        inputs['requests']=[r for r in inputs['requests'] if r['rid']==rid]
        force=json.loads((results/'r65-numeric-r64-v1/force.json').read_text())
        force['requests']=[r for r in force['requests'] if r['rid']==rid]
        (a.output/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
        (a.output/'force.json').write_text(json.dumps(force,indent=2)+'\n')
        env=dict(os.environ,KT_R65_STAGEWISE='1')
        for name,profile in [('npu','all-R64'),('hybrid','four-R32')]:
            run(name,[sys.executable,tools/'capture_static_logits.py','--inputs',a.output/'inputs.json',
                '--force',a.output/'force.json','--profile',results/f'r65-profiles-v1/{profile}.json',
                '--output',a.output/name],env=env)
        state['status']='LAYER_TRACES_COMPLETE_REVIEW_REQUIRED'
    except BaseException as exc:state.update(status='FAILED',exception=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
