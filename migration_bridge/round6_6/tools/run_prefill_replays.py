"""Finish real-operand prefill checks after capture; no model qualification."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    out=Path('/workspace/results/r65-r66-prefill-sequence-v2');out.mkdir(exist_ok=False)
    progress=dict(status='WAIT_OPERATOR_CAPTURE',QD6='NOT_RUN',HD6='NOT_RUN')
    def save():(out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    try:
        save();deadline=time.monotonic()+1800
        pred=Path('/workspace/results/r65-r66-followups-guard-v1')
        while not (pred/'reaping.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('PREFILL_CAPTURE_TIMEOUT')
            time.sleep(5)
        if json.loads((pred/'result.json').read_text())['reason']!='EXIT_ZERO':raise ValueError('CAPTURE_FAILED')
        cap=Path('/workspace/results/r65-r66-operator-prefill-capture-v1')
        d=json.loads((cap/'capture-result.json').read_text())
        if d['status']!='CAPTURE_INTEGRITY_PASS' or d['forward_count']!=6:raise ValueError('CAPTURE_INTEGRITY')
        base='/workspace/results/r65-r66-prefill-output-v2'
        stage='/workspace/results/r65-r66-prefill-stages-v2'
        py=sys.executable;guard='/workspace/r65-tools/migration_bridge/round6_5/tools/owned_guard.py'
        common=['bash','/workspace/r66-tools/launch_python.sh','/workspace/r66-tools/three_way_probe_v4.py',
            '--trace',str(cap/'stagewise'),'--model','/workspace/models/DeepSeek-V2-Lite-604d5664',
            '--gguf','/workspace/results/r65-assets-v1/all-moe-f32.gguf',
            '--calls','1,3,5','--prefill-only','--experts','1,9,17,25,33,41,49,57']
        for name,result,extra,env in [
            ('production',base,[],dict(os.environ)),
            ('stage',stage,['--stage-capture','--baseline',base+'/three_way.json'],dict(os.environ,PYTHONPATH='/workspace/r66-debug-python'))]:
            progress['status']='RUNNING_'+name;save()
            with (out/(name+'.log')).open('x') as log:
                subprocess.run([py,guard,'--output',f'/workspace/results/r65-r66-prefill-{name}-guard-v2',
                    '--timeout-seconds','3600','--',*common,'--output',result,*extra],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            d=json.loads((Path(result)/'three_way.json').read_text())
            if len(d['rows'])!=96 or any(not r['finite'] or not r['local_safety_pass'] or not r['deterministic'] for r in d['rows']):
                raise ValueError('PREFILL_LOCAL_GATE')
        progress['status']='PREFILL_OUTPUT_AND_STAGE_CAPTURE_COMPLETE_AWAIT_AUDIT'
    except BaseException as exc:progress.update(status='BLOCKED',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
