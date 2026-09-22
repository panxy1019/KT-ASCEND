"""Bounded operator-only sequence. Never starts QD, HD, or quality."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    out=Path('/workspace/results/r65-r66-operator-followups-v1');out.mkdir(exist_ok=False)
    progress=dict(status='WAIT_STAGE_SCREEN',model_qualification_allowed=False,events=[])
    def save():(out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    def run(name,args,env=None):
        progress['status']='RUNNING_'+name;save()
        with (out/(name+'.log')).open('x') as log:
            subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        progress['events'].append(name+'_COMPLETE');save()
    py=sys.executable;guard='/workspace/r65-tools/migration_bridge/round6_5/tools/owned_guard.py'
    launcher='/workspace/r66-tools/launch_python.sh'
    try:
        save();deadline=time.monotonic()+1800
        pred=Path('/workspace/results/r65-r66-stages-guard-v2')
        while not (pred/'reaping.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('PREDECESSOR_TIMEOUT')
            time.sleep(5)
        if json.loads((pred/'result.json').read_text())['reason']!='EXIT_ZERO':raise ValueError('STAGE_SCREEN_FAILURE')
        if not json.loads((pred/'reaping.json').read_text())['complete']:raise ValueError('PREDECESSOR_REAPING')
        d=json.loads(Path('/workspace/results/r65-r66-stages-v2/three_way.json').read_text())
        if len(d['rows'])!=512:raise ValueError('STAGE_COVERAGE')
        for row in d['rows']:
            if not row['production_observer_parity'] or not row['local_safety_pass']:raise ValueError('STAGE_PARITY')
            for stage in row['stage_comparisons'].values():
                if not all(m['finite'] and m['relative_l2']<=.01 for m in stage.values()):
                    raise ValueError('PROJECTION_ANOMALY_REVIEW')
        run('layout',[py,guard,'--output','/workspace/results/r65-r66-layout-guard-v1',
            '--timeout-seconds','900','--','bash',launcher,'/workspace/r66-tools/probe_oracle_layout.py',
            '--baseline','/workspace/results/r65-r66-threeway-v1',
            '--model','/workspace/models/DeepSeek-V2-Lite-604d5664',
            '--output','/workspace/results/r65-r66-layout-v1'])
        layout=json.loads(Path('/workspace/results/r65-r66-layout-v1/layout.json').read_text())
        if layout['status']!='NPU_ORACLE_ONE_VS_64_LAYOUT_EXACT':raise ValueError('LAYOUT_REVIEW')
        run('real_prefill_capture',[py,'/workspace/r65-tools/migration_bridge/round6_5/tools/capture_static_logits.py',
            '--inputs','/workspace/results/r65-r66-operator-prefill-inputs-v1.json',
            '--profile','/workspace/results/r65-profiles-v1/four-R32.json',
            '--output','/workspace/results/r65-r66-operator-prefill-capture-v1'],
            env=dict(os.environ,KT_R65_STAGEWISE='1'))
        progress['status']='REAL_PREFILL_OPERANDS_CAPTURED_AWAIT_OPERATOR_REPLAY'
    except BaseException as exc:progress.update(status='BLOCKED',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
