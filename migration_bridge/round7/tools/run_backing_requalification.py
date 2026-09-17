"""Serial post-fix predecessors, then DEV-only migration probes; no held-out fit."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from static_qualification import quality,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--dev',type=Path,required=True);p.add_argument('--holdout',type=Path,required=True)
    p.add_argument('--runtime-env',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    lease=(a.base/'backing-requalification.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tools=Path(__file__).parent
    report=dict(status='RUNNING',started_unix=time.time(),stages=[])
    def save():write(a.output/'progress.json',report)
    def stage(name,args):
        report['phase']=name;save()
        with (a.output/f'{name}.log').open('x') as log:
            subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,check=True)
        report['stages'].append(name);save()
    try:
        stage('operators',[str(tools/'operator_representatives.py'),str(a.base),'--runtime-env',str(a.runtime_env)])
        for name in ('uniform','frequency'):
            stage(f'{name}-compact',[str(tools/'static_qualification.py'),'--base',str(a.base),
                '--holdout',str(a.holdout),'--profile',name,'--stage','compact'])
        report['phase']='fresh-r64-quality';save()
        q=quality(a.base,'fresh-r64',configuration=a.base/'fresh-r64-config.json',
            target=a.base/'fresh-r64-cmmlu-v1',summary_path=a.base/'fresh-r64-quality-summary.json')
        # Score reproduction never silently changes frozen174/product criterion.
        assert q['correct']==174 and q['invalid']==0,'FRESH_R64_REFERENCE_CHANGED'
        report['fresh_r64']=dict(correct=q['correct'],count=q['count'],invalid=q['invalid'])
        report['stages'].append('fresh-r64-quality');save()
        for threshold in (256,512,1024):
            stage(f'dynamic-dev-probe-{threshold}',[str(tools/'dynamic_probe.py'),'--base',str(a.base),
                '--dev',str(a.dev),'--profile','uniform','--threshold',str(threshold),
                '--output',str(a.output/f'dynamic-dev-probe-{threshold}')])
            r=json.loads((a.output/f'dynamic-dev-probe-{threshold}/summary.json').read_text())
            assert r['status']=='PASS','DYNAMIC_MIGRATION_PROBE_NOT_QUALIFIED'
        report.update(status='PREDECESSORS_COMPLETE_AWAITING_DEV_THRESHOLD_SCREEN',ended_unix=time.time())
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),ended_unix=time.time());raise
    finally:save()


if __name__=='__main__':main()
