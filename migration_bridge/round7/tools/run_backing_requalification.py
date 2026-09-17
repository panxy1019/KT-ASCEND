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
    p.add_argument('--resume-from',type=Path,help='Preserved failed R64 adapter run with closed earlier gates')
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
        if a.resume_from:
            from static_qualification import sha
            prior=json.loads((a.resume_from/'progress.json').read_text())
            assert prior['status']=='FAIL' and prior['error']=="AssertionError('FRESH_R64_REFERENCE_CHANGED')"
            assert prior['phase']=='fresh-r64-quality' and prior['stages']==['operators','uniform-compact','frequency-compact']
            paths=[a.base/'operator-representatives/summary.json',a.base/'uniform-system-summary.json',a.base/'frequency-system-summary.json']
            assert all(json.loads(path.read_text())['status']=='PASS' for path in paths),'RESUME_PREDECESSORS_NOT_CLOSED'
            report.update(stages=list(prior['stages']),resumed_from=str(a.resume_from),
                retained_gate_sha256={str(path):sha(path) for path in paths},
                prior_failure_sha256=sha(a.resume_from/'progress.json'))
            save()
        else:
            stage('operators',[str(tools/'operator_representatives.py'),str(a.base),'--runtime-env',str(a.runtime_env)])
            for name in ('uniform','frequency'):
                stage(f'{name}-compact',[str(tools/'static_qualification.py'),'--base',str(a.base),
                    '--holdout',str(a.holdout),'--profile',name,'--stage','compact'])
        report['phase']='fresh-r64-quality';save()
        cfg=json.loads((a.base/'fresh-r64-config.json').read_text())
        cfgpath=a.output/'fresh-r64-direct-config.json';write(cfgpath,cfg)
        q=quality(a.base,'fresh-r64',configuration=cfgpath,
            target=a.output/'fresh-r64-direct-process',summary_path=a.output/'fresh-r64-quality-summary.json')
        fresh=json.loads((a.output/'fresh-r64-direct-process/resolved-command.json').read_text())['argv']
        historical=json.loads(Path('/workspace/results/r65-r67-serial-v5/R64/quality/resolved-command.json').read_text())['argv']
        assert fresh==historical,'FRESH_R64_LAUNCH_CONTRACT_DIFFERS'
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
