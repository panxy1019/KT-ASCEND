"""Serial Round7A orchestration after explicit local candidate commit freeze.

No polling or candidate fitting occurs here. The caller must first complete
DEV attribution, freeze the official Frequency mask and commit those artifacts.
Failures preserve results and return to the agent for reproduction and repair.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def validate_ack(ack,profile_sha,activation_sha):
    if not re.fullmatch(r'[0-9a-f]{40}',ack.get('local_candidate_commit','')):
        raise ValueError('MISSING_LOCAL_COMMIT_FREEZE')
    if ack.get('frequency_profile_sha256')!=profile_sha:
        raise ValueError('CANDIDATE_MASK_DRIFT')
    if ack.get('frequency_activation_sha256')!=activation_sha:
        raise ValueError('CANDIDATE_ACTIVATION_DRIFT')


def main():
    from static_qualification import sha,write
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--holdout',type=Path,required=True)
    p.add_argument('--freeze-ack',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    lease=(a.base/'static-phase.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((a.base/'dev-stat-summary.json').read_text())['status']=='DEV_STAT_COMPLETE'
    lifecycle=json.loads((a.base/'uniform-dev-v1/lifecycle.json').read_text())
    assert lifecycle['status']=='COMPLETE' and lifecycle['reaping']['complete'],'DEV_NOT_REAPED'
    ack=json.loads(a.freeze_ack.read_text())
    validate_ack(ack,sha(a.base/'profiles/frequency.json'),sha(a.base/'profiles/frequency.pt'))
    tools=Path(__file__).parent
    report=dict(status='RUNNING',started_unix=time.time(),stages=[],freeze_ack=ack,
        tools={x.name:sha(x) for x in sorted(tools.iterdir()) if x.is_file()})
    write(a.output/'progress.json',report)
    def stage(name,args):
        report.update(phase=name);write(a.output/'progress.json',report)
        with (a.output/f'{name}.log').open('x') as log:
            subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,check=True)
        report['stages'].append(name);write(a.output/'progress.json',report)
    def static(name,step):
        stage(f'{name}-{step}',[str(tools/'static_qualification.py'),'--base',str(a.base),
            '--holdout',str(a.holdout),'--profile',name,'--stage',step])
    try:
        stage('operator-representatives',[str(tools/'operator_representatives.py'),str(a.base)])
        for name in ('uniform','frequency'):static(name,'compact')
        for name in ('uniform','frequency'):static(name,'holdout')
        static('frequency','quality')
        quality=json.loads((a.base/'frequency-quality-summary.json').read_text())
        if quality['status']=='PASS':
            stage('uniform-vs-frequency-performance',[str(tools/'performance_screen.py'),
                '--base',str(a.base),'--corpus',str(a.holdout),'--left','uniform','--right','frequency',
                '--output',str(a.output/'uniform-vs-frequency-performance')])
        else:
            report['performance']='NOT_RUN_FREQUENCY_QUALITY_REJECTION'
        report.update(status='STATIC_EVIDENCE_COMPLETE_AWAITING_AGENT_SELECTION',
            frequency_quality=quality['status'],ended_unix=time.time())
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),ended_unix=time.time());raise
    finally:write(a.output/'progress.json',report);write(a.output/'summary.json',report)


if __name__=='__main__':main()
