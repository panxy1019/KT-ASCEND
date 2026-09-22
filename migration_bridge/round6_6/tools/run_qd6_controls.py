"""Owned post-QD exact same-path controls, no HD6 or quality outputs."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    out=Path('/workspace/results/r65-r66-qd6-controls-v1');out.mkdir(exist_ok=False)
    report=dict(status='WAIT_QD6',events=[],heldout_started=False)
    def save():(out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        save();pred=Path('/workspace/results/r65-r66-qd6-sequence-guard-v1')
        deadline=time.monotonic()+7200
        while not (pred/'reaping.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('QD6_TIMEOUT')
            time.sleep(10)
        assert json.loads((pred/'result.json').read_text())['reason']=='EXIT_ZERO'
        assert json.loads((pred/'reaping.json').read_text())['complete']
        for kind in ('repeat','restart'):
            command=[sys.executable,str(Path(__file__).with_name('run_study_generation.py')),
                '--inputs','/workspace/results/r65-r66-study-freeze-v1/qd6.json',
                '--profile','/workspace/results/r65-profiles-v1/four-R32.json',
                '--kind',kind,'--output',str(out/kind)]
            if kind=='restart':command+=['--reference',str(out/'repeat/generation.json')]
            report['status']='RUNNING_'+kind.upper();save()
            with (out/(kind+'.log')).open('x') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((out/kind/'generation.json').read_text())
            assert result['status']=='GENERATION_PASS'
            assert result['request_count']==(52 if kind=='repeat' else 4)
            report['events'].append(kind+'_PASS');save()
        report['status']='SAME_PATH_10_REPEAT_PREFIX_AND_RESTART_PASS'
    except BaseException as exc:
        report.update(status='BLOCKED',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
