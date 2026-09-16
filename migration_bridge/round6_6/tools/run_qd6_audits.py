"""Wait for owned QD capture completion, audit read-only; never starts HD/quality."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('/workspace/results/r65-r66-qd6-v1')
    out=Path('/workspace/results/r65-r66-qd6-audits-v1');out.mkdir(exist_ok=False)
    report=dict(status='WAIT_QD6',events=[],hd6_started=False,quality_started=False)
    def save():(out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        save();deadline=time.monotonic()+7200
        guard=Path('/workspace/results/r65-r66-qd6-sequence-guard-v1')
        while not (guard/'reaping.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('QD6_TIMEOUT')
            time.sleep(10)
        assert json.loads((guard/'result.json').read_text())['reason']=='EXIT_ZERO'
        assert json.loads((guard/'reaping.json').read_text())['complete']
        assert json.loads((root/'progress.json').read_text())['status']=='QD6_CAPTURED_AWAIT_SYSTEM_AND_NUMERICAL_AUDIT'
        tools=Path(__file__).parent
        for name,script,args in [
            ('transfers','audit_study_transfers.py',['--root',str(root/'hybrid'),'--profile','/workspace/results/r65-profiles-v1/four-R32.json']),
            ('logits','summarize_study_logits.py',['--root',str(root)]),
            ('routers','summarize_study_routers.py',['--root',str(root)]),
        ]:
            report['status']='RUNNING_'+name.upper();save()
            command=[sys.executable,str(tools/script),*args,'--output',str(out/(name+'.json'))]
            with (out/(name+'.log')).open('x') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            report['events'].append(name+'_COMPLETE');save()
        report['status']='QD6_AUDITS_COMPLETE_AWAIT_CANDIDATE_REVIEW'
    except BaseException as exc:
        report.update(status='BLOCKED',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
