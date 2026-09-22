"""Honor the frozen conditional QD review; stop after one-shot HD audits."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    base=Path('/workspace/results');tools=Path(__file__).parent
    out=base/'r65-r66-heldout-sequence-v1';out.mkdir(exist_ok=False)
    report=dict(status='WAIT_EXACT_CONTROLS',events=[],quality_started=False)
    def save():(out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(name,script,args):
        report['status']='RUNNING_'+name.upper();save()
        with (out/(name+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(tools/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
        report['events'].append(name+'_COMPLETE');save()
    try:
        save();deadline=time.monotonic()+10800
        for name in ('r65-r66-qd6-controls-guard-v1','r65-r66-qd6-audit-guard-v1'):
            guard=base/name
            while not (guard/'reaping.json').exists():
                if time.monotonic()>deadline:raise TimeoutError('PREDECESSOR_TIMEOUT')
                time.sleep(10)
            assert json.loads((guard/'result.json').read_text())['reason']=='EXIT_ZERO'
            assert json.loads((guard/'reaping.json').read_text())['complete']
        # Validate each observer-OFF request against its frozen logical prompt;
        # padded backend log counters are not substituted for logical lengths.
        inputs=json.loads((base/'r65-r66-study-freeze-v1/qd6.json').read_text())
        prompts={r['rid']:r['input_ids'] for r in inputs['requests']}
        control=base/'r65-r66-qd6-controls-v1'
        for kind,count in [('repeat',52),('restart',4)]:
            raw=[json.loads(s) for s in (control/kind/'requests.jsonl').read_text().splitlines()]
            assert len(raw)==count
            for r in raw:
                payload=r['payload'];meta=r['response']['meta_info']
                original=payload['rid'].rsplit('-'+kind+'-',1)[0]
                assert payload['input_ids']==prompts[original]
                assert meta['prompt_tokens']==len(prompts[original])
        review=base/'r65-r66-qd6-candidate-review-v1.json'
        assert hashlib.sha256(review.read_bytes()).hexdigest()=='9b1b9d011552d70d9e7203061c8cf3b29eb4f2eb38aab40bd215850c1182af66'
        candidate=base/'r65-r66-candidate-v1.json'
        run('freeze_candidate','freeze_candidate.py',['--review',str(review),'--output',str(candidate)])
        digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
        report['candidate_sha256']=digest;save()
        run('hd6_capture','run_hd6_capture.py',['--candidate',str(candidate),'--candidate-sha256',digest])
        hroot=base/'r65-r66-hd6-v1'
        for name,script,args in [
            ('transfers','audit_study_transfers.py',['--root',str(hroot/'hybrid'),'--profile',str(base/'r65-profiles-v1/four-R32.json')]),
            ('logits','summarize_study_logits.py',['--root',str(hroot)]),
            ('routers','summarize_study_routers.py',['--root',str(hroot)]),
        ]:
            run('hd6_'+name,script,[*args,'--output',str(out/('hd6_'+name+'.json'))])
        report['status']='HD6_ONE_SHOT_AUDITS_COMPLETE_AWAIT_REVIEW'
    except BaseException as exc:
        report.update(status='BLOCKED',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
