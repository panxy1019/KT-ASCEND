"""Reviewed HD -> observer-OFF free generation -> paired frozen quality; bounded."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--approval',type=Path,required=True)
    p.add_argument('--approval-sha256',required=True)
    a=p.parse_args();base=Path('/workspace/results');tools=Path(__file__).parent
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(a.approval)==a.approval_sha256
    approval=json.loads(a.approval.read_text())
    assert approval['heldout_system_review']=='PASS' and approval['quality_execution_allowed'] is True
    assert approval['candidate_sha256']==sha(base/'r65-r66-candidate-v1.json')
    hd=base/'r65-r66-heldout-sequence-v1'
    assert json.loads((hd/'progress.json').read_text())['status']=='HD6_ONE_SHOT_AUDITS_COMPLETE_AWAIT_REVIEW'
    assert sha(hd/'hd6_logits.json')==approval['heldout_logits_sha256']
    assert sha(hd/'hd6_routers.json')==approval['heldout_routers_sha256']
    guard=base/'r65-r66-heldout-sequence-guard-v1'
    assert json.loads((guard/'result.json').read_text())['reason']=='EXIT_ZERO'
    assert json.loads((guard/'reaping.json').read_text())['complete']
    out=base/'r65-r66-final-screen-v1';out.mkdir(exist_ok=False)
    report=dict(status='RUNNING',events=[],approval_sha256=a.approval_sha256,
                final_qualification=False,population_noninferiority_claim=False)
    def save():(out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(name,script,args):
        report['status']='RUNNING_'+name.upper();save()
        with (out/(name+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(tools/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
        report['events'].append(name+'_COMPLETE');save()
    try:
        save()
        run('free_hybrid','run_study_generation.py',[
            '--inputs',str(base/'r65-r66-study-freeze-v1/hd6.json'),'--profile',str(base/'r65-profiles-v1/four-R32.json'),
            '--kind','free','--output',str(out/'free_hybrid')])
        result=json.loads((out/'free_hybrid/generation.json').read_text())
        assert result['status']=='GENERATION_PASS' and result['request_count']==12
        run('free_analysis','analyze_free_generation.py',[
            '--capture',str(base/'r65-r66-hd6-v1'),'--generation',str(out/'free_hybrid/generation.json'),
            '--diagnostics',str(hd/'hd6_logits.json'),'--output',str(out/'free_analysis.json')])
        for mode,profile in [('npu','four-R64'),('hybrid','four-R32')]:
            run('quality_'+mode,'run_quality_screen.py',[
                '--approval',str(a.approval),'--inputs',str(base/'r65-r66-study-freeze-v1/quality.json'),
                '--profile',str(base/'r65-profiles-v1'/ (profile+'.json')),
                '--parser-dir',str(tools),'--output',str(out/('quality_'+mode))])
        run('paired_quality','compare_quality_screen.py',[
            '--npu',str(out/'quality_npu/quality-report.json'),
            '--hybrid',str(out/'quality_hybrid/quality-report.json'),
            '--statistics-dir',str(tools),'--output',str(out/'paired_quality.json')])
        decision=json.loads((out/'paired_quality.json').read_text())
        if decision['status']!='PAIRED_SCREEN_PASS_AWAIT_ADR':raise ValueError('QUALITY_SCREEN_FAILURE')
        report['status']='FINAL_SCREEN_COMPLETE_AWAIT_QUALITY_LIMITATIONS_AND_ADR_REVIEW'
    except BaseException as exc:
        report.update(status='BLOCKED',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
