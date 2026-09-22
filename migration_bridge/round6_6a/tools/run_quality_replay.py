"""Bounded diagnostic after failed V1; no new quality scores or promotion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    base=Path('/workspace/results');tools=Path('/workspace/r66-v6/tools')
    old=Path('/workspace/r65-tools/migration_bridge/round6_5/tools')
    inputs=base/'r65-r66a-quality-discordances-v1.json'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    read=lambda p:json.loads(p.read_text())
    assert sha(inputs)=='81e78b3ce6902d49daf6b0c8558424ce4d9eca40d5fbed41c7376ad3fe29d4e1'
    assert read(base/'r65-r66-final-screen-v1/progress.json')['status']=='BLOCKED'
    assert read(base/'r65-r66-final-postprocess-v1/progress.json')['drift']==[]
    assert read(base/'r65-r66-final-screen-guard-v1/reaping.json')['complete']
    assert read(base/'r65-r66-raw-freeze-guard-v1/result.json')['reason']=='EXIT_ZERO'
    for item in read(base/'r65-r66-runtime-source-final-b0.json')['files']:
        assert sha(Path(item['path']))==item['sha256'],item['path']
    root=base/'r65-r66a-quality-replay-v1';root.mkdir(exist_ok=False)
    report=dict(status='RUNNING',input_sha256=sha(inputs),scope='DIAGNOSTIC_NOT_QUALIFICATION',events=[])
    save=lambda:(root/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(name,script,args,env=None):
        report['status']='RUNNING_'+name;save()
        with (root/(name+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(script),*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report['events'].append(name);save()
    try:
        save()
        for mode,profile in [('npu','four-R64'),('hybrid','four-R32')]:
            args=['--inputs',str(inputs),'--profile',str(base/'r65-profiles-v1'/(profile+'.json')),
                  '--output',str(root/mode)]
            if mode=='hybrid':args+=['--force',str(root/'npu/force.json')]
            run(mode+'_capture',old/'capture_static_logits.py',args,dict(os.environ,KT_R65_STAGEWISE='1'))
        npu={r['rid']:r for r in read(root/'npu/capture-result.json')['rows']}
        natural={}
        for line in (root/'hybrid/forcing.jsonl').read_text().splitlines():
            r=json.loads(line);natural[(r['rid'],r['step'])]=r['natural_token']
        parity=[]
        for r in read(inputs)['requests']:
            stop=r['original_first_difference']+1
            n=npu[r['rid']]['baseline_output_ids'][:stop]
            h=[natural[(r['rid'],i)] for i in range(stop)]
            assert n==r['original_npu_tokens'][:stop], 'NPU_DIAGNOSTIC_PREFIX_PARITY:'+r['rid']
            assert h==r['original_hybrid_tokens'][:stop], 'HYBRID_DIAGNOSTIC_PREFIX_PARITY:'+r['rid']
            parity.append(dict(rid=r['rid'],quality_id=r['quality_id'],state=r['state'],through_index=stop-1,
                               npu_prefix_exact=True,hybrid_common_history_exact=True))
        (root/'original_quality_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        run('transfer_audit',tools/'audit_study_transfers.py',[
            '--root',str(root/'hybrid'),'--profile',str(base/'r65-profiles-v1/four-R32.json'),
            '--output',str(root/'transfers.json')])
        run('logit_diagnostics',tools/'summarize_study_logits.py',[
            '--root',str(root),'--output',str(root/'logits.json')])
        run('router_diagnostics',tools/'summarize_study_routers.py',[
            '--root',str(root),'--output',str(root/'routers.json')])
        report['status']='DIAGNOSTIC_CAPTURE_COMPLETE_AWAIT_CAUSAL_REVIEW'
    except BaseException as exc:
        report.update(status='BLOCKED',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
