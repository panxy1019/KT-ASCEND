"""One-shot held-out capture, requires an immutable reviewed QD candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--candidate-sha256',required=True)
    a=p.parse_args()
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(a.candidate)==a.candidate_sha256
    candidate=json.loads(a.candidate.read_text())
    assert candidate['status']=='FROZEN_BEFORE_HD6' and candidate['heldout_execution_allowed']
    frozen=Path('/workspace/results/r65-r66-study-freeze-v1')
    assert sha(frozen/'hd6.json')==candidate['hd6_sha256']=='e812852ccd9ca9f9427b31f56c29ebca8167c7b7116ecd27b8c8e2aa973ab15f'
    assert sha(frozen/'CONTRACT_V1.json')==candidate['contract_sha256']
    root=Path('/workspace/results/r65-r66-hd6-v1');root.mkdir(exist_ok=False)
    report=dict(status='RUNNING',events=[],candidate_sha256=a.candidate_sha256,
                quality_started=False,refitting_allowed=False)
    def save():(root/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        save()
        for mode,profile in [('npu','four-R64'),('hybrid','four-R32')]:
            runtime=Path('/workspace/results/r65-r66-runtime-source-final-b0.json')
            assert sha(runtime)==candidate['runtime_manifest_sha256']
            for item in json.loads(runtime.read_text())['files']:
                assert sha(Path(item['path']))==item['sha256'],'ACTUAL_RUNTIME_DRIFT'
            args=[sys.executable,'/workspace/r65-tools/migration_bridge/round6_5/tools/capture_static_logits.py',
                  '--inputs',str(frozen/'hd6.json'),'--profile','/workspace/results/r65-profiles-v1/'+profile+'.json',
                  '--output',str(root/mode)]
            if mode=='hybrid':args+=['--force',str(root/'npu/force.json')]
            env=dict(os.environ,KT_R65_STAGEWISE='1')
            if 'r66-debug' in env.get('PYTHONPATH',''):raise ValueError('DIAGNOSTIC_EXTENSION')
            report['status']='RUNNING_'+mode.upper();save()
            with (root/(mode+'.log')).open('x') as log:
                subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((root/mode/'capture-result.json').read_text())
            lifecycle=json.loads((root/mode/'lifecycle.json').read_text())
            assert result['status']=='CAPTURE_INTEGRITY_PASS' and result['forward_count']==780 and len(result['rows'])==12
            assert lifecycle['status']=='COMPLETE' and lifecycle['reaping']['complete']
            report['events'].append(mode+'_CAPTURE_INTEGRITY_PASS');save()
        report['status']='HD6_CAPTURED_AWAIT_ONE_SHOT_AUDIT'
    except BaseException as exc:
        report.update(status='BLOCKED',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
