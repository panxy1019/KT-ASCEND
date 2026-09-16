"""C1/C2/C3 diagnostic replay; C4 is the immutable reproduced failure.

No profile promotion. Reuse all original selected-layer masks and token history.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    import torch
    p=argparse.ArgumentParser()
    p.add_argument('--tools', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--existing-c1', type=Path)
    a=p.parse_args()
    sys.path.insert(0,str(a.tools))
    from build_ratio_profiles import build
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path('/workspace/models/DeepSeek-V2-Lite-604d5664/config.json').read_text())
    trace=Path('/workspace/results/r65-four-R32-localization-v1')
    frozen=json.loads(Path('/workspace/results/r65-profiles-v1/four-R32.json').read_text())
    report=dict(status='RUNNING',scope='DIAGNOSTIC_ONLY_NO_PROMOTION',rows=[],
                baseline=str(trace/'npu'),C4=str(trace/'hybrid'))
    try:
        for count in (1,2,3):
            name=f'C{count}'
            data=build(cfg,[1,9,17,26][:count],32)
            for row in data['rows']:
                if row['layer'] in data['selected_layers']:
                    assert row==next(r for r in frozen['rows'] if r['layer']==row['layer'])
            pt=a.output/(name+'.pt')
            mask=torch.zeros((1,27,64),dtype=torch.int64)
            for row in data['rows']:mask[0,row['layer'],row['npu_ids']]=1
            torch.save(dict(logical_count=mask),pt)
            data['pt_sha256']=hashlib.sha256(pt.read_bytes()).hexdigest()
            profile=a.output/(name+'.json');profile.write_text(json.dumps(data,indent=2)+'\n')
            report['active']=name
            (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
            env=dict(os.environ,KT_R65_STAGEWISE='1')
            capture=a.existing_c1 if count==1 and a.existing_c1 else a.output/name
            if capture==a.output/name:
                with (a.output/(name+'.log')).open('x') as log:
                    subprocess.run([sys.executable,str(a.tools/'capture_static_logits.py'),
                        '--inputs',str(trace/'inputs.json'),'--force',str(trace/'force.json'),
                        '--profile',str(profile),'--output',str(capture)],
                        env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            else:
                actual=json.loads((capture/'resolved-command.json').read_text())['profile']
                if actual['canonical_sha256']!=data['canonical_sha256']:
                    raise ValueError('REUSED_C1_PROFILE_MISMATCH')
            result=a.output/(name+'-pairwise.json')
            r=subprocess.run([sys.executable,str(Path(__file__).parent/'compare_diagnostic.py'),
                '--baseline','/workspace/results/r65-numeric-r64-v1','--hybrid',str(capture),'--output',str(result)])
            if r.returncode not in (0,2):raise RuntimeError('PAIRWISE_TOOL_ERROR')
            report['rows'].append(dict(profile=name,result=json.loads(result.read_text())))
        report['status']='BISECTION_CAPTURE_COMPLETE_REVIEW_REQUIRED'
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc));raise
    finally:
        (a.output/'cumulative_bisection.json').write_text(json.dumps(report,indent=2)+'\n')
        (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
