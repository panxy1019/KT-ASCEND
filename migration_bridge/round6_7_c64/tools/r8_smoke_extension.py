"""R8 captured local subset and fixed 16-request short timing before CMMLU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from common import sha,write
from serial_session import session,admission
sys.path.append('/workspace/r67-tools-v5/migration_bridge/round6_7/tools')
from run_serial_sweep import request

p=argparse.ArgumentParser();p.add_argument('--canary',type=Path,required=True)
p.add_argument('--resume-local',action='store_true');a=p.parse_args()
c=json.loads(a.canary.read_text());assert c['status']=='PASS' and c['profile']=='R8'
root=a.canary.parent/'extended-smoke';root.mkdir(exist_ok=a.resume_local)
tools=Path(__file__).parent;rows=[]
for layer in (1,9,17,26):
    for tp in (1,2):
        out=root/f'tp{tp}-L{layer:02d}'
        if a.resume_local:
            assert (out/'result.json').exists(), 'MISSING_LOCAL_EVIDENCE'
        else:
          with (root/f'tp{tp}-L{layer:02d}.log').open('x') as log:
            subprocess.run(['bash',str(tools/'with_c64_env.sh'),str(tools/'operator_oracle.py'),
                '--layer',str(layer),'--tp',str(tp),'--operands',str(a.canary.parent/'observer/operands'),
                '--output',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        result=json.loads((out/'result.json').read_text());assert result['status']=='PASS'
        if tp==2:rows.extend(result['rows'])
manifest=Path('/workspace/c64-dev/performance-freeze-v1/perf_prompt_set.json')
source=json.loads(manifest.read_text())['rows'][-1]['input_ids']
prompts=[dict(id=f'r8-smoke-{n}',input_ids=source[:n]) for n in range(128,609,32)]
assert len(prompts)==16
write(root/'timing-freeze.json',dict(prompts=prompts,source_sha256=sha(manifest),output_tokens=1,
    use='DESCRIPTIVE_SHORT_SMOKE_NOT_FORMAL_ABBA'))
os.environ.update(C64_TP='2',KT_R67_CPU_SET='128-191',KT_R67_NPU_ID='0')
timings=[]
with session(root/('short-timing-retry1' if a.resume_local else 'short-timing'),a.canary.parent/'R8.json') as output:
    for row in prompts:
        admission();_,ids,seconds=request(output,row,row['id'],1)
        timings.append(dict(id=row['id'],seconds=seconds,output_ids=ids))
write(root/'summary.json',dict(status='PASS',local_cases=len(rows),local_layers=[1,9,17,26],
    max_rel_l2=max(v for r in rows for v in r['metrics'].values()),timing_requests=16,timings=timings,
    canary_sha256=sha(a.canary)))
