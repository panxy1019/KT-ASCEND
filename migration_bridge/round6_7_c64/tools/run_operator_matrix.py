"""Bounded sequential operator qualification, no serving/quality promotion."""
import argparse
from pathlib import Path
import subprocess
import time
from common import sha, write
import json

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
tools=Path(__file__).parent
report=dict(status='RUNNING',started=time.time(),runs=[],native_sha256=sha(
    '/workspace/c64-dev/runtime/kt_kernel/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'),
    wrapper_sha256=sha('/workspace/c64-dev/runtime/kt_kernel/utils/llamafile.py'),
    oracle_sha256=sha(tools/'operator_oracle.py'))
write(a.output/'progress.json',report)
try:
    for layer in [17,*[i for i in range(1,27) if i!=17]]:
        for tp in (1,2):
            target=a.output/f'tp{tp}-L{layer:02d}'
            cmd=['bash',str(tools/'with_c64_env.sh'),str(tools/'operator_oracle.py'),
                 '--layer',str(layer),'--tp',str(tp),'--output',str(target)]
            if layer in (1,9,17,26):cmd+=['--prefill']
            with (a.output/f'tp{tp}-L{layer:02d}.log').open('x') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            result=json.loads((target/'result.json').read_text())
            assert result['status']=='PASS'
            report['runs'].append(dict(layer=layer,tp=tp,rows=len(result['rows']),
                result_sha256=sha(target/'result.json'),max_rel_l2=max(
                    v for r in result['rows'] for v in r['metrics'].values())))
            write(a.output/'progress.json',report);print(json.dumps(report['runs'][-1]),flush=True)
    report['status']='PASS'
except BaseException as exc:
    report.update(status='FAIL',error=repr(exc));raise
finally:
    report['ended']=time.time();write(a.output/'progress.json',report);write(a.output/'summary.json',report)
