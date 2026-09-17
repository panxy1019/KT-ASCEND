"""Reuse frozen C32/C64 same-operand oracle, only early/mid/late reps."""
import json
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('base',type=Path)
    p.add_argument('--runtime-env',type=Path)
    a=p.parse_args();base=a.base;out=base/'operator-representatives';out.mkdir(exist_ok=False)
    source=Path('/workspace/c64-dev/tools-v16/migration_bridge/round6_7_c64/tools')
    report=dict(status='RUNNING',runs=[],gate=.01)
    try:
        for layer in (1,13,26):
            for tp in (1,2):
                target=out/f'tp{tp}-L{layer:02d}'
                cmd=['bash',str(a.runtime_env or source/'with_c64_env.sh'),str(source/'operator_oracle.py'),
                     '--layer',str(layer),'--tp',str(tp),'--output',str(target)]
                if layer in (1,26):cmd+=['--prefill']
                with (out/f'tp{tp}-L{layer:02d}.log').open('x') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
                r=json.loads((target/'result.json').read_text());assert r['status']=='PASS'
                report['runs'].append(dict(layer=layer,tp=tp,rows=len(r['rows']),
                    max_rel_l2=max(v for row in r['rows'] for v in row['metrics'].values())))
                (out/'progress.json').write_text(json.dumps(report,indent=2))
        report['status']='PASS'
    except BaseException as e:report.update(status='FAIL',error=repr(e));raise
    finally:(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
