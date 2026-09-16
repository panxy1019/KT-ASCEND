"""Independently reject hidden guard failures in completed static evidence."""
import argparse
import json
from pathlib import Path


def audit(root):
    rows=[]
    for path in sorted(root.rglob('guard/result.json')):
        guard=json.loads(path.read_text())
        reaping=json.loads(path.with_name('reaping.json').read_text())
        # Serving is deliberately terminated after the bounded request set.
        # Kernel probes/exporters instead finish normally with EXIT_ZERO.
        ok=(guard['reason'] in ('CANCELLED','EXIT_ZERO')
            and guard.get('matched_error') is None and reaping['complete'])
        rows.append(dict(path=str(path),reason=guard['reason'],reaped=reaping['complete'],pass_gate=ok))
    if not rows:raise ValueError('NO_COMPLETED_GUARDS')
    return dict(status='GUARD_AUDIT_PASS' if all(r['pass_gate'] for r in rows) else 'GUARD_AUDIT_FAIL',rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.root)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
    if result['status']!='GUARD_AUDIT_PASS':raise SystemExit(2)


if __name__=='__main__':main()
