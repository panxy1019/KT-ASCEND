"""Compare a diagnostic subset against the original natural all-NPU oracle.

Reuse frozen metric verbatim. Join exact prompt/history IDs, not a forged
whole-corpus SHA. This report is never a qualification/promotion result.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--hybrid',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,'/workspace/r65-tools/migration_bridge/round6_5/tools')
    from evaluate_static_pairwise import metrics
    n=json.loads((a.baseline/'capture-result.json').read_text())
    h=json.loads((a.hybrid/'capture-result.json').read_text())
    if n['status']!='CAPTURE_INTEGRITY_PASS' or h['status']!='CAPTURE_INTEGRITY_PASS' or n['forced'] or not h['forced']:
        raise ValueError('ORACLE_CAPTURE_CONTRACT')
    baseline={r['rid']:r for r in n['rows']}
    trace=[json.loads(s) for s in (a.hybrid/'forcing.jsonl').read_text().splitlines()]
    natural={(r['rid'],r['step']):r['natural_token'] for r in trace}
    if len(trace)!=len(natural) or len(trace)!=h['forward_count']:raise ValueError('TRACE_COVERAGE')
    rows=[]
    for hr in h['rows']:
        rid=hr['rid'];nr=baseline[rid]
        for key in ('input_ids','baseline_output_ids'):
            if nr[key]!=hr[key]:raise ValueError('HISTORY_MISMATCH')
        nz=torch.load(a.baseline/(rid+'.logits.pt'),weights_only=True).float()
        hz=torch.load(a.hybrid/(rid+'.logits.pt'),weights_only=True).float()
        if nz.shape!=hz.shape or nz.shape!=(64,102400):raise ValueError('SHAPE')
        for i,token in enumerate(nr['baseline_output_ids']):
            rows.append(dict(rid=rid,token_index=i,**metrics(nz[i],hz[i],token,natural[(rid,i)])))
    bad=[r for r in rows if not r['pass_gate']]
    result=dict(status='DIAGNOSTIC_COMPARISON_COMPLETE',promotion_allowed=False,B_pair=2.1875,K=32,
        positions=len(rows),rows=rows,max_pairwise=max(r['max_pairwise'] for r in rows),
        overflow_count=sum(r['overflow'] for r in rows),first_failure=bad[0] if bad else None)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
