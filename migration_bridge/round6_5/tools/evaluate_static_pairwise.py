"""Apply the frozen B=2.1875/K=32 contract; never derive new thresholds."""
import argparse
import json
from pathlib import Path


def metrics(n,h,baseline_id):
    import torch
    if n.shape!=h.shape or n.ndim!=1 or not torch.isfinite(n).all() or not torch.isfinite(h).all():
        raise ValueError('LOGIT_SHAPE_OR_FINITE')
    a=int(baseline_id);b=int(h.argmax())
    if n[a]!=n.max():raise ValueError('BASELINE_NOT_MAX')
    # Include actual serving top1 in case tied top32 cutoff omits its ID.
    u=torch.unique(torch.cat((n.topk(32).indices,h.topk(32).indices,torch.tensor([a,b]))))
    relevant=u[u!=a]
    mn=n[a]-n[relevant];mh=h[a]-h[relevant]
    distortion=(mn-mh).abs();bound=2.1875
    stable=bool((mn>bound).all())
    overflow=int((distortion>bound).sum())
    membership=bool(n[a]-n[b]<=bound)
    return dict(stable=stable,stable_exact=(b==a) if stable else None,
                membership=membership,hybrid_top1=b,baseline_top1=a,
                ambiguity_size=int(((n[a]-n[u])<=bound).sum()),
                max_pairwise=float(distortion.max()),overflow=overflow,
                pass_gate=overflow==0 and (b==a if stable else membership))


def main():
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--hybrid',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    n=json.loads((a.baseline/'capture-result.json').read_text())
    h=json.loads((a.hybrid/'capture-result.json').read_text())
    if n['status']!='CAPTURE_INTEGRITY_PASS' or h['status']!='CAPTURE_INTEGRITY_PASS':raise ValueError('CAPTURE_NOT_PASS')
    if n['inputs_sha256']!=h['inputs_sha256'] or n['forced'] or not h['forced']:raise ValueError('WRONG_COMPARISON')
    rows=[]
    if len(n['rows'])!=len(h['rows']):raise ValueError('REQUEST_COUNT')
    for nr,hr in zip(n['rows'],h['rows']):
        for k in ('rid','input_ids','baseline_output_ids'):
            if nr[k]!=hr[k]:raise ValueError('MATCHED_HISTORY')
        rid=nr['rid']
        nz=torch.load(a.baseline/(rid+'.logits.pt'),weights_only=True).float()
        hz=torch.load(a.hybrid/(rid+'.logits.pt'),weights_only=True).float()
        if nz.shape!=hz.shape or nz.shape!=(64,102400):raise ValueError('LOGIT_MATRIX')
        for i,token in enumerate(nr['baseline_output_ids']):
            rows.append(dict(rid=rid,token_index=i,**metrics(nz[i],hz[i],token)))
    bad=[r for r in rows if not r['pass_gate']]
    result=dict(status='STATIC_PAIRWISE_PASS' if not bad else 'STATIC_NUMERICAL_SCALE_BLOCKER',
                B_pair=2.1875,K=32,positions=len(rows),rows=rows,first_failure=bad[0] if bad else None,
                overflow_count=sum(r['overflow'] for r in rows),
                max_pairwise=max(r['max_pairwise'] for r in rows),
                stable=sum(r['stable'] for r in rows),stable_exact=sum(r['stable_exact'] is True for r in rows),
                ambiguous=sum(not r['stable'] for r in rows),
                membership=sum(not r['stable'] and r['membership'] for r in rows))
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    if bad:raise SystemExit(2)


if __name__=='__main__':main()
