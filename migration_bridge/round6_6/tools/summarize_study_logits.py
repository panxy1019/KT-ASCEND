"""Matched-history diagnostics; never refits a bound or promotes qualification."""
import argparse
import json
from pathlib import Path


def main():
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    n=json.loads((a.root/'npu/capture-result.json').read_text())
    h=json.loads((a.root/'hybrid/capture-result.json').read_text())
    assert n['status']==h['status']=='CAPTURE_INTEGRITY_PASS'
    assert n['inputs_sha256']==h['inputs_sha256'] and not n['forced'] and h['forced']
    assert n['forward_count']==h['forward_count']
    assert [r['rid'] for r in n['rows']]==[r['rid'] for r in h['rows']]
    forcing=[json.loads(s) for s in (a.root/'hybrid/forcing.jsonl').read_text().splitlines()]
    choices={(r['rid'],r['step']):r['natural_token'] for r in forcing}
    rows=[]
    for nr,hr in zip(n['rows'],h['rows']):
        assert nr['input_ids']==hr['input_ids'] and nr['baseline_output_ids']==hr['baseline_output_ids']
        rid=nr['rid']
        nz=torch.load(a.root/'npu'/(rid+'.logits.pt'),map_location='cpu',weights_only=True).float()
        hz=torch.load(a.root/'hybrid'/(rid+'.logits.pt'),map_location='cpu',weights_only=True).float()
        assert nz.shape==hz.shape==(len(nr['baseline_output_ids']),102400)
        assert torch.isfinite(nz).all() and torch.isfinite(hz).all()
        for t,(x,y,token) in enumerate(zip(nz,hz,nr['baseline_output_ids'])):
            hybrid=int(choices[rid,t])
            assert y[hybrid]==y.max() and x[token]==x.max()
            diff=y-x
            union=torch.unique(torch.cat((x.topk(32).indices,y.topk(32).indices,
                                          torch.tensor([token,hybrid]))))
            margins=x[token]-x[union]
            distortion=(y[token]-y[union])-margins
            stable=bool((margins[union!=token]>2.1875).all())
            rows.append(dict(rid=rid,token_index=t,baseline_top1=token,hybrid_top1=hybrid,
                top1_equal=token==hybrid,max_abs=float(diff.abs().max()),mean_abs=float(diff.abs().mean()),
                relative_l2=float(diff.norm()/x.norm().clamp_min(1e-20)),
                cosine=float(torch.nn.functional.cosine_similarity(x,y,dim=0)),
                max_pairwise=float(distortion.abs().max()),
                historical_B_pair_overflow_pairs=int((distortion.abs()>2.1875).sum()),
                historical_stable=stable,historical_stable_exact=(not stable or token==hybrid),
                historical_ambiguity_membership=bool(x[token]-x[hybrid]<=2.1875)))
    distribution={}
    for key in ['max_abs','mean_abs','relative_l2','cosine','max_pairwise']:
        values=torch.tensor([r[key] for r in rows],dtype=torch.float64)
        distribution[key]={name:float(torch.quantile(values,q)) for name,q in
                           [('min',0.),('p50',.5),('p95',.95),('p99',.99),('max',1.)]}
    result=dict(status='MATCHED_HISTORY_LOGIT_DIAGNOSTICS_COMPLETE',positions=len(rows),
                distribution=distribution,rows=rows,
                historical_bound_role='DIAGNOSTIC_ONLY; original Round6.5 BLOCKED unchanged',
                candidate_approved=False,heldout_verified=False,quality_verified=False)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(distribution,indent=2))


if __name__=='__main__':main()
