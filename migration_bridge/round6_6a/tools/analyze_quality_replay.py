"""Describe all original correctness-changing decision boundaries, no fitted gate."""
import argparse
import json
from pathlib import Path


def main():
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    spec=json.loads(a.inputs.read_text())
    assert json.loads((a.root/'progress.json').read_text())['status']=='DIAGNOSTIC_CAPTURE_COMPLETE_AWAIT_CAUSAL_REVIEW'
    routes={(r['rid'],r['token_index']):r for r in json.loads((a.root/'routers.json').read_text())['rows']}
    diagnostics={(r['rid'],r['token_index']):r for r in json.loads((a.root/'logits.json').read_text())['rows']}
    rows=[]
    for r in spec['requests']:
        t=r['original_first_difference'];rid=r['rid']
        n=torch.load(a.root/'npu'/(rid+'.logits.pt'),map_location='cpu',weights_only=True)[t].float()
        h=torch.load(a.root/'hybrid'/(rid+'.logits.pt'),map_location='cpu',weights_only=True)[t].float()
        na=r['original_npu_tokens'][t];ha=r['original_hybrid_tokens'][t]
        m_n=float(n[na]-n[ha]);m_h=float(h[na]-h[ha])
        rows.append(dict(quality_id=r['quality_id'],rid=rid,state=r['state'],token_index=t,
            npu_token=na,hybrid_token=ha,npu_margin=m_n,hybrid_margin=m_h,
            pair_margin_distortion=m_h-m_n,
            npu_top16_ids=n.topk(16).indices.tolist(),npu_top16_logits=n.topk(16).values.tolist(),
            hybrid_top16_ids=h.topk(16).indices.tolist(),hybrid_top16_logits=h.topk(16).values.tolist(),
            full_logit_diagnostics=diagnostics[(rid,t)],router_diagnostics=routes[(rid,t)]))
    report=dict(status='ORIGINAL_QUALITY_BOUNDARIES_DESCRIBED',rows=rows,
                scope='All correctness discordances, including gains; diagnostic not heldout or new threshold')
    with a.output.open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps([{k:r[k] for k in ('quality_id','state','token_index','npu_margin','hybrid_margin','pair_margin_distortion')}
                     for r in rows],indent=2))


if __name__=='__main__':main()
