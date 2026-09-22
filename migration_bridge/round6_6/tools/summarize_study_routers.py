"""Router discontinuity/exposure diagnostics on matched histories, read-only."""
import argparse
from collections import defaultdict
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
    assert n['inputs_sha256']==h['inputs_sha256']
    def load(mode,layer,name,call):
        path=a.root/mode/'stagewise'/f'model_layer{layer:02d}_{name}_call{call:04d}.pt'
        return torch.load(path,map_location='cpu',weights_only=False)['tensor']
    rows=[];counts=defaultdict(lambda:dict(positions=0,set_flips=0,jaccard_sum=0.))
    offset=0
    for nr,hr in zip(n['rows'],h['rows']):
        assert nr['rid']==hr['rid'] and nr['baseline_output_ids']==hr['baseline_output_ids']
        for step in range(len(nr['baseline_output_ids'])):
            call=offset+step+1;first=None;layer_rows=[]
            for layer in range(1,27):
                ni=load('npu',layer,'router_topk_ids',call)[-1].tolist()
                hi=load('hybrid',layer,'router_topk_ids',call)[-1].tolist()
                assert len(set(ni))==len(set(hi))==6
                ns,hs=set(ni),set(hi);flipped=ns!=hs
                if flipped and first is None:first=layer
                j=len(ns&hs)/len(ns|hs)
                counts[layer]['positions']+=1;counts[layer]['set_flips']+=flipped
                counts[layer]['jaccard_sum']+=j
                if layer in (1,9,17,26) or flipped:
                    nx=load('npu',layer,'mlp_input',call)[-1].float()
                    hx=load('hybrid',layer,'mlp_input',call)[-1].float()
                    w=load('hybrid',layer,'router_weights',call)[-1].float()
                    assert torch.isfinite(nx).all() and torch.isfinite(hx).all() and torch.isfinite(w).all()
                    cpu=[i for i,e in enumerate(hi) if layer in (1,9,17,26) and e%2==1]
                    layer_rows.append(dict(layer=layer,npu_ids=ni,hybrid_ids=hi,jaccard=j,
                        hidden_rel_l2=float((hx-nx).norm()/nx.norm().clamp_min(1e-20)),
                        cpu_route_events=len(cpu),cpu_route_weight_mass=float(w[cpu].sum())))
            rows.append(dict(rid=nr['rid'],token_index=step,first_router_set_divergence_layer=first,layers=layer_rows))
        offset+=len(nr['steps'])
    assert offset==n['forward_count']
    report=dict(status='ROUTER_DIAGNOSTICS_COMPLETE',positions=len(rows),per_layer=dict(counts),rows=rows,
                interpretation='Different hidden inputs may cross TopK boundaries; not automatically a router implementation defect')
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(counts)))


if __name__=='__main__':main()
