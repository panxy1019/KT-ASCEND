"""Read-only complete-study transfer/merge/coverage audit; no NPU execution."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=json.loads((a.root/'capture-result.json').read_text())
    assert result['status']=='CAPTURE_INTEGRITY_PASS'
    calls=result['forward_count']
    profile=json.loads(a.profile.read_text())
    stage=a.root/'stagewise'
    def load(src,layer,name,call):
        return torch.load(stage/f'{src}_layer{layer:02d}_{name}_call{call:04d}.pt',
                          map_location='cpu',weights_only=False)['tensor']
    rows=[]
    for config in profile['rows']:
        owned=config['cpu_ids'];layer=config['layer']
        if not owned:continue
        hits=Counter();eligible=0;positions=0;mass=0.
        for call in range(1,calls+1):
            ids=load('model',layer,'router_topk_ids',call)
            weights=load('model',layer,'router_weights',call)
            hidden=load('model',layer,'mlp_input',call)
            cpu=load('ktep',layer,'cpu_partial',call)
            npu=load('ktep',layer,'npu_owned_partial',call)
            merged=load('ktep',layer,'routed_merge',call)
            checks={
                'D2H_HIDDEN':torch.equal(hidden,load('cpu',layer,'input_hidden_after_sync',call)),
                'D2H_IDS':torch.equal(ids.long(),load('cpu',layer,'immediate_ids_after_sync',call).long()),
                'D2H_WEIGHTS':torch.equal(weights.float(),load('cpu',layer,'routing_weights_after_sync',call).float()),
                'DEFERRED_OFF':bool((load('cpu',layer,'deferred_ids_after_sync',call)==-1).all()),
                'H2D':torch.equal(cpu,load('cpu',layer,'cpu_output_after_sync',call)),
                'FP32_MERGE':torch.equal(merged,(cpu.float()+npu.float()).to(merged.dtype)),
                'FINITE':all(bool(torch.isfinite(x).all()) for x in (hidden,weights,cpu,npu,merged)),
                'IDS_IN_RANGE':bool(((ids>=0)&(ids<64)).all()),
            }
            for name,ok in checks.items():
                if not ok:raise ValueError(f'{name}:L{layer}:call{call}')
            selected=torch.zeros_like(ids,dtype=torch.bool)
            for expert in owned:
                hit=(ids==expert);hits[expert]+=int(hit.sum());selected|=hit
            nohit=~selected.any(dim=-1)
            if not bool((cpu[nohit]==0).all()) or not torch.equal(merged[nohit],npu[nohit].to(merged.dtype)):
                raise ValueError(f'CPU_NOT_HIT:L{layer}:call{call}')
            eligible+=int(nohit.sum());positions+=ids.shape[0]
            mass+=float(weights.float()[selected].sum())
        rows.append(dict(layer=layer,forwards=calls,positions=positions,
                         cpu_not_hit_exact_positions=eligible,expert_hits=dict(hits),
                         cpu_route_events=sum(hits.values()),cpu_route_weight_mass=mass))
    structures=[json.loads(s) for s in (a.root/'physical-structure.jsonl').read_text().splitlines()]
    expected={r['layer']:r for r in profile['rows']}
    assert len(structures)==len(expected)==26
    assert len({s['layer'] for s in structures})==26
    for s in structures:
        r=expected[s['layer']]
        assert s['npu_ids']==r['npu_ids'] and s['cpu_ids']==r['cpu_ids']
        assert set(r['npu_ids']).isdisjoint(r['cpu_ids'])
        assert sorted(r['npu_ids']+r['cpu_ids'])==list(range(64))
        for w in s['weights'].values():assert w['shape'][0]==len(r['npu_ids'])
    report=dict(status='STUDY_TRANSFER_MERGE_COVERAGE_AUDIT_PASS',rows=rows,
                physical_structure=structures,
                scope='Observed transfers, merge and ownership; not standalone proof of all model semantics')
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(rows))


if __name__=='__main__':main()
