"""Inspect all four CPU layers across the65failure-replay forwards, no NPU work."""
import argparse
import json
from pathlib import Path


def main():
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    profile=json.loads(Path('/workspace/results/r65-profiles-v1/four-R32.json').read_text())
    stage=a.root/'hybrid/stagewise';rows=[]
    def load(source,layer,name,call):
        return torch.load(stage/f'{source}_layer{layer:02d}_{name}_call{call:04d}.pt',
            map_location='cpu',weights_only=False)['tensor']
    for layer in (1,9,17,26):
        config=next(r for r in profile['rows'] if r['layer']==layer)
        count=0
        for call in range(1,66):
            ids=load('model',layer,'router_topk_ids',call)
            weights=load('model',layer,'router_weights',call)
            x=load('model',layer,'mlp_input',call)
            cx=load('cpu',layer,'input_hidden_after_sync',call)
            ci=load('cpu',layer,'immediate_ids_after_sync',call)
            cw=load('cpu',layer,'routing_weights_after_sync',call)
            deferred=load('cpu',layer,'deferred_ids_after_sync',call)
            if not torch.equal(x,cx):raise ValueError(f'D2H_HIDDEN:{layer}:{call}')
            if not torch.equal(ids.long(),ci.long()):raise ValueError(f'D2H_IDS:{layer}:{call}')
            if not torch.equal(weights.float(),cw.float()):raise ValueError(f'D2H_WEIGHTS:{layer}:{call}')
            if (deferred!=-1).any():raise ValueError('UNEXPECTED_DEFERRED')
            cpu=load('ktep',layer,'cpu_partial',call);npu=load('ktep',layer,'npu_owned_partial',call)
            merged=load('ktep',layer,'routed_merge',call)
            if not torch.equal(cpu,load('cpu',layer,'cpu_output_after_sync',call)):raise ValueError('H2D_OUTPUT')
            if not torch.equal(merged,(cpu.float()+npu.float()).to(merged.dtype)):raise ValueError('MERGE_BOUNDARY')
            if not all(torch.isfinite(t).all() for t in (cx,cw,cpu,npu,merged)):raise ValueError('NONFINITE')
            count+=1
        ids=load('model',layer,'router_topk_ids',55)[0].tolist()
        first=next(e for e in ids if e in config['cpu_ids'])
        rows.append(dict(layer=layer,forwards=count,first_routed_cpu_expert_at_failure=first,
            routing_ids_at_failure=ids,transfer_exact=True,merge_exact=True,deferred_off=True))
    structures=[json.loads(s) for s in (a.root/'hybrid/physical-structure.jsonl').read_text().splitlines()]
    if len(structures)!=26:raise ValueError('STRUCTURE_COUNT')
    report=dict(status='FAILURE_TRANSFER_AND_MERGE_AUDIT_PASS',rows=rows,physical_structure=structures)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(rows))


if __name__=='__main__':main()
