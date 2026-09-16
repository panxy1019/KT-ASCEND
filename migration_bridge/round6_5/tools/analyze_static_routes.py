"""Audit bounded natural route capture; distinguish layer rows and model tokens."""
import argparse
import json
from pathlib import Path


def main():
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cap=json.loads((a.capture/'report.json').read_text())
    if cap['status']!='ROUTE_CAPTURE_PARITY_PASS':raise ValueError('CAPTURE_NOT_PASS')
    profile=json.loads(a.profile.read_text());stage=a.capture/'stagewise'
    results=[];full_hits={};controls=[]
    def load(source,layer,name,call):
        path=stage/f'{source}_layer{layer:02d}_{name}_call{call:04d}.pt'
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data['layer']!=layer or data['stage']!=name:raise ValueError('STAGE_IDENTITY')
        return data['tensor']
    for row in profile['rows']:
        layer=row['layer'];mask=torch.zeros(64,dtype=torch.bool);mask[row['cpu_ids']]=True
        per_expert=torch.zeros(64,dtype=torch.int64)
        stats={phase:dict(layer_token_rows=0,cpu_events=0,cpu_weight_mass=0.,total_weight_mass=0.,touches_cpu=0)
               for phase in ('prefill','decode','lookahead')}
        expected=36
        if len(list(stage.glob(f'model_layer{layer:02d}_router_topk_ids_call*.pt')))!=expected:
            raise ValueError(f'ROUTER_CALL_COUNT_LAYER_{layer}')
        eligible=0;verified=0
        for call in range(1,expected+1):
            ids=load('model',layer,'router_topk_ids',call).long()
            weights=load('model',layer,'router_weights',call)
            if ids.ndim!=2 or ids.shape[1]!=6 or ids.shape!=weights.shape or (ids<0).any() or (ids>=64).any():
                raise ValueError('ROUTER_SHAPE_IDS')
            if not torch.isfinite(weights).all():raise ValueError('ROUTING_NONFINITE')
            selected=mask[ids];hits=selected.any(dim=1)
            phase='prefill' if (call-1)%9==0 else ('lookahead' if call%9==0 else 'decode')
            s=stats[phase];s['layer_token_rows']+=len(ids);s['cpu_events']+=int(selected.sum())
            s['cpu_weight_mass']+=float(weights[selected].float().sum())
            s['total_weight_mass']+=float(weights.float().sum());s['touches_cpu']+=int(hits.sum())
            if phase!='lookahead':per_expert+=torch.bincount(ids[selected],minlength=64)
            if call not in full_hits:full_hits[call]=hits.clone()
            elif len(full_hits[call])!=len(hits):raise ValueError('CROSS_LAYER_TOKEN_SHAPE')
            else:full_hits[call]|=hits
            if row['cpu_ids']:
                k_ids=load('ktep',layer,'topk_ids',call).long()
                k_weights=load('ktep',layer,'routing_weights',call)
                if not torch.equal(k_ids,ids) or not torch.equal(k_weights,weights):raise ValueError('ROUTING_BOUNDARY')
                cpu=load('ktep',layer,'cpu_partial',call)
                npu=load('ktep',layer,'npu_owned_partial',call)
                merged=load('ktep',layer,'routed_merge',call)
                if not all(torch.isfinite(x).all() for x in (cpu,npu,merged)):raise ValueError('PARTIAL_NONFINITE')
                reference=(npu.float()+cpu.float()).to(merged.dtype)
                if not torch.equal(reference,merged):raise ValueError('FP32_MERGE_BOUNDARY')
                nohit=~hits
                eligible+=int(nohit.sum())
                if nohit.any():
                    if not torch.equal(cpu[nohit],torch.zeros_like(cpu[nohit])):raise ValueError('CPU_NOT_HIT_CONTAMINATION')
                    if not torch.equal(merged[nohit],npu[nohit].to(merged.dtype)):raise ValueError('NOHIT_MERGE')
                    verified+=int(nohit.sum())
        results.append(dict(layer=layer,cpu_ids=row['cpu_ids'],per_expert_hits=per_expert.tolist(),stats=stats))
        if row['cpu_ids']:controls.append(dict(layer=layer,eligible=eligible,verified=verified,
                    status='PASS' if verified else 'NOT_OBSERVED'))
    whole={phase:dict(token_rows=0,touches_cpu=0) for phase in ('prefill','decode','lookahead')}
    for call,hits in full_hits.items():
        phase='prefill' if (call-1)%9==0 else ('lookahead' if call%9==0 else 'decode')
        whole[phase]['token_rows']+=len(hits);whole[phase]['touches_cpu']+=int(hits.sum())
    command=json.loads((a.capture/'resolved-command.json').read_text())
    structure=[]
    if '--forward-hooks' in command['argv']:
        structure=[json.loads(x) for x in (a.capture/'physical-structure.jsonl').read_text().splitlines()]
        if len(structure)!=26 or {r['layer'] for r in structure}!=set(range(1,27)):
            raise ValueError('PHYSICAL_STRUCTURE_COVERAGE')
    result=dict(status='ROUTE_SEMANTIC_CONTROL_PASS',layers=results,whole_model=whole,cpu_not_hit=controls,
                sample_scope='FOUR_FROZEN_S6_PROMPTS_8_TARGETS_PLUS_ONE_LOOKAHEAD',
                physical_weight_shape_verified=bool(structure),physical_structure=structure)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='layers'}))


if __name__=='__main__':main()
