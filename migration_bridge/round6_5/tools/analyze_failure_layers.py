"""Offline layerwise comparison of matched-history failure traces."""
import argparse
import json
from pathlib import Path


def main():
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();roots={name:a.root/name for name in ('npu','hybrid')}
    stages=['decoder_input_hidden','decoder_input_residual','attention_output','mlp_input',
        'mlp_residual','router_logits','router_topk_ids','router_weights','mlp_output',
        'decoder_output_hidden','decoder_output_residual']
    report=dict(status='DEBUG_LAYER_COMPARISON',positions={},observer_parity={})
    rid='r65-h2-h2_en_01'
    for name,root in roots.items():
        c=json.loads((root/'capture-result.json').read_text())
        if c['status']!='CAPTURE_INTEGRITY_PASS' or len(c['rows'])!=1:raise ValueError('TRACE_INTEGRITY')
        original=Path('/workspace/results/r65-numeric-r64-v1') if name=='npu' else Path('/workspace/results/r65-scale-matrix-v2/four-R32/numeric')
        z=torch.load(root/(rid+'.logits.pt'),weights_only=True).float()
        old=torch.load(original/(rid+'.logits.pt'),weights_only=True).float()
        report['observer_parity'][name]=dict(exact=torch.equal(z,old),
            max_abs=float((z-old).abs().max()),different_positions=torch.where((z!=old).any(dim=1))[0].tolist())
    def load(name,layer,stage,call):
        path=roots[name]/f'stagewise/model_layer{layer:02d}_{stage}_call{call:04d}.pt'
        if not path.exists():return None
        d=torch.load(path,map_location='cpu',weights_only=False)
        if d['layer']!=layer or d['stage']!=stage:raise ValueError('STAGE_IDENTITY')
        return d['tensor']
    for index in (53,54,55):
        rows=[]
        for layer in range(1,27):
            for stage in stages:
                n=load('npu',layer,stage,index+1);h=load('hybrid',layer,stage,index+1)
                if n is None or h is None:
                    rows.append(dict(layer=layer,stage=stage,status='MISSING',npu=n is not None,hybrid=h is not None));continue
                if n.shape!=h.shape:raise ValueError('STAGE_SHAPE')
                row=dict(layer=layer,stage=stage,status='COMPARED',exact=torch.equal(n,h),shape=list(n.shape))
                if stage=='router_topk_ids':
                    row.update(npu=n.tolist(),hybrid=h.tolist(),same_sets=torch.equal(n.sort().values,h.sort().values))
                else:
                    nf=n.float();hf=h.float();delta=hf-nf
                    row.update(max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/nf.norm().clamp_min(1e-30)))
                rows.append(row)
        report['positions'][index]=rows
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report['observer_parity']))
    print(json.dumps([r for r in report['positions'][54] if r.get('same_sets') is False]))


if __name__=='__main__':main()
