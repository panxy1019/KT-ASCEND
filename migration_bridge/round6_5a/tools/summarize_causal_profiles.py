"""Offline layerwise diagnostics, never promotion or tolerance fitting."""
import argparse
import json
from pathlib import Path
import torch
from numerics import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--profiles',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    torch.set_num_threads(1)
    profiles=json.loads(a.profiles.read_text())
    baseline=Path('/workspace/results/r65-four-R32-localization-v1/npu/stagewise')
    report=dict(scope='DIAGNOSTIC_ONLY',profiles={})
    for name,path in profiles.items():
        root=Path(path)
        capture=json.loads((root/'capture-result.json').read_text())
        if capture['status']!='CAPTURE_INTEGRITY_PASS':raise ValueError('CAPTURE_INTEGRITY')
        rows=[];first_router_change=None
        for layer in range(1,27):
            row=dict(layer=layer)
            for stage in ('decoder_input_hidden','decoder_input_residual','attention_output',
                          'mlp_input','mlp_output','decoder_output_hidden','decoder_output_residual'):
                file=f'model_layer{layer:02d}_{stage}_call0055.pt'
                n=torch.load(baseline/file,map_location='cpu',weights_only=False)['tensor']
                h=torch.load(root/'stagewise'/file,map_location='cpu',weights_only=False)['tensor']
                row[stage]=metrics(h,n)
            file=f'model_layer{layer:02d}_router_topk_ids_call0055.pt'
            n=torch.load(baseline/file,map_location='cpu',weights_only=False)['tensor'].long()
            h=torch.load(root/'stagewise'/file,map_location='cpu',weights_only=False)['tensor'].long()
            equal=torch.equal(n.sort().values,h.sort().values)
            row['router']=dict(npu=n.tolist(),diagnostic=h.tolist(),same_set=equal)
            if not equal and first_router_change is None:first_router_change=layer
            rows.append(row)
        filename='r65-h2-h2_en_01.logits.pt'
        n=torch.load(Path('/workspace/results/r65-numeric-r64-v1')/filename,weights_only=True).float()
        h=torch.load(root/filename,weights_only=True).float()
        full_logits=dict(exact=torch.equal(n,h),exact_positions=int((n==h).all(dim=1).sum()),
            max_abs=float((n-h).abs().max()),failure_token54=metrics(h[54],n[54]))
        report['profiles'][name]=dict(first_router_set_change=first_router_change,rows=rows,full_logits=full_logits,
            logits_file_sha256=__import__('hashlib').sha256((root/filename).read_bytes()).hexdigest())
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')


if __name__=='__main__':main()
