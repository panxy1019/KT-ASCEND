"""Same deployed router GEMM and Ascend TopK operator on both captured inputs."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open
from sglang.jit_kernel.deepseek_v4 import linear_bf16_fp32
from probe_npu_subset import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    cfg=json.loads((a.model/'config.json').read_text())
    if cfg['topk_method']!='greedy':raise ValueError('UNEXPECTED_ROUTER')
    index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    source=Path(inspect.getsourcefile(linear_bf16_fp32))
    report=dict(status='RUNNING',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=[])
    def load(mode,layer,stage,call=55):
        return torch.load(a.trace/f'{mode}/stagewise/model_layer{layer:02d}_{stage}_call{call:04d}.pt',
                          map_location='cpu',weights_only=False)['tensor']
    for layer in (12,17):
        key=f'model.layers.{layer}.mlp.gate.weight'
        with safe_open(a.model/index[key],framework='pt',device='cpu') as sf:
            weight=sf.get_tensor(key).clone().to('npu',dtype=torch.bfloat16)
        row=dict(layer=layer,paths={})
        values={}
        for mode in ('npu','hybrid'):
            x=load(mode,layer,'mlp_input')
            expected=load(mode,layer,'router_logits')
            logits=linear_bf16_fp32(x.to('npu'),weight)
            _,ids,_=torch.ops.npu.npu_moe_gating_top_k_softmax(logits,k=6)
            actual=logits.cpu();order=torch.argsort(actual[0],descending=True,stable=True).tolist()
            ids=ids.cpu().long();captured_ids=load(mode,layer,'router_topk_ids').long()
            row['paths'][mode]=dict(logits_vs_capture=metrics(actual,expected),ids=ids.tolist(),
                ids_exact=torch.equal(ids,captured_ids),rank6_7=order[5:7],
                rank6_7_margin=float(actual[0,order[5]]-actual[0,order[6]]),
                E62=float(actual[0,62]),E29=float(actual[0,29]),
                E62_minus_E29=float(actual[0,62]-actual[0,29]),
                E62_rank=order.index(62)+1,E29_rank=order.index(29)+1)
            values[mode]=(x,actual,set(ids[0].tolist()))
        n,h=values['npu'],values['hybrid']
        row.update(hidden_error=metrics(h[0],n[0]),router_error=metrics(h[1],n[1]),
                   topk_jaccard=len(n[2]&h[2])/len(n[2]|h[2]),
                   pair_score_perturbation=row['paths']['hybrid']['E62_minus_E29']-row['paths']['npu']['E62_minus_E29'])
        report['rows'].append(row)
    report['status']='ROUTER_REPLAY_EXACT' if all(p['logits_vs_capture']['exact'] and p['ids_exact']
        for r in report['rows'] for p in r['paths'].values()) else 'ROUTER_REPLAY_REVIEW_REQUIRED'
    # Identify history evidence at the first two layers; do not claim KV hashes
    # were captured when only decoder states/attention outputs were captured.
    history=[]
    for call in range(1,56):
        entry=dict(call=call)
        for layer,stage in ((1,'decoder_output_hidden'),(1,'decoder_output_residual'),
                            (2,'decoder_input_hidden'),(2,'attention_output')):
            n=load('npu',layer,stage,call);h=load('hybrid',layer,stage,call)
            entry[f'L{layer}_{stage}']=metrics(h,n)
        history.append(entry)
    report['history_boundary_diagnostics']=history
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='history_boundary_diagnostics'}))


if __name__=='__main__':main()
