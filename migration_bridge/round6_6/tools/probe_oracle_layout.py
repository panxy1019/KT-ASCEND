"""Check the isolated one-slot oracle against actual 64-slot GMM layout."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from safetensors import safe_open
from three_way_probe import digest, metrics, npu_format_cast, UnquantizedFusedMoEMethod


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    baseline=json.loads((a.baseline/'three_way.json').read_text())
    index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    method=NS(with_bias=False,moe_runner_config=NS(activation='silu'),_kt_return_fp32_contribution=True)
    report=dict(status='RUNNING',rows=[],production_change=False)
    try:
        for layer in [1,9,17,26]:
            gateup=[];down=[]
            for e in range(64):
                w={}
                for name in ['gate','up','down']:
                    key=f'model.layers.{layer}.mlp.experts.{e}.{name}_proj.weight'
                    with safe_open(a.model/index[key],framework='pt',device='cpu') as sf:w[name]=sf.get_tensor(key).clone()
                gateup.append(torch.cat([w['gate'],w['up']]));down.append(w['down'])
            full=NS(w13_weight=npu_format_cast(torch.stack(gateup).to('npu').transpose(1,2)),
                    w2_weight=npu_format_cast(torch.stack(down).to('npu').transpose(1,2)),top_k=6)
            del gateup,down,w
            for r in [r for r in baseline['rows'] if r['layer']==layer]:
                e,c=r['expert'],r['call']
                data=torch.load(a.baseline/f'L{layer:02d}-E{e:02d}-C{c:04d}.pt',weights_only=False,map_location='cpu')
                x=data['hidden'];ids=torch.full((len(x),6),-1,dtype=torch.int32);ids[:,0]=e
                weights=torch.zeros((len(x),6));weights[:,0]=1
                result=UnquantizedFusedMoEMethod.forward_npu(method,full,NS(hidden_states=x.to('npu'),
                    topk_output=(weights.to('npu'),ids.to('npu'),None))).hidden_states.cpu()
                m=metrics(result,data['npu'])
                row=dict(layer=layer,expert=e,call=c,input_sha256=digest(x),comparison=m)
                report['rows'].append(row)
                if not m['finite'] or not m['exact']:raise ValueError('ORACLE_LAYOUT_DIFFERENCE_REQUIRES_REVIEW')
            del full;torch.npu.empty_cache()
            print(json.dumps(dict(layer=layer,exact_rows=len(report['rows']))),flush=True)
        report['status']='NPU_ORACLE_ONE_VS_64_LAYOUT_EXACT'
    except BaseException as exc:report.update(status='BLOCKED',error=repr(exc));raise
    finally:(a.output/'layout.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
