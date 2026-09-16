"""Offline first-failure raw-logit audit; never modifies the frozen contract."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairwise',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--hybrid',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();gate=json.loads(a.pairwise.read_text());bad=gate['first_failure']
    if bad is None:raise ValueError('NO_FAILURE')
    rid=bad['rid'];index=bad['token_index'];top1=bad['baseline_top1']
    n=torch.load(a.baseline/(rid+'.logits.pt'),weights_only=True).float()
    h=torch.load(a.hybrid/(rid+'.logits.pt'),weights_only=True).float()
    u=torch.unique(torch.cat((n[index].topk(32).indices,h[index].topk(32).indices,torch.tensor([top1]))))
    details=[]
    for expert in u.tolist():
        mn=float(n[index,top1]-n[index,expert]);mh=float(h[index,top1]-h[index,expert])
        details.append(dict(token=expert,npu_logit=float(n[index,expert]),hybrid_logit=float(h[index,expert]),
            npu_margin=mn,hybrid_margin=mh,pairwise_distortion=abs(mn-mh)))
    report=dict(status='FIRST_FAILURE_RAW_AUDIT',first_failure=bad,
        failed_positions=[r for r in gate['rows'] if not r['pass_gate']],
        candidates=sorted(details,key=lambda x:-x['pairwise_distortion']),nearby=[],captures={})
    for i in range(max(0,index-3),min(64,index+4)):
        delta=h[i]-n[i]
        report['nearby'].append(dict(token_index=i,max_abs=float(delta.abs().max()),
            mean_abs=float(delta.abs().mean()),relative_l2=float(delta.norm()/n[i].norm()),
            npu_top1=int(n[i].argmax()),hybrid_top1_cpu_tie_diagnostic=int(h[i].argmax())))
    for name,root in [('npu',a.baseline),('hybrid',a.hybrid)]:
        capture=json.loads((root/'capture-result.json').read_text())
        row=next(r for r in capture['rows'] if r['rid']==rid)
        file=Path(row['steps'][index]['file']);data=torch.load(file,weights_only=False,map_location='cpu')
        report['captures'][name]=dict(status=capture['status'],forward_count=capture['forward_count'],
            input_manifest_sha256=capture['inputs_sha256'],file=str(file),file_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            transaction=data['capture.transaction'],batch={k:data[k].tolist() for k in
                ('capture.forward_batch_info.input_ids','capture.forward_batch_info.positions','capture.forward_batch_info.seq_lens')})
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('first_failure','nearby')}))


if __name__=='__main__':main()
