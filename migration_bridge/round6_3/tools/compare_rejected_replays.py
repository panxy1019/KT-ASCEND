"""Compare frozen failed histories; no threshold fitting or qualification."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    torch.set_num_threads(1)
    root=args.root
    diagnostic=root/'r63-rejected-replay-v1'
    assert json.loads((diagnostic/'result.json').read_text())['status']=='DIAGNOSTIC_CAPTURE_COMPLETE_PENDING_COMPARISON'
    def rows(path):
        return {r['rid']:r for r in json.loads(path.read_text())['rows']}
    runs=[rows(root/'r63-qd3-v1-force1-session/capture/result.json')]
    for mode in ('production','explicit'):
        assert json.loads((diagnostic/mode/'final-integrity.json').read_text())['status']=='PASS'
        runs.append(rows(diagnostic/mode/'capture/result.json'))
    summaries=[]
    for left,right,label in ((0,1,'original_vs_production_replay'),(1,2,'production_vs_explicit')):
        checks=exact=logits_exact=0
        mismatches=[]
        differing_keys=set()
        for rid in ('qd3-math-11','qd3-math-19'):
            a,b=runs[left][rid],runs[right][rid]
            assert a['input_ids']==b['input_ids'] and a['baseline_output_ids']==b['baseline_output_ids']
            assert len(a['steps'])==len(b['steps'])==65
            for s,t in zip(a['steps'],b['steps']):
                assert s['step']==t['step']
                tensors=[torch.load(x['file'],map_location='cpu',weights_only=False,mmap=True) for x in (s,t)]
                keys=[{k for k,v in d.items() if isinstance(v,torch.Tensor)} for d in tensors]
                differing_keys.update(keys[0]^keys[1])
                for key in sorted(keys[0]&keys[1]):
                    x,y=(d[key] for d in tensors)
                    checks+=1
                    same=x.shape==y.shape and x.dtype==y.dtype and torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
                    exact+=int(same)
                    if key=='capture.next_token_logits':
                        logits_exact+=int(same)
                    if not same:
                        mismatches.append(dict(rid=rid,step=s['step'],key=key,
                            max_abs=float((x.double()-y.double()).abs().max()) if x.shape==y.shape else None))
        summaries.append(dict(comparison=label,common_tensor_checks=checks,byte_exact=exact,
            full_logit_forwards=130,full_logit_exact=logits_exact,mismatches=mismatches,
            unpaired_diagnostic_keys=sorted(differing_keys)))
    result=dict(status='DIAGNOSTIC_COMPARISON_COMPLETE',qualification=False,comparisons=summaries,
        original_qd3_status='REJECTED',tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    main()
