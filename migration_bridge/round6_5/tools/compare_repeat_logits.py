"""Byte/value repeat check of all matched-history logits, not free generation."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True)
    p.add_argument('--repeat',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();reports=[json.loads((root/'capture-result.json').read_text()) for root in (a.original,a.repeat)]
    if any(r['status']!='CAPTURE_INTEGRITY_PASS' for r in reports):raise ValueError('CAPTURE_INTEGRITY')
    rows=[]
    for first,second in zip(reports[0]['rows'],reports[1]['rows'],strict=True):
        for key in ('rid','input_ids','baseline_output_ids'):
            if first[key]!=second[key]:raise ValueError('REPEAT_HISTORY_DRIFT')
        rid=first['rid'];n=torch.load(a.original/(rid+'.logits.pt'),weights_only=True)
        h=torch.load(a.repeat/(rid+'.logits.pt'),weights_only=True)
        def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        rows.append(dict(rid=rid,exact=torch.equal(n,h),original_sha256=sha(n),repeat_sha256=sha(h),
            different_positions=torch.where((n!=h).any(dim=1))[0].tolist(),max_abs=float((n.float()-h.float()).abs().max())))
    result=dict(status='MATCHED_HISTORY_LOGITS_REPEAT_EXACT' if all(r['exact'] for r in rows) else 'REPEAT_MISMATCH',
        scope='FORCED_HISTORY_LOGITS_NOT_FREE_RUNNING_TOKENS',rows=rows)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
