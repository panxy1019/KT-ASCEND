"""Validate complete output coverage without promoting an incomplete B0 gate."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import math


def validate(d):
    expected={(l,e,c) for l in [1,9,17,26] for e in range(1,64,2) for c in [2,33,55,1]}
    rows=d['rows']
    keys=[(r['layer'],r['expert'],r['call']) for r in rows]
    if len(keys)!=512 or set(keys)!=expected:raise ValueError('COVERAGE_OR_DUPLICATE')
    if not all(r['deterministic'] and r['finite'] and r['local_safety_pass'] for r in rows):
        raise ValueError('LOCAL_SAFETY')
    for r in rows:
        for name in ['production_vs_npu','production_vs_canonical_fp32']:
            value=r['comparisons'][name]['relative_l2']
            if not math.isfinite(value) or value<0 or value>.01:raise ValueError('METRIC_SAFETY_NOT_JUST_FLAG')
        if len(r['production_hashes'])!=10 or len(set(r['production_hashes']))!=1:raise ValueError('REPEAT_PROOF')
        if len(r['npu_hashes'])!=10 or len(set(r['npu_hashes']))!=1:raise ValueError('REPEAT_PROOF')
    def summary(values):return dict(min=min(values),median=statistics.median(values),max=max(values))
    return dict(status='OUTPUT_SCREEN_VERIFIED_NOT_MODEL_QUALIFICATION',expert_instances=128,
        decode_cases=384,prefill_cases=128,prefill_qlens=sorted({r['qlen'] for r in rows if r['call']==1}),
        metrics={k:summary([r['comparisons'][k]['relative_l2'] for r in rows]) for k in rows[0]['comparisons']},
        relative_error_ratio=summary([r['relative_error_ratio'] for r in rows]),
        stage_cases=sum('stage_comparisons' in r for r in rows),
        production_observer_parity_cases=sum(r.get('production_observer_parity',False) for r in rows),
        B0='IN_PROGRESS',QD6='NOT_RUN',HD6='NOT_RUN',promotion_allowed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.input.read_bytes();result=validate(json.loads(raw))
    result['input_sha256']=hashlib.sha256(raw).hexdigest()
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
