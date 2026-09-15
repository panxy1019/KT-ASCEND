"""Read-only QD3 rejection analysis; never derive or replace a contract."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def analyze(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    assert report['status'] == 'REJECTED'
    bound = report['reserves']['contraction']['threshold']
    failures = []
    for row in report['rows']:
        if not row['failures']:
            continue
        ids = row['candidate_ids']
        b, d = row['baseline_candidate_logits'], row['deferred_candidate_logits']
        pairs = []
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i != j and b[i]-b[j] > bound and d[i]-d[j] <= 0:
                    pairs.append(dict(tokens=[ids[i], ids[j]], baseline_margin=b[i]-b[j],
                        deferred_margin=d[i]-d[j], contraction=(b[i]-b[j])-(d[i]-d[j]),
                        includes_anchor=row['baseline_token'] in (ids[i], ids[j])))
        failures.append(dict(prompt_id=row['prompt_id'], token_index=row['token_index'],
            reasons=row['failures'], baseline_token=row['baseline_token'],
            deferred_token=row['deferred_token'], exposure=row['deferred_exposure'],
            anchor_contraction=row['contraction'], anchor_expansion=row['beneficial_expansion'],
            pairs=pairs, baseline_capture=row['baseline_capture'],
            deferred_capture=row['deferred_capture']))
    return dict(status='REJECTED_PRESERVED', source_raw_sha256=hashlib.sha256(raw).hexdigest(),
        source_canonical_sha256=report['sha256'], threshold_unchanged=bound,
        failure_counts=dict(Counter(f for r in report['rows'] for f in r['failures'])),
        positions=len(report['rows']), exposed=report['exposure_positions'],
        stable_positions=sum(r['stable'] for r in report['rows']),
        stable_top1_mismatches=sum(r['stable'] and r['baseline_token'] != r['deferred_token'] for r in report['rows']),
        failures=failures, not_claimed='No production root cause or repaired contract established')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('report', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.report)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
