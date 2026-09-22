"""Independent route-only count; no torch import or numerical logit access."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest-gzip', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('refuse exposure evidence overwrite')
    raw = gzip.decompress(args.manifest_gzip.read_bytes())
    manifest = json.loads(raw)
    rows = manifest['rows']
    actual = Counter()
    categories = Counter()
    experts = Counter()
    prompts = set()
    mismatch = []
    for row in rows:
        current = row['deferred_event_count_current_row']
        prefix = row['deferred_event_count_prefix']
        exposed = current + prefix > 0
        actual['positions'] += 1
        actual['exposed_positions'] += exposed
        actual['current_exposed_positions'] += current > 0
        actual['prefix_exposed_positions'] += prefix > 0
        actual['current_route_events'] += current
        actual['prefix_route_events'] += prefix
        if exposed:
            categories[row['category']] += 1
            prompts.add(row['prompt_id'])
        if exposed != (row['deferred_exposure'] != 'NO_EXPOSURE'):
            mismatch.append([row['prompt_id'], row['token_index']])
        for route in row['deferred_route_records']:
            experts[str(route['expert_logical_id'])] += 1
    report = dict(status='INSUFFICIENT_EXPOSURE' if actual['exposed_positions'] < 500 else 'MINIMUM_MET',
        required_minimum=500, counts=dict(actual), exposed_prompt_count=len(prompts),
        exposed_positions_by_category=dict(categories), route_records_by_expert=dict(experts),
        label_mismatches=mismatch, raw_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        numerical_metrics_read=False, source='archived manifest route metadata only')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
