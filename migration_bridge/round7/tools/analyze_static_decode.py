"""Separate prefill/decode using original frozen static archives, no A3 rerun.

Classification comes from exact prompt/token-history association, not str(IntEnum).
No changes to frequency mask, corpus, quality policy or production arithmetic.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

from associate_forwards import associate
from dynamic_route_summary import summarize_decode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_prefill(rows, keys):
    groups = {}
    total = dict(assignments=0, npu_hits=0, cpu_hits=0, total_weight=0., cpu_weight=0.)
    per_layer = {}
    for row in rows:
        context = row['forward_context'];key = (context['epoch'], context['forward_id'])
        if key not in keys:continue
        layers = groups.setdefault(key, {})
        if row['layer'] in layers:raise ValueError('DUPLICATE_PREFILL_LAYER')
        layers[row['layer']] = row
        for field in total:total[field] += row[field]
        layer = per_layer.setdefault(str(row['layer']), dict(assignments=0, npu_hits=0, cpu_hits=0))
        for field in layer:layer[field] += row[field]
    if set(groups) != keys:raise ValueError('MISSING_PREFILL_FORWARD')
    tokens = 0;fully_npu = 0
    for layers in groups.values():
        if set(layers) != set(range(1, 27)):raise ValueError('PREFILL_LAYER_COMPLETENESS')
        sizes = {r['tokens'] for r in layers.values()}
        if len(sizes) != 1:raise ValueError('PREFILL_ROW_SHAPE')
        count = next(iter(sizes));tokens += count
        for row in layers.values():
            if row['assignments'] != 6*count or len(row['fully_npu_row_flags']) != count:
                raise ValueError('PREFILL_ROUTE_SHAPE')
        fully_npu += sum(all(r['fully_npu_row_flags'][i] for r in layers.values()) for i in range(count))
    if total['assignments'] != tokens*26*6:raise ValueError('PREFILL_ROUTE_COMPLETENESS')
    return dict(**total, npu_hit_rate=total['npu_hits']/total['assignments'],
        cpu_route_rate=total['cpu_hits']/total['assignments'],
        cpu_weight_mass=total['cpu_weight']/total['total_weight'], per_layer=per_layer,
        model_token_rows=tokens, fully_npu_model_token_rows=fully_npu,
        model_token_touches_cpu_fraction=1-fully_npu/tokens,
        scope='ATTRIBUTED_STATIC_HYBRID_PREFILL_ONLY_NOT_DYNAMIC_FULL_DEVICE_PREFILL')


def from_archive(path, name, corpus):
    with tarfile.open(path) as archive:
        roots=[base for base in ('','runtime-v2/') if base+f'{name}-holdout-summary.json' in archive.getnames()]
        if len(roots)!=1:raise ValueError('AMBIGUOUS_OR_MISSING_ARCHIVE_LAYOUT')
        base=roots[0];prefix=base+f'{name}-holdout-v1/'
        def load(relative):return json.load(archive.extractfile(relative))
        def lines(relative):return [json.loads(line) for line in archive.extractfile(relative)]
        summary = load(base+f'{name}-holdout-summary.json')
        lifecycle = load(prefix+'lifecycle.json')
        if summary['status'] != 'PASS' or lifecycle['status'] != 'COMPLETE' or not lifecycle['reaping']['complete']:
            raise ValueError('STATIC_CAPTURE_NOT_CLOSED')
        requests = lines(prefix+'requests.jsonl')
        if len(requests) != len(corpus):raise ValueError('REQUEST_COUNT')
        operands = []
        for i, (request, row) in enumerate(zip(requests, corpus)):
            answer = request['response'];rid = f'{name}-holdout-{i:03d}'
            if request['id'] != rid or request['prompt_id'] != row['id'] or answer['meta_info']['id'] != rid:
                raise ValueError('FROZEN_REQUEST_IDENTITY')
            if request['input_tokens'] != len(row['input_ids']) or answer['meta_info']['prompt_tokens'] != len(row['input_ids']):
                raise ValueError('FROZEN_INPUT_LENGTH')
            if len(answer['output_ids']) != 16:raise ValueError('GENERATED_BUDGET')
            operands.append(dict(input_ids=row['input_ids'], output_ids=answer['output_ids']))
        metadata = [m for m in lines(prefix+'forward-metadata.jsonl') if m['epoch'] == 1]
        attribution = associate(metadata, operands)
        for key, original in summary['forward_attribution'].items():
            if attribution[key] != original:raise ValueError('REASSOCIATION_DISAGREES_WITH_ORIGINAL')
        def keys(indices):return {(metadata[i]['epoch'], metadata[i]['forward_id']) for i in indices}
        all_keys = keys(range(len(metadata)))
        if len(all_keys) != len(metadata):raise ValueError('DUPLICATE_RAW_FORWARD_ID')
        routing = lines(prefix+'routing.jsonl')
        for row in routing:
            context = row['forward_context']
            if (context.get('epoch'), context.get('forward_id')) not in all_keys:
                raise ValueError('ROUTING_OUTSIDE_ATTRIBUTED_CAPTURE')
        decode = summarize_decode(routing, keys(attribution['kept_indices']),
            decode_keys=keys(attribution['decode_indices']), include_counterfactual=False)
        prefill = summarize_prefill(routing, keys(attribution['prefill_indices']))
        if decode['model_token_rows'] != len(corpus)*15 or prefill['model_token_rows'] != sum(len(r['input_ids']) for r in corpus):
            raise ValueError('STAGE_TOKEN_COMPLETENESS')
        for field in ('assignments', 'npu_hits', 'cpu_hits', 'total_weight', 'cpu_weight'):
            if not math.isclose(prefill[field]+decode[field], summary['routing'][field], rel_tol=1e-12, abs_tol=1e-6):
                raise ValueError('STAGE_SUM_DISAGREES_WITH_ORIGINAL')
        return dict(prefill=prefill, decode=decode, original_combined=summary['routing'],
            attribution_counts={k:len(attribution[k]) for k in ('prefill_indices', 'decode_indices', 'unused_final_token_indices')},
            unknown_passes=attribution['unknown_passes'], archive_sha256=sha(path),
            archive_root=base,
            classification_source='EXACT_FROZEN_PROMPT_AND_RETURNED_TOKEN_HISTORY_NOT_NUMERIC_MODE_GUESS')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--uniform-archive', type=Path, required=True)
    p.add_argument('--frequency-archive', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    freeze = json.loads((a.corpus.parent/'CORPUS_FREEZE.json').read_text())
    if sha(a.corpus) != freeze['corpora']['HOLDOUT']['sha256']:raise ValueError('FROZEN_HOLDOUT_HASH')
    corpus = [json.loads(line) for line in a.corpus.read_text().splitlines()]
    if len(corpus) != 128:raise ValueError('FROZEN_HOLDOUT_COUNT')
    profiles = {name:from_archive(path, name, corpus) for name,path in [('uniform', a.uniform_archive), ('frequency', a.frequency_archive)]}
    comparison = {}
    for stage in ('prefill', 'decode'):
        u,f = (profiles[name][stage] for name in ('uniform', 'frequency'))
        comparison[stage] = dict(npu_hit_gain_pp=100*(f['npu_hit_rate']-u['npu_hit_rate']),
            cpu_assignment_relative_reduction_percent=100*(1-f['cpu_hits']/u['cpu_hits']),
            cpu_weight_mass_relative_reduction_percent=100*(1-f['cpu_weight_mass']/u['cpu_weight_mass']))
    report = dict(status='ANALYTIC_DERIVED_FROM_CLOSED_A3_CAPTURES', holdout_sha256=sha(a.corpus),
        profiles=profiles, comparison=comparison, formal_performance=False,
        static_selection='PENDING_COMPLETE_OBSERVER_OFF_PERFORMANCE')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'static-stage-routing.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':main()
