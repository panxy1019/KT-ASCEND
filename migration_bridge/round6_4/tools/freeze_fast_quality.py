"""Freeze the user-approved 32/256 engineering screen without inspecting outputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

SOURCE_SHA = "779d1318137ee1a3573bafe20e265f2aa31170660c08b49ada0507fdb179a3ba"
SEED = 6404


def rank(row):
    return hashlib.sha256(f"{SEED}:{row['id']}".encode()).hexdigest()


def select(rows):
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('DUPLICATE_ID')
    gsm = sorted((r for r in rows if r['benchmark'] == 'gsm8k'), key=rank)[:32]
    groups = defaultdict(list)
    for r in rows:
        if r['benchmark'] == 'cmmlu':
            groups[r['subject']].append(r)
    if len(groups) != 67 or len(gsm) != 32:
        raise ValueError('SOURCE_COVERAGE')
    # Cover every subject, then proportional largest-remainder allocation.
    total = sum(len(v) - 1 for v in groups.values())
    remaining = 256 - len(groups)
    counts = {s: 1 + remaining * (len(v)-1) // total for s,v in groups.items()}
    order = sorted(groups, key=lambda s: (-(remaining*(len(groups[s])-1) % total), s))
    for s in order[:256-sum(counts.values())]:
        counts[s] += 1
    cm = [r for s in sorted(groups) for r in sorted(groups[s],key=rank)[:counts[s]]]
    if len(cm) != 256:
        raise ValueError('INSUFFICIENT_SUBJECT_ROWS')
    return gsm + cm


def observed_pass(off_correct, on_correct, count):
    if count not in (32, 256) or not 0 <= off_correct <= count or not 0 <= on_correct <= count:
        raise ValueError('INVALID_COUNTS')
    return 100 * (off_correct-on_correct) <= 2 * count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if hashlib.sha256(a.source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('SOURCE_DRIFT')
    selected = select(json.loads(a.source.read_text())['rows'])
    a.output.mkdir(parents=True, exist_ok=False)
    data = (json.dumps({'rows': selected}, ensure_ascii=False, sort_keys=True, separators=(',',':'))+'\n').encode()
    (a.output/'primary.json').write_bytes(data)
    manifest = dict(version='round6.4-fast-quality-v1', status='INPUT_AND_SCREEN_RULE_FROZEN',
        primary_execution_allowed=False, execution_blocker='bind reviewed dev/runtime/source/evaluator evidence before launch',
        source_primary_sha256=SOURCE_SHA, primary_input_sha256=hashlib.sha256(data).hexdigest(),
        seed=SEED, counts=dict(Counter(r['benchmark'] for r in selected)),
        cmmlu_subject_counts=dict(Counter(r['subject'] for r in selected if r['benchmark']=='cmmlu')),
        selected_ids=[r['id'] for r in selected], selection='SHA256(seed:id) rank; CMMLU all67 subjects plus largest-remainder proportional allocation within frozen2048 pool',
        quality_claim='FROZEN_SUBSET_ENGINEERING_SCREEN_ONLY', margin_absolute=.02,
        pass_rule='each benchmark observed ON accuracy minus OFF accuracy >= -0.02',
        maximum_net_correct_loss={'gsm8k':0,'cmmlu':5}, confidence_interval_is_gate=False,
        population_noninferiority_claim=False, no_adaptive_sample_expansion=True,
        batch=1, prompt_and_token_ids='unchanged subset of draft-v2', invalid_and_truncated_are_wrong=True,
        evaluator_sha256=hashlib.sha256(Path(__file__).with_name('quality_answers_v2.py').read_bytes()).hexdigest(),
        baseline_floor_and_truncation_disclosure_required=True)
    (a.output/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(primary_sha256=manifest['primary_input_sha256'], manifest_sha256=hashlib.sha256((a.output/'MANIFEST.json').read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
