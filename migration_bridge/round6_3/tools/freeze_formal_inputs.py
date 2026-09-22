"""Pinned tokenization of reviewed sources after exposure-only N selection.

Creates input-only QD3/HD3/FD3 freezes; never generates held-out histories.
Historical token audit and statistical preregistration remain separate gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'pilot-result', 'size-plan', 'review', 'tokenizer', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    assert json.loads(args.pilot_result.read_text())['status'] == 'EXPOSURE_PILOT_COMPLETE'
    plan = json.loads(args.size_plan.read_text())
    n = plan['selected_prompts']
    assert n in (128, 256, 512), 'PILOT_CANNOT_SUPPORT_FORMAL_SIZE'
    review = json.loads(args.review.read_text())
    assert review['status'] == 'INPUT_ONLY_SEMANTIC_REVIEW_COMPLETE'
    assert sha(args.tokenizer) == '41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669'
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    outputs, seen = {}, set()
    for name in ('qd3', 'hd3', 'fd3'):
        path = args.sources / (name + '_authoring_draft.json')
        assert review['source_hashes'][name] == sha(path), 'SOURCE_REVIEW_DRIFT'
        source = json.loads(path.read_text())
        expected = (dict(english=2, chinese=2, math=2, structured=1, code=1) if name == 'fd3' else
                    dict(english=3*n//16, chinese=3*n//16, math=3*n//16, structured=3*n//16, code=4*n//16))
        assert {k:len(v) for k,v in source['categories'].items()} == expected, 'AUTHORING_NOT_READY_FOR_N'
        rows = []
        for category, texts in source['categories'].items():
            for i, text in enumerate(texts):
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                assert ids and len(ids) + 64 <= 512, 'CONTEXT_LIMIT'
                assert tuple(ids) not in seen, 'CROSS_CORPUS_TOKEN_DUPLICATE'
                seen.add(tuple(ids))
                rows.append(dict(rid=f'{name}-{category}-{i:02d}', category=category, text=text, input_ids=ids))
        random.Random(6303).shuffle(rows)
        value = dict(version='r63-formal-input-v1', corpus=name.upper(), token_budget=64,
                     order_seed=6303, generation_seed=0, requests=rows, categories=expected,
                     source_sha256=sha(path), review_sha256=sha(args.review),
                     size_plan_sha256=sha(args.size_plan), tokenizer_sha256=sha(args.tokenizer),
                     history_rule='one natural defer0 history per prompt, frozen before replay; HD3 only after candidate-bound runtime PASS')
        value['sha256'] = canonical(value)
        outputs[name] = value
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in outputs.items():
        (args.output / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='INPUT_ONLY_FROZEN_PENDING_TOKEN_AUDIT',
                         corpora={name:v['sha256'] for name,v in outputs.items()})))


if __name__ == '__main__':
    main()
