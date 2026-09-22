"""Text and numeric-template overlap audit before formal corpus freeze.

Similarity is a review aid, not a proof of semantic independence. Never reads
inference outputs or filters prompts by routing/numerical performance.
"""
import argparse
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def words(text):
    return set(re.findall(r'[a-z]+|[\u4e00-\u9fff]', norm(text)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    historical = {}
    for path in sorted(args.root.rglob('*.json')):
        if 'round6_3' in path.parts or not {'corpora', 'fixtures'} & set(path.parts):
            continue
        for text in strings(json.loads(path.read_text())):
            if 20 <= len(text) <= 4000:
                historical.setdefault(norm(text), dict(text=text, source=str(path)))
    here = Path(__file__).resolve().parents[1] / 'corpora'
    pilot = json.loads((here / 'pilot_source.json').read_text())
    for row in pilot['prompts']:
        historical[norm(row['text'])] = dict(text=row['text'], source='R63_PILOT')
    drafts = []
    source_hashes = {}
    for name in ('qd3', 'hd3', 'fd3'):
        path = here / (name + '_authoring_draft.json')
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        source = json.loads(path.read_text())
        for category, texts in source['categories'].items():
            for i, text in enumerate(texts):
                drafts.append(dict(id=f'{name}-{category}-{i:02d}', text=text, source=name))
    pool = list(historical.values()) + drafts
    pool_words = [words(r['text']) for r in pool]
    duplicates, reviews = [], []
    for row in drafts:
        tokens = words(row['text'])
        ranks = []
        for candidate, other in zip(pool, pool_words):
            if candidate is row:
                continue
            if norm(candidate['text']) == norm(row['text']):
                duplicates.append(dict(id=row['id'], other=candidate))
            jaccard = len(tokens & other) / max(1, len(tokens | other))
            ranks.append((jaccard, candidate))
        candidates = sorted(ranks, key=lambda x:x[0], reverse=True)[:5]
        close = []
        for overlap, candidate in candidates:
            a, b = [re.sub(r'\d+(?:\.\d+)?', '<n>', norm(t)) for t in (row['text'], candidate['text'])]
            ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
            if ratio >= .55 or overlap >= .5:
                close.append(dict(similarity=ratio, lexical_overlap=overlap, other=candidate))
        if close:
            reviews.append(dict(id=row['id'], text=row['text'], matches=close))
    result = dict(status='FAIL_EXACT_DUPLICATE' if duplicates else 'EXACT_DEDUPE_PASS_SEMANTIC_REVIEW_REQUIRED',
                  historical_unique=len(historical), draft_count=len(drafts), source_hashes=source_hashes,
                  duplicates=duplicates, reviews=reviews,
                  numerical_outputs_read=False, limitation='lexical review aid, no semantic independence proof')
    with args.output.open('x') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status=result['status'], duplicates=len(duplicates), reviews=len(reviews))))


if __name__ == '__main__':
    main()
