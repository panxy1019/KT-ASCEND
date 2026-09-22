"""Small deterministic contract helpers; no accelerator/network import."""
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def validate_profile(data):
    n = data['resident']
    if n not in (64, 32, 16, 8) or data['selected_layers'] != list(range(1, 27)):
        raise ValueError('PROFILE_SCOPE')
    if data['deferred'] != 0 or data['dynamic'] or len(data['rows']) != 26:
        raise ValueError('FEATURE_OR_LAYER_DRIFT')
    for layer, row in enumerate(data['rows'], 1):
        if row['layer'] != layer or row['npu_ids'] != list(range(0, 64, 64 // n)):
            raise ValueError('NESTED_MASK_DRIFT')
        if row['cpu_ids'] != sorted(set(range(64)) - set(row['npu_ids'])):
            raise ValueError('OWNERSHIP_DRIFT')
    raw = {k: v for k, v in data.items() if k not in ('canonical_sha256', 'pt_sha256')}
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != data['canonical_sha256']:
        raise ValueError('CANONICAL_PROFILE_DRIFT')


def validate_answer(answer, rid, input_ids, budget, exact_budget=False):
    meta, ids = answer['meta_info'], answer['output_ids']
    lp = meta['output_token_logprobs']
    if meta['id'] != rid or meta['prompt_tokens'] != len(input_ids):
        raise ValueError('REQUEST_CORRELATION')
    if not 0 < len(ids) <= budget or (exact_budget and len(ids) != budget):
        raise ValueError('TOKEN_BUDGET')
    if len(lp) != len(ids) or [x[1] for x in lp] != ids:
        raise ValueError('LOGPROB_TOKEN_IDENTITY')
    if not all(x[0] is not None and math.isfinite(float(x[0])) for x in lp):
        raise ValueError('NONFINITE_OUTPUT')
    if meta['finish_reason']['type'] not in ('stop', 'length'):
        raise ValueError('REQUEST_FINISH')
    return ids


def score_letters(values, candidates):
    if len(values) != 4 or len(set(candidates)) != 4 or {x[1] for x in values} != set(candidates):
        raise ValueError('CANDIDATE_IDENTITY')
    probabilities = {x[1]: float(x[0]) for x in values}
    if not all(math.isfinite(x) for x in probabilities.values()):
        raise ValueError('NONFINITE_CANDIDATE')
    index = max(range(4), key=lambda i: probabilities[candidates[i]])
    return 'ABCD'[index], [probabilities[x] for x in candidates]


def product_gate(correct, baseline, count=256):
    if count != 256 or not 0 <= correct <= count or not 0 <= baseline <= count:
        raise ValueError('QUALITY_COUNT')
    delta = 100 * (correct - baseline) / count
    return dict(delta_pp=delta, pass_5pp=delta >= -5.0)
