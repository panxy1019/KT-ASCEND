"""Run Q32 on the already frozen Q64 Qwen CMMLU manifest and scorer."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import requests

from qwen_cmmlu_reference import SOURCE_SHA, score


ROOT = Path('/workspace/round8')
MANIFEST_SHA = '365f557c0746db1e02ddee4ec66c7afb5e72eeb17e8dd8e0a47d1ed2b7af1b9d'
Q32_PROFILE_SHA = 'b358399e6405d60b81d8231529b1ec6024ff7138cddab2e0cbac3ccf0b25f486'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-commit', required=True)
    args = parser.parse_args()
    assert len(args.frozen_commit) == 40 and all(c in '0123456789abcdef' for c in args.frozen_commit)

    reference = ROOT / 'qwen-cmmlu-reference-v1'
    manifest_path = reference / 'manifest.json'
    assert sha(manifest_path) == MANIFEST_SHA
    manifest = json.loads(manifest_path.read_text())
    assert manifest['source_sha256'] == SOURCE_SHA
    assert len(manifest['rows']) == 256
    assert sha(ROOT / 'model-sample/tokenizer.json') == manifest['tokenizer_sha256']
    assert sha(ROOT / 'model-sample/tokenizer_config.json') == manifest['tokenizer_config_sha256']
    assert sha(ROOT / 'q32_profile.json') == Q32_PROFILE_SHA
    assert json.loads((ROOT / 'q32-compact-v1/same-summary.json').read_text())['status'] == 'PASS'
    assert json.loads((ROOT / 'q32-compact-v1/restart-summary.json').read_text())['status'] == 'PASS'
    assert json.loads((ROOT / 'q32-generation-screen-v1/summary.json').read_text())['status'] == 'PASS'

    reference_summary = json.loads((reference / 'summary.json').read_text())
    assert reference_summary['correct'] == 222 and reference_summary['count'] == 256
    assert reference_summary['invalid'] == 0
    assert reference_summary['manifest_sha256'] == MANIFEST_SHA

    out = ROOT / 'qwen-cmmlu-q32-v1'
    out.mkdir(exist_ok=False)
    write(out / 'entry.json', dict(status='FROZEN_BEFORE_Q32_QUALITY',
                                   local_commit=args.frozen_commit,
                                   manifest_sha256=MANIFEST_SHA,
                                   q32_profile_sha256=Q32_PROFILE_SHA,
                                   q64_reference_correct=222,
                                   q32_min_correct=210))
    correct = 0
    predictions = []
    with (out / 'raw.jsonl').open('x') as raw:
        for index, row in enumerate(manifest['rows']):
            payload = dict(rid='qwen-q32-' + row['id'], input_ids=row['input_ids'],
                           sampling_params=dict(temperature=0, max_new_tokens=1,
                                                ignore_eos=False, stop=row['stop']),
                           return_logprob=True, logprob_start_len=-1,
                           top_logprobs_num=0, token_ids_logprob=row['candidate_ids'])
            response = requests.post('http://127.0.0.1:18400/generate', json=payload, timeout=3600)
            data = response.json()
            raw.write(json.dumps(dict(id=row['id'], payload=payload,
                                      http_status=response.status_code,
                                      response=data), ensure_ascii=False) + '\n')
            raw.flush()
            response.raise_for_status()
            assert data['meta_info']['id'] == payload['rid']
            positions = data['meta_info']['output_token_ids_logprobs']
            assert len(positions) == 1
            predicted = score(positions[0], row['candidate_ids'])
            predictions.append(predicted)
            correct += int(predicted == row['answer'])
            write(out / 'progress.json', dict(completed=index + 1, total=256,
                                              correct=correct, invalid=0))
            print(f'{index + 1}/256 correct={correct}', flush=True)
    write(out / 'summary.json', dict(status='Q32_SCORED_PENDING_INDEPENDENT_AUDIT',
                                     correct=correct, count=256, invalid=0,
                                     reference_correct=222, observed_delta_pp=(correct-222)/256*100,
                                     observed_gate_min_correct=210,
                                     manifest_sha256=MANIFEST_SHA,
                                     entry_sha256=sha(out / 'entry.json'),
                                     answer_distribution={c: predictions.count(c) for c in 'ABCD'},
                                     formal_performance=False))


if __name__ == '__main__':
    main()
