"""Audit all frozen quality answer pairs without selecting or replacing samples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--parser-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(a.inputs) == '9618d054cf765e1d6328d1581435ec97f9f4271715b794dc07e0a9889a483fcb'
    assert sha(a.parser_dir/'quality_answers_v2.py') == '7d394487111e227f1f5b709d8199564fe5688b91363628c2b4fa4ef4ab743fbc'
    assert sha(a.parser_dir/'quality_answers.py') == '4cae28e44dba632d4a4bf2e4d2c5cd6138abc77e6b5401f5e682bff451d882ba'
    sys.path.insert(0, str(a.parser_dir))
    from quality_answers_v2 import score
    frozen = json.loads(a.inputs.read_text())['rows']
    lanes = {}
    for mode in ('npu', 'hybrid'):
        folder = a.root/('quality_'+mode)
        report = json.loads((folder/'quality-report.json').read_text())
        assert report['status'] == 'QUALITY_SCREEN_COLLECTION_COMPLETE'
        raw = [json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
        assert len(raw) == len(report['rows']) == len(frozen) == 288
        lanes[mode] = []
        for expected, observed, saved in zip(frozen, raw, report['rows']):
            assert expected['id'] == observed['id'] == saved['id']
            assert expected['input_ids'] == observed['payload']['input_ids']
            assert expected['max_new_tokens'] == observed['payload']['sampling_params']['max_new_tokens']
            assert expected['stop'] == observed['payload']['sampling_params']['stop']
            response = observed['response']
            actual = score(expected['benchmark'], response['text'], expected['answer'],
                           truncated=response['meta_info']['finish_reason']['type']=='length')
            for key, value in actual.items():
                assert saved[key] == value, (mode, expected['id'], key)
            lanes[mode].append(dict(text=response['text'], tokens=response['output_ids'], score=actual))
    counts = Counter()
    rows = []
    for item, n, h in zip(frozen, lanes['npu'], lanes['hybrid']):
        correct_n, correct_h = n['score']['correct'], h['score']['correct']
        state = ('gain' if correct_h else 'loss') if correct_n != correct_h else ('both_correct' if correct_n else 'both_wrong')
        counts[(item['benchmark'],state)] += 1
        first = next((i for i,(x,y) in enumerate(zip(n['tokens'],h['tokens'])) if x!=y), None)
        if first is None and len(n['tokens'])!=len(h['tokens']):
            first = min(len(n['tokens']),len(h['tokens']))
        rows.append(dict(id=item['id'],benchmark=item['benchmark'],subject=item.get('subject'),
            answer=item['answer'],state=state,first_token_difference=first,npu=n,hybrid=h))
    report = dict(status='ALL_288_FROZEN_INPUTS_AND_SCORES_REAUDITED',
        counts={bench:{state:counts[(bench,state)] for state in ('gain','loss','both_correct','both_wrong')}
                for bench in ('gsm8k','cmmlu')},rows=rows,
        caveat='Answer differences do not alone attribute a numerical mechanism. No sample replacement or rule change.')
    with a.output.open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(report['counts']))


if __name__=='__main__':
    main()
