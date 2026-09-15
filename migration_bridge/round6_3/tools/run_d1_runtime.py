"""Round6.3 candidate-bound S6 determinism; only one runtime subgate.

Reuses the retained response validator, never relabels a Round6.3 candidate as
Round6.2. No invocation before QD3 success and candidate commit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'round6_2/tools'))
from run_runtime_determinism import validate
from evaluate_stateful_contract import frozen, canonical

S6_SHA = 'e51af126389507e03b58ae37e5b39c912ece14d54c1daa0ba4baea634ff8f9c5'


def validate_entry(corpus, candidate, prereg, commit):
    if corpus['corpus'] != 'S6' or len(corpus['prompts']) != 12 or corpus['sha256'] != S6_SHA:
        raise ValueError('FROZEN_RUNTIME_CORPUS_MISMATCH')
    if (candidate['version'] != 'round6.3-v2-candidate' or
            candidate['preregistration_sha256'] != prereg['sha256']):
        raise ValueError('ROUND63_CANDIDATE_PROTOCOL_MISMATCH')
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('CANDIDATE_COMMIT_REQUIRED')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'candidate', 'preregistration', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--candidate-commit', required=True)
    p.add_argument('--restart-reference', type=Path)
    args = p.parse_args()
    corpus, candidate, prereg = map(frozen, (args.corpus, args.candidate, args.preregistration))
    validate_entry(corpus, candidate, prereg, args.candidate_commit)
    references = None
    if args.restart_reference:
        prior = frozen(args.restart_reference)
        if (prior['status'] != 'PASS' or prior['candidate_sha256'] != candidate['sha256'] or
                prior['corpus_sha256'] != corpus['sha256'] or prior['candidate_commit'] != args.candidate_commit):
            raise ValueError('RESTART_REFERENCE_DRIFT')
        references = {r['prompt_id']:r['reference64'] for r in prior['rows']}
    args.output.mkdir(parents=True, exist_ok=False)
    rows, count = [], 0
    with (args.output/'requests.jsonl').open('x') as stream:
        for prompt in corpus['prompts']:
            baseline = references[prompt['id']] if references else None
            hashes, prefixes = [], {}
            for index, budget in enumerate([64] if references else [64]*10 + [8,16,32]):
                rid = f'r63-runtime-{prompt["id"]}-{index}'
                payload = dict(rid=rid, input_ids=prompt['input_ids'],
                    sampling_params=dict(temperature=0, max_new_tokens=budget, ignore_eos=True),
                    return_logprob=True, logprob_start_len=-1, top_logprobs_num=0)
                request = urllib.request.Request('http://127.0.0.1:31000/generate',
                    data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
                begin = time.monotonic()
                with urllib.request.urlopen(request, timeout=600) as response:
                    answer = json.load(response)
                stream.write(json.dumps(dict(payload=payload, response=answer, seconds=time.monotonic()-begin))+'\n')
                stream.flush()
                tokens = validate(answer, rid, budget)
                if baseline is None:
                    baseline = tokens
                if tokens != baseline[:budget]:
                    failure = dict(status='FAIL', prompt_id=prompt['id'], request_id=rid,
                        classification='CLEAN_RESTART_MISMATCH' if references else 'SAME_PATH_NONDETERMINISM',
                        expected=baseline[:budget], actual=tokens)
                    (args.output/'failure.json').write_text(json.dumps(failure, indent=2))
                    raise ValueError(failure['classification'])
                if budget == 64:
                    hashes.append(hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
                else:
                    prefixes[str(budget)] = tokens
                count += 1
                print(json.dumps(dict(requests=count, prompt_id=prompt['id'], budget=budget, exact=True)), flush=True)
            rows.append(dict(prompt_id=prompt['id'], reference64=baseline, hashes64=hashes, short_outputs=prefixes))
    report = dict(status='PASS', kind='clean_restart' if references else 'same_process',
        candidate_sha256=candidate['sha256'], candidate_commit=args.candidate_commit,
        corpus_sha256=corpus['sha256'], preregistration_sha256=prereg['sha256'],
        request_count=count, all_returned_logprobs_finite=True, token_exact=True, rows=rows,
        not_proven='This alone does not close terminal, no-hit, explicit replay, lifecycle or the combined runtime gate.')
    report['sha256'] = canonical(report)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
