"""Read-only evidence collation; never automatically approves an ADR."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def wilson(k, n):
    z = 1.959963984540054
    center = (k / n + z*z/(2*n)) / (1 + z*z/n)
    radius = z * math.sqrt((k/n)*(1-k/n)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [max(0., center-radius), min(1., center+radius)]


def summarize(root):
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    progress = read(root / 'progress.json')
    evidence = {}
    result = dict(sequence_status=progress['status'], final_qualification=False,
                  population_noninferiority_claim=False, performance_claim=False,
                  modes={}, evidence_sha256=evidence)
    for name in ('free_analysis.json', 'paired_quality.json'):
        p = root / name
        if p.exists():
            result[name[:-5]] = read(p)
            evidence[name] = sha(p)
    for mode in ('quality_npu', 'quality_hybrid'):
        folder = root / mode
        report = folder / 'quality-report.json'
        if not report.exists():
            result['modes'][mode] = {'status': 'NOT_COMPLETE'}
            continue
        data = read(report)
        evidence[str(report.relative_to(root))] = sha(report)
        requests = folder / 'requests.jsonl'
        evidence[str(requests.relative_to(root))] = sha(requests)
        lifecycle = read(folder / 'lifecycle.json')
        evidence[str((folder / 'lifecycle.json').relative_to(root))] = sha(folder / 'lifecycle.json')
        samples = [json.loads(line) for line in requests.read_text().splitlines()]
        assert len(samples) == len(data['rows'])
        assert [s['id'] for s in samples] == [r['id'] for r in data['rows']]
        benchmarks = {}
        for name in ('gsm8k', 'cmmlu'):
            rows = [r for r in data['rows'] if r['benchmark'] == name]
            n = len(rows)
            if not n:
                continue
            correct = sum(r['correct'] for r in rows)
            tokens = sum(r['generated_tokens'] for r in rows)
            seconds = sum(r['seconds'] for r in rows)
            subjects = {}
            for r in rows:
                sub = r.get('subject') or name
                item = subjects.setdefault(sub, dict(n=0, correct=0, invalid=0, truncated=0))
                item['n'] += 1
                for key in ('correct', 'invalid', 'truncated'):
                    item[key] += int(r[key])
            benchmarks[name] = dict(n=n, correct=correct, accuracy=correct/n,
                accuracy_wilson95=wilson(correct,n), invalid=sum(r['invalid'] for r in rows),
                truncated=sum(r['truncated'] for r in rows), generated_tokens=tokens,
                request_seconds=seconds, request_latency_median=statistics.median(r['seconds'] for r in rows),
                end_to_end_tokens_per_second=tokens/seconds if seconds else None,
                not_decode_only=True, subjects=subjects,
                baseline_floor_review_required=(name=='cmmlu' and correct/n<=.25) or correct==0)
        result['modes'][mode] = dict(status=data['status'], lifecycle=lifecycle['status'],
                                    profile_sha256=data['profile_sha256'], benchmarks=benchmarks)
    result['review_requirements'] = [
        'No final promotion without same-path, source/provenance and ADR review.',
        'CMMLU small subject cells cannot support subject-level equivalence claims.',
        'A zero-width empirical bootstrap is not zero population uncertainty.',
        'Request throughput includes prefill and differs with output lengths; not a causal speedup claim.',
        'Old Round6.5 remains blocked under its unchanged global numerical bound.',
    ]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.root)
    with a.output.open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('modes', 'free_analysis')},indent=2))


if __name__ == '__main__':
    main()
