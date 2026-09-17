"""Offline, deterministic scheduler-only corpus; never uses quality outcomes."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--cmmlu', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    topics = ['river monitoring', 'library indexing', 'orbital mechanics',
              'database transactions', 'food preservation', 'urban planning',
              'integer algorithms', 'solar generation', 'JSON validation',
              'language translation', 'network reliability', 'historical archives']
    corpus_hashes = set()
    # Exact text and token-identity disjointness is checked against every nested
    # prompt/input-ID value in the frozen quality manifest, not its answer labels.
    excluded_text, excluded_ids = set(), set()
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ('prompt', 'text') and isinstance(v, str): excluded_text.add(v)
                if k in ('input_ids', 'input_token_ids') and isinstance(v, list): excluded_ids.add(tuple(v))
                walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(json.loads(a.cmmlu.read_text()))
    summary = {'seed': 0, 'protocol': 'greedy batch1; collect prefill+16 decode; no quality labels',
               'cmmlu_sha256': digest(a.cmmlu), 'corpora': {}}
    tokenizer_files = {str(x.relative_to(Path(a.tokenizer))): digest(x)
                       for x in sorted(Path(a.tokenizer).rglob('*')) if x.is_file()}
    summary['tokenizer_files'] = tokenizer_files
    for split, buckets in [('DEV', [64,96,64,32]), ('HOLDOUT', [32,48,32,16])]:
        rows, index = [], 0
        for bucket, count, low, high in zip(['short','medium','long','structured'], buckets,
                                          [64,256,1024,128], [255,1023,4095,767]):
            for j in range(count):
                family = ['English','Chinese','code','math','JSON'][index % 5]
                topic = topics[(index * 7 + (1 if split == 'HOLDOUT' else 0)) % len(topics)]
                # Decode/re-encode at a subword cut can shift a few tokens.
                # Stay inside the frozen bucket rather than assuming a BOS.
                target = low + 8 + (j * 97 + index * 31) % (high-low-15)
                prefix = f'Scheduler case {split}-{index:03d}. Topic: {topic}. '
                if family == 'Chinese':
                    passage = f'请阅读关于{topic}的记录。每段包含观测、约束和解释；注意区分事实与假设，保持单位一致。'
                    question = '\n请用中文总结主要约束，给出一个可验证的建议。'
                elif family == 'code':
                    passage = 'A service stores records by integer ID. Insertions are atomic; retries must be idempotent. Validate empty input, duplicate keys and overflow. Tests should cover successful and failing transactions. '
                    question = '\nWrite Python pseudocode with explicit validation and three tests.'
                elif family == 'math':
                    passage = 'Each observation uses SI units. A reservoir receives 17 units and releases 11 per hour; initial inventory is 240. Model conservation before rounding, and distinguish rate from accumulated volume. '
                    question = '\nState the equation and calculate inventory after 12 hours.'
                elif family == 'JSON':
                    passage = 'The schema contains id, status, readings, and unit. IDs are unique, status is either valid or missing, readings are finite numbers, and missing readings must be null rather than zero. '
                    question = '\nReturn a valid JSON example and explain one validation invariant.'
                else:
                    passage = f'A report about {topic} separates observations from assumptions. It records dates, units, operational constraints, and failure recovery. Independent checks are necessary before drawing causal conclusions. '
                    question = '\nSummarize the constraints and propose a testable improvement.'
                text = prefix
                while len(tok.encode(text + question, add_special_tokens=True)) < target:
                    text += passage
                # Reserve an intact task suffix while truncating context. Freeze
                # re-encoded IDs, not a token truncation masquerading as text.
                suffix_n = len(tok.encode(question, add_special_tokens=False))
                context = tok.encode(text, add_special_tokens=False)[:max(1,target-suffix_n-1)]
                text = tok.decode(context, skip_special_tokens=True) + question
                ids = tok.encode(text, add_special_tokens=True)
                assert low <= len(ids) <= high, (split,bucket,len(ids))
                assert text not in excluded_text and tuple(ids) not in excluded_ids
                fingerprint = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                assert fingerprint not in corpus_hashes
                corpus_hashes.add(fingerprint)
                rows.append(dict(id=f'{split.lower()}_{index:03d}', family=family,
                                 bucket=bucket, prompt=text, input_ids=ids, tokens=len(ids)))
                index += 1
        path = a.output / f'SCHED_{split}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        summary['corpora'][split] = dict(count=len(rows),sha256=digest(path),
            bucket_counts=dict(zip(['short','medium','long','structured'],buckets)),
            min_tokens=min(r['tokens'] for r in rows),max_tokens=max(r['tokens'] for r in rows),
            total_tokens=sum(r['tokens'] for r in rows))
    summary.update(dev_holdout_overlap=0,cmmlu_exact_overlap=0,
                   limitation='Synthetic representative scheduler corpus, not a task-quality benchmark.')
    (a.output/'CORPUS_FREEZE.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary['corpora']))


if __name__ == '__main__': main()
