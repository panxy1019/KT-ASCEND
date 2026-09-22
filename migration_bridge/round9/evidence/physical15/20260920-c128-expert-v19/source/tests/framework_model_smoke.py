"""Small HTTP corpus run with raw token/logprob evidence; no benchmark claim."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request

from transformers import AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--corpus', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--reference', type=Path)
a = p.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
model = Path('/models/de4b8e4d43b917e7706784d8bb445c9af86a3540')
tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
corpus = json.loads(a.corpus.read_text())
report = dict(status='RUNNING', scope='FRAMEWORK_SMALL_SMOKE_NOT_QUALITY_OR_PERFORMANCE',
              corpus_sha256=hashlib.sha256(a.corpus.read_bytes()).hexdigest(),
              tokenizer_files={name: hashlib.sha256((model/name).read_bytes()).hexdigest()
                               for name in ('tokenizer.json', 'tokenizer_config.json')},
              rows=[])
try:
    for prompt in corpus['prompts']:
        rendered = tokenizer.apply_chat_template([dict(role='user', content=prompt['text'])],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) > 256: raise ValueError('short framework smoke prompt exceeds256')
        body = dict(rid=prompt['id'], input_ids=ids, sampling_params=dict(temperature=0, max_new_tokens=24),
                    return_logprob=True, logprob_start_len=-1)
        request = urllib.request.Request('http://127.0.0.1:30091/generate', data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response: data = json.load(response)
        logs = data['meta_info']['output_token_logprobs']
        row = dict(request=body, response=data, finite=bool(logs) and all(math.isfinite(x[0]) for x in logs))
        report['rows'].append(row)
        if not row['finite'] or not data['text'].strip(): raise RuntimeError('empty/nonfinite generation')
        if 'expected_integer' in prompt:
            row['known_answer'] = re.fullmatch(r'\s*' + str(prompt['expected_integer']) + r'\s*', data['text']) is not None
            if not row['known_answer']: raise RuntimeError('integer canary mismatch')
        print(prompt['id'], 'FINITE', repr(data['text']), flush=True)
    if a.reference:
        reference = json.loads(a.reference.read_text())
        if report['corpus_sha256'] != reference['corpus_sha256']: raise RuntimeError('different reference corpus')
        for current, previous in zip(report['rows'], reference['rows'], strict=True):
            current_ids = [x[1] for x in current['response']['meta_info']['output_token_logprobs']]
            previous_ids = [x[1] for x in previous['response']['meta_info']['output_token_logprobs']]
            if current_ids != previous_ids: raise RuntimeError('same-placement token trajectory differs')
        report['same_placement_token_identity'] = True
    report['status'] = 'PASS_FRAMEWORK_HTTP2'
except BaseException as error:
    report.update(status='FAIL', error=repr(error))
    raise
finally:
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
