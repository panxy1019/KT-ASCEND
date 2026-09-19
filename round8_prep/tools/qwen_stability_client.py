"""One frozen Q32 stability campaign: six prompts, three cycles, 64 tokens."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


ROOT = Path('/workspace/round8')
PROMPTS = [
    'Explain in English how a river monitoring station validates its readings.',
    '请用中文说明数据库事务为什么需要原子性，并给出一个失败恢复例子。',
    'Calculate the sum of the first twenty positive integers and explain the formula.',
    'Write Python pseudocode that validates unique integer IDs and rejects duplicates.',
    'Return a JSON object with fields id, status, readings, and unit, then explain it.',
    'Compare cache eviction policies and propose a reproducible benchmark.',
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', choices=('A', 'B'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(ROOT/'model-sample', local_files_only=True)
    inputs=[]
    for prompt in PROMPTS:
        text=tok.apply_chat_template([{'role':'user','content':prompt}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
        inputs.append(tok.encode(text, add_special_tokens=False))
    protocol=dict(version=1, campaign=a.campaign, cycles=3, output_tokens=64,
                  temperature=0, ignore_eos=True, prompts=PROMPTS,
                  input_ids=inputs, batch=1, dynamic=False, deferred=False,
                  graph=False, diagnostic_hooks=False)
    (a.output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    references={}; rows=[]
    raw=(a.output/'raw.jsonl').open('x')
    for cycle in range(3):
        for index, ids in enumerate(inputs):
            payload=dict(input_ids=ids,sampling_params=dict(
                temperature=0,max_new_tokens=64,ignore_eos=True))
            started=time.perf_counter()
            response=requests.post('http://127.0.0.1:18400/generate',json=payload,timeout=1800)
            response.raise_for_status(); data=response.json(); tokens=data['output_ids']
            assert len(tokens)==64 and all(isinstance(x,int) for x in tokens)
            if index in references:
                assert tokens==references[index],f'trajectory drift prompt {index} cycle {cycle}'
            else: references[index]=tokens
            row=dict(campaign=a.campaign,cycle=cycle,prompt_index=index,
                     input_tokens=len(ids),output_tokens=64,seconds=time.perf_counter()-started,
                     token_sha256=digest(tokens),finite=True,exact=True)
            rows.append(row);raw.write(json.dumps(dict(row=row,response=data))+'\n');raw.flush()
            (a.output/'progress.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
    raw.close()
    (a.output/'summary.json').write_text(json.dumps(dict(
        status='CAMPAIGN_PASS',campaign=a.campaign,requests=len(rows),generated_tokens=1152,
        unique_prompt_trajectories=len({digest(v) for v in references.values()}),
        reference_tokens=references,rows=rows),indent=2)+'\n')


if __name__=='__main__': main()
