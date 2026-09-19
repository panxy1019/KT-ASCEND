"""Frozen diagnostic compact matrix, not timed performance or task quality."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import requests
from transformers import AutoTokenizer

PROMPTS={
    'zh':'用两句话解释为什么天空通常呈蓝色。',
    'en':'Explain the difference between a list and a set in two sentences.',
    'math':'Compute 17 times 23 and briefly show the calculation.',
    'json':'Return only a JSON object with keys name and count, values apple and 3.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['same','restart'],required=True)
    a=p.parse_args();root=Path('/workspace/round8')
    out=root/'q64-compact-v1';out.mkdir(exist_ok=True)
    t=AutoTokenizer.from_pretrained(root/'model-sample',local_files_only=True)
    requests_frozen={}
    for key,prompt in PROMPTS.items():
        rendered=t.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,
                                      add_generation_prompt=True,enable_thinking=False)
        requests_frozen[key]=dict(text=prompt,input_ids=t.encode(rendered,add_special_tokens=False))
    protocol=dict(prompts=requests_frozen,temperature=0,seed=0,thinking=False,
                  same_budgets=[32,16,16,16,16,16,16,8],restart_budgets=[32])
    frozen=json.dumps(protocol,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
    path=out/'protocol.json'
    if path.exists():assert path.read_text()==frozen
    else:
        assert a.phase=='same';path.write_text(frozen)
    raw=out/(a.phase+'-raw.jsonl')
    assert not raw.exists(), 'DO_NOT_OVERWRITE_RAW'
    baseline={}
    if a.phase=='restart':
        rows=[json.loads(s) for s in (out/'same-raw.jsonl').read_text().splitlines()]
        baseline={r['prompt_id']:r['response']['output_ids'] for r in rows if r['budget']==32}
        assert json.loads((out/'same-summary.json').read_text())['status']=='PASS'
    count=0
    with raw.open('x') as f:
        for key,request in requests_frozen.items():
            for budget in protocol[a.phase+'_budgets']:
                payload=dict(input_ids=request['input_ids'],sampling_params=dict(temperature=0,max_new_tokens=budget),return_logprob=True)
                r=requests.post('http://127.0.0.1:18400/generate',json=payload,timeout=900)
                row=dict(prompt_id=key,budget=budget,http_status=r.status_code,response=r.json())
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();r.raise_for_status()
                data=row['response'];ids=data['output_ids'];assert ids and data['text'].strip()
                assert all(math.isfinite(v[0]) for v in data['meta_info']['output_token_logprobs'])
                if a.phase=='same' and budget==32:baseline[key]=ids
                else:assert ids==baseline[key][:budget], f'TRAJECTORY_MISMATCH:{key}:{budget}'
                count+=1;print(f'{a.phase} {count} {key} {budget} PASS',flush=True)
    summary=dict(status='PASS',phase=a.phase,requests=count,prompts=4,
                 protocol_sha256=hashlib.sha256(frozen.encode()).hexdigest(),
                 purpose='SYSTEM_COMPACT_NOT_QUALITY_OR_PERFORMANCE')
    (out/(a.phase+'-summary.json')).write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
