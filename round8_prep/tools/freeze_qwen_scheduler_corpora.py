"""Freeze Qwen scheduler DEV/HOLDOUT corpora, disjoint from frozen CMMLU."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--cmmlu',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    cmmlu=json.loads(a.cmmlu.read_text());excluded={tuple(row['input_ids']) for row in cmmlu['rows']}
    families=['English','Chinese','code','math','JSON']
    texts={
      'English':'Explain how a monitoring service validates observations, handles retries, and records evidence. ',
      'Chinese':'请说明一个监控系统如何验证观测值、处理失败重试并记录可审计证据。',
      'code':'Write Python pseudocode for atomic inserts, duplicate detection, bounded retries, and tests. ',
      'math':'Model inventory conservation with an initial value, an input rate, an output rate, units, and checks. ',
      'JSON':'Describe a JSON schema with unique IDs, finite readings, explicit nulls, status, and unit validation. '}
    specs={'DEV':[('short',64,96,224),('medium',96,384,896),('long',64,1200,3072),('structured',32,512,768)],
           'HOLDOUT':[('short',32,96,224),('medium',48,384,896),('long',32,1200,3072),('structured',16,512,768)]}
    all_ids=set();summary=dict(version='qwen35-scheduler-v1',cmmlu_sha256=sha(a.cmmlu),splits={})
    for split,buckets in specs.items():
        rows=[];index=0
        for bucket,count,low,high in buckets:
            for item in range(count):
                family=families[index%len(families)]
                target=low+(item*97+index*31+(17 if split=='HOLDOUT' else 0))%(high-low+1)
                content=(f'{split} scheduler case {index}. '+texts[family])*(target//12+8)
                rendered=tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)
                ids=tok.encode(rendered,add_special_tokens=False)[:target]
                assert len(ids)==target and tuple(ids) not in excluded and tuple(ids) not in all_ids
                all_ids.add(tuple(ids));rows.append(dict(id=f'{split.lower()}_{index:03d}',
                    family=family,bucket=bucket,input_ids=ids,tokens=len(ids),
                    prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest()))
                index+=1
        path=a.output/f'SCHED_{split}.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        summary['splits'][split]=dict(count=len(rows),sha256=sha(path),
            bucket_counts={b:c for b,c,_,_ in buckets},min_tokens=min(x['tokens'] for x in rows),
            max_tokens=max(x['tokens'] for x in rows),total_tokens=sum(x['tokens'] for x in rows))
    summary.update(dev_holdout_exact_overlap=0,cmmlu_exact_overlap=0,
        purpose='scheduler threshold and held-out routing; no quality labels')
    (a.output/'CORPUS_FREEZE.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['splits']))


if __name__=='__main__':main()
