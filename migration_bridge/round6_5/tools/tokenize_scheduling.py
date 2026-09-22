"""Finish a previously audited scheduling manifest with the pinned tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--texts',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.texts.read_text());audit=json.loads(a.audit.read_text())
    if audit['status']!='TEXT_OVERLAP_AUDITED_NOT_TOKENIZED' or not audit['overlap']['no_exact_overlap']:
        raise ValueError('SCHEDULING_AUDIT')
    if audit['texts_sha256']!=hashlib.sha256(a.texts.read_bytes()).hexdigest():raise ValueError('TEXT_BYTES_DRIFT')
    sha=hashlib.sha256(a.tokenizer.read_bytes()).hexdigest()
    if sha!='41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669':raise ValueError('TOKENIZER_DRIFT')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(a.tokenizer))
    rows=[]
    for split in ('dev','holdout'):
        for family,entries in spec[split].items():
            for i,text in enumerate(entries):
                ids=[100000]+tokenizer.encode(text,add_special_tokens=False).ids
                rows.append(dict(id=f'sched-{split}-{family}-{i:02d}',split=split,family=family,
                    text=text,input_ids=ids,bos_policy='explicit_100000'))
    if len(rows)!=32 or len({tuple(r['input_ids']) for r in rows})!=32:raise ValueError('SCHED_TOKEN_DUPLICATE')
    audit.update(status='SCHEDULING_INPUTS_FROZEN',tokenizer_sha256=sha,rows=rows,
        text_audit_raw_sha256=hashlib.sha256(a.audit.read_bytes()).hexdigest())
    with a.output.open('x') as f:json.dump(audit,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':main()
