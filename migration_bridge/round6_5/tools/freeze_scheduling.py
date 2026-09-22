"""Freeze offline scheduling IDs and exact normalized-text overlap audit."""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata


def normalize(text):return ' '.join(unicodedata.normalize('NFKC',text).lower().split())


def texts(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('text','prompt','question') and isinstance(item,str):yield item
            else:yield from texts(item)
    elif isinstance(value,list):
        for item in value:yield from texts(item)


def audit(spec,paths):
    reserved=set();sources={}
    for path in sorted(paths):
        raw=path.read_bytes();doc=json.loads(raw)
        reserved.update(normalize(t) for t in texts(doc))
        sources[str(path)]=hashlib.sha256(raw).hexdigest()
    seen=set()
    for split in ('dev','holdout'):
        if set(spec[split])!={'english','chinese','numeric','structured'}:raise ValueError('SCHED_FAMILY')
        for family,rows in spec[split].items():
            if len(rows)!=4:raise ValueError('SCHED_COUNT')
            for text in rows:
                key=normalize(text)
                if key in seen or key in reserved:raise ValueError('SCHED_EXACT_TEXT_OVERLAP')
                seen.add(key)
    return dict(source_sha256=sources,normalized_reserved_count=len(reserved),
                new_unique=32,no_exact_overlap=True,semantic_overlap_not_proven=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--texts',type=Path,required=True)
    p.add_argument('--prior',type=Path,nargs='+',required=True)
    p.add_argument('--tokenizer',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.texts.read_text())
    paths=[]
    for path in a.prior:paths.extend(path.rglob('*.json') if path.is_dir() else [path])
    result=dict(version='r65-scheduling-frozen-v1',texts_sha256=hashlib.sha256(a.texts.read_bytes()).hexdigest(),
        overlap=audit(spec,paths),token_budget=64,selection_rule='per_layer_count_desc_then_logical_id_asc',
        rows=[],status='TEXT_OVERLAP_AUDITED_NOT_TOKENIZED')
    if a.tokenizer:
        sha=hashlib.sha256(a.tokenizer.read_bytes()).hexdigest()
        if sha!='41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669':raise ValueError('TOKENIZER_DRIFT')
        from tokenizers import Tokenizer
        tok=Tokenizer.from_file(str(a.tokenizer))
        for split in ('dev','holdout'):
            for family,rows in spec[split].items():
                for i,text in enumerate(rows):
                    ids=[100000]+tok.encode(text,add_special_tokens=False).ids
                    result['rows'].append(dict(id=f'sched-{split}-{family}-{i:02d}',split=split,
                        family=family,text=text,input_ids=ids,bos_policy='explicit_100000'))
        result.update(status='SCHEDULING_INPUTS_FROZEN',tokenizer_sha256=sha)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':main()
