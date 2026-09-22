"""One independently ranked base-model screen; no inference or outcome input."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rank(identifier):return hashlib.sha256(('660602:'+identifier).encode()).hexdigest()


def main():
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    p=argparse.ArgumentParser()
    for name in ('train','cmmlu','pool','old','tokenizer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    expected={'train':'ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d',
              'cmmlu':'22ecf70b28bef447ee7d8aa5fe144f56996762f901a8537b03b7693773c672a6',
              'pool':'779d1318137ee1a3573bafe20e265f2aa31170660c08b49ada0507fdb179a3ba',
              'old':'9618d054cf765e1d6328d1581435ec97f9f4271715b794dc07e0a9889a483fcb',
              'tokenizer':'41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669'}
    for name,digest in expected.items():assert sha(getattr(a,name))==digest,name
    tok=Tokenizer.from_file(str(a.tokenizer))
    encode=lambda text:tok.encode(text,add_special_tokens=False).ids
    train=pq.read_table(a.train).to_pylist()
    exemplars=[]
    for i,row in enumerate(train):
        answer=re.sub(r'<<.*?>>','',row['answer'])
        text='Question: '+row['question']+'\nAnswer: '+answer+'\n\n'
        if len(encode(text))<=128:
            exemplars.append(dict(id=f'gsm8k-train-main-{i:05d}',text=text))
    exemplars=sorted(exemplars,key=lambda r:rank(r['id']))[:8]
    assert len(exemplars)==8
    old=json.loads(a.old.read_text())
    excluded=set(old['excluded_old_quality_ids'])|{r['id'] for r in old['rows']}
    pool=[r for r in json.loads(a.pool.read_text())['rows'] if r['id'] not in excluded]
    selected=sorted([r for r in pool if r['benchmark']=='gsm8k'],key=lambda r:rank(r['id']))[:32]
    subjects=sorted({r['subject'] for r in pool if r['benchmark']=='cmmlu'})
    assert len(subjects)==67
    cm=[min((r for r in pool if r['benchmark']=='cmmlu' and r['subject']==s),key=lambda r:rank(r['id'])) for s in subjects]
    chosen={r['id'] for r in cm}
    cm+=sorted([r for r in pool if r['benchmark']=='cmmlu' and r['id'] not in chosen],key=lambda r:rank(r['id']))[:189]
    selected+=cm;assert len(selected)==288
    letter_ids=[encode(c) for c in 'ABCD'];assert all(len(x)==1 for x in letter_ids)
    letter_ids=[x[0] for x in letter_ids];assert len(set(letter_ids))==4
    def item(r,answer=False):
        text='题目：'+r['Question']+'\n'+''.join(c+'. '+r[c]+'\n' for c in 'ABCD')+'答案：'
        return text+(r['Answer']+'\n\n' if answer else '')
    rows=[];dev={}
    with zipfile.ZipFile(a.cmmlu) as z:
        for s in subjects:
            records=list(csv.DictReader(io.StringIO(z.read('dev/'+s+'.csv').decode('utf-8-sig'))))
            assert len(records)==5
            dev[s]=records
        for r in selected:
            if r['benchmark']=='gsm8k':
                prompt=''.join(e['text'] for e in exemplars)+'Question: '+r['question']+'\nAnswer:'
                budget=512;stop=['\nQuestion:','\n\nQuestion:','\nQ:'];candidates=None
            else:
                records=list(csv.DictReader(io.StringIO(z.read('test/'+r['subject']+'.csv').decode('utf-8-sig'))))
                record=records[r['source_row_index']]
                assert record['Question']==r['question'] and record['Answer']==r['answer']
                prompt='以下是单项选择题及其正确答案。\n\n'+''.join(item(e,True) for e in dev[r['subject']])+item(record)
                budget=1;stop=[];candidates=letter_ids
                for c,tid in zip('ABCD',letter_ids):assert encode(prompt+c)==encode(prompt)+[tid],'LETTER_BOUNDARY'
            ids=[100000]+encode(prompt)
            assert len(ids)+budget<=2048,('CONTEXT_PREFLIGHT_FAILURE',r['id'],len(ids),budget)
            rows.append(dict(id=r['id'],benchmark=r['benchmark'],subject=r.get('subject'),
                             answer=r['answer'],question=r['question'],prompt=prompt,input_ids=ids,
                             max_new_tokens=budget,stop=stop,candidate_ids=candidates))
    result=dict(version='r66b-base-protocol-v1',rows=rows,excluded_ids=sorted(excluded),
                source_sha256=expected,selection_seed='660602',gsm_exemplars=exemplars,
                cmmlu_dev=dev,letter_ids=letter_ids,
                quality_rule='zero_observed_net_loss_per_endpoint;invalid_count_no_increase',
                population_noninferiority_claim=False,old_v1_status='BLOCKED_UNCHANGED')
    a.output.mkdir(exist_ok=False)
    raw=(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
    (a.output/'quality.json').write_bytes(raw)
    manifest=dict(quality_sha256=hashlib.sha256(raw).hexdigest(),tool_sha256=sha(__file__),
                  sources=expected,counts={'gsm8k':32,'cmmlu':256},
                  maximum_input_tokens=max(len(r['input_ids']) for r in rows),
                  context_limit=2048,contract_preregistered=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
