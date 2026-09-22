"""Freeze new QD/HD and disjoint quality IDs before new model outputs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata
from tokenizers import Tokenizer


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ['b0','spec','tokenizer','quality_source','old_quality','old_dev','old_prompts','output']:
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    assert json.loads(a.b0.read_text())['status']=='BACKEND_DECOMPOSITION_GATE_PASS'
    assert sha(a.tokenizer)=='41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669'
    assert sha(a.quality_source)=='779d1318137ee1a3573bafe20e265f2aa31170660c08b49ada0507fdb179a3ba'
    spec=json.loads(a.spec.read_text());tok=Tokenizer.from_file(str(a.tokenizer))
    old=json.loads(a.old_prompts.read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    bindings={};seen=set()
    def write(name,obj):
        raw=(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
        with (a.output/name).open('xb') as f:f.write(raw)
        bindings[name]=hashlib.sha256(raw).hexdigest()
    for corpus in ['qd6','hd6']:
        rows=[]
        assert len(spec[corpus])==12
        for i,(family,text) in enumerate(spec[corpus]):
            norm=' '.join(unicodedata.normalize('NFKC',text).split())
            assert norm not in seen;seen.add(norm)
            assert hashlib.sha256(norm.encode()).hexdigest() not in old['text_sha256'],'HISTORICAL_PROMPT_DUPLICATE'
            ids=tok.encode(text,add_special_tokens=False).ids
            assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest() not in old['input_ids_sha256'],'HISTORICAL_TOKEN_DUPLICATE'
            rows.append(dict(rid=f'r66-{corpus}-{i:02d}',prompt_id=f'{corpus}-{i:02d}',
                corpus=corpus,family=family,text=text,input_ids=ids))
        write(corpus+'.json',dict(version='r66-heavy-study-v1',token_budget=64,requests=rows,
            bos_policy='no_added_bos_matches_frozen_static_numerical_protocol'))
    excluded={r['id'] for path in [a.old_quality,a.old_dev] for r in json.loads(path.read_text())['rows']}
    pool=[r for r in json.loads(a.quality_source.read_text())['rows'] if r['id'] not in excluded]
    rank=lambda r:hashlib.sha256(('6606:'+r['id']).encode()).hexdigest()
    selected=sorted([r for r in pool if r['benchmark']=='gsm8k'],key=rank)[:32]
    subjects=defaultdict(list)
    for r in pool:
        if r['benchmark']=='cmmlu':subjects[r['subject']].append(r)
    assert len(subjects)==67
    selected_cm=[sorted(subjects[s],key=rank)[0] for s in sorted(subjects)]
    chosen={r['id'] for r in selected_cm}
    selected_cm+=sorted([r for rs in subjects.values() for r in rs if r['id'] not in chosen],key=rank)[:189]
    selected+=selected_cm
    assert len(selected)==288 and not ({r['id'] for r in selected}&excluded)
    write('quality.json',dict(version='r66-quality-screen-v1',rows=selected,
        source_sha256=sha(a.quality_source),excluded_old_quality_ids=sorted(excluded),
        selection='SHA256(6606:id); CMMLU one per67 subjects then global rank to256; no outputs used',
        pass_rule='no observed net accuracy loss on either benchmark and no increase in invalid count',
        margin_absolute=0.,population_noninferiority_claim=False))
    contract=dict(version='r66-heavy-static-v1',status='FROZEN_BEFORE_QD6_AND_HD6',
        B0_sha256=sha(a.b0),bindings=bindings,tokenizer_sha256=sha(a.tokenizer),
        expert_rel_l2_max=.01,system_semantics='EXACT',same_path_determinism='EXACT',
        model_distribution='DIAGNOSTIC_ONLY_NO_REFITTED_GLOBAL_MAX_BOUND',
        model_finite_required=True,matched_history_required=True,
        quality_rule='zero_observed_net_loss_each_endpoint; invalid_count_no_increase',
        quality_scope='FROZEN_SUBSET_ENGINEERING_SCREEN_NOT_POPULATION_NONINFERIORITY',
        deferred=0,dynamic=False,tp=1,batch=1,
        primary_profile='four-R32',historical_round6_5='BLOCKED_UNCHANGED',
        candidate_frozen=False,heldout_one_shot=True)
    write('CONTRACT_V1.json',contract)
    write('FREEZE_MANIFEST.json',dict(files=dict(bindings),spec_sha256=sha(a.spec),
        b0_sha256=sha(a.b0),old_quality_sha256=sha(a.old_quality),old_dev_sha256=sha(a.old_dev),
        historical_prompt_index_sha256=sha(a.old_prompts)))
    print(json.dumps(bindings,indent=2))


if __name__=='__main__':main()
