"""Frozen four-R32 engineering quality screen; requires reviewed HD6 predecessor."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.request


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--approval',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--parser-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    approval=json.loads(a.approval.read_text())
    if approval.get('quality_execution_allowed') is not True or approval.get('heldout_system_review')!='PASS':
        raise ValueError('HELDOUT_REVIEW_REQUIRED')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(a.inputs)!='9618d054cf765e1d6328d1581435ec97f9f4271715b794dc07e0a9889a483fcb':
        raise ValueError('QUALITY_INPUT_DRIFT')
    for name in ('quality_answers.py','quality_answers_v2.py'):
        if sha(a.parser_dir/name)!=approval['parser_sha256'][name]:raise ValueError('PARSER_DRIFT')
    if sha(a.profile) not in approval['allowed_profile_sha256']:raise ValueError('PROFILE_DRIFT')
    sys.path.insert(0,str(a.parser_dir))
    from quality_answers_v2 import score
    sys.path.insert(0,'/workspace/r65-tools/migration_bridge/round6_5/tools')
    from static_session import session
    if 'r66-debug' in os.environ.get('PYTHONPATH',''):raise ValueError('DIAGNOSTIC_EXTENSION')
    os.environ.pop('KT_R65_STAGEWISE',None)
    rows=json.loads(a.inputs.read_text())['rows']
    assert len(rows)==288 and len({r['id'] for r in rows})==288
    assert sum(r['benchmark']=='gsm8k' for r in rows)==32
    assert sum(r['benchmark']=='cmmlu' for r in rows)==256
    report=dict(status='RUNNING',rows=[],input_sha256=sha(a.inputs),profile_sha256=sha(a.profile),
                approval_sha256=sha(a.approval),started_unix=time.time(),population_noninferiority_claim=False)
    try:
        with session(a.output,a.profile,capture=False):
            with (a.output/'requests.jsonl').open('x') as log:
                for row in rows:
                    rid='r66-quality-'+row['id']
                    budget=512 if row['benchmark']=='gsm8k' else 32
                    assert row['max_new_tokens']==budget and row['input_ids'][0]==100000
                    payload=dict(rid=rid,input_ids=row['input_ids'],sampling_params=dict(
                        temperature=0,max_new_tokens=budget,ignore_eos=False,stop=row['stop']),
                        return_logprob=True,logprob_start_len=-1,top_logprobs_num=0)
                    req=urllib.request.Request('http://127.0.0.1:31000/generate',data=json.dumps(payload).encode(),
                                               headers={'Content-Type':'application/json'})
                    start=time.monotonic()
                    with urllib.request.urlopen(req,timeout=900) as response:answer=json.load(response)
                    seconds=time.monotonic()-start
                    log.write(json.dumps(dict(id=row['id'],payload=payload,response=answer,seconds=seconds))+'\n');log.flush()
                    tokens=answer['output_ids'];meta=answer['meta_info'];lp=meta['output_token_logprobs']
                    assert meta['id']==rid and meta['prompt_tokens']==len(row['input_ids'])
                    assert 0<len(tokens)<=budget and all(type(t) is int and 0<=t<102400 for t in tokens)
                    assert len(lp)==len(tokens) and [r[1] for r in lp]==tokens
                    assert all(r[0] is not None and math.isfinite(float(r[0])) for r in lp)
                    finish=meta['finish_reason']['type'];assert finish in ('stop','length')
                    report['rows'].append(dict(id=row['id'],benchmark=row['benchmark'],subject=row.get('subject'),
                        generated_tokens=len(tokens),seconds=seconds,finish_reason=finish,
                        **score(row['benchmark'],answer['text'],row['answer'],truncated=finish=='length')))
                    progress=dict(completed=len(report['rows']),total=288,elapsed_seconds=time.time()-report['started_unix'])
                    (a.output/'progress.json').write_text(json.dumps(progress)+'\n')
                    print(json.dumps(progress),flush=True)
        lifecycle=json.loads((a.output/'lifecycle.json').read_text())
        if lifecycle['status']!='COMPLETE':raise ValueError('LIFECYCLE_NOT_COMPLETE')
        report['status']='QUALITY_SCREEN_COLLECTION_COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc));raise
    finally:
        report['ended_unix']=time.time()
        if a.output.exists():(a.output/'quality-report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
