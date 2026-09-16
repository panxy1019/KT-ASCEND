"""One frozen independent quality study, production unchanged; batch1 only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.request


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score_letters(values,candidate_ids):
    if len(values)!=4 or {x[1] for x in values}!=set(candidate_ids):
        raise ValueError('CANDIDATE_LOGPROB_IDENTITY')
    mapping={x[1]:float(x[0]) for x in values}
    if not all(math.isfinite(v) for v in mapping.values()):raise ValueError('NONFINITE_CANDIDATE')
    # Explicit deterministic label-order tie break, same for both paths.
    index=max(range(4),key=lambda i:mapping[candidate_ids[i]])
    return 'ABCD'[index],[mapping[t] for t in candidate_ids]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--input-sha256',required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();base=Path('/workspace/results')
    assert sha(a.inputs)==a.input_sha256
    assert json.loads((base/'r65-r66a-mixed-partials-v1/mixed_partials.json').read_text())['status']=='ALL_MIXED_PARTIALS_EXACT'
    assert json.loads((base/'r65-r66a-mixed-partials-guard-v1/reaping.json').read_text())['complete']
    approval=json.loads((base/'r65-r66-quality-execution-approval-v1.json').read_text())
    assert sha(a.profile) in approval['allowed_profile_sha256']
    runtime=base/'r65-r66-runtime-source-final-b0.json'
    assert sha(runtime)==approval['runtime_manifest_sha256']
    parsers=Path('/workspace/r66-v6/tools')
    for name,digest in approval['parser_sha256'].items():assert sha(parsers/name)==digest
    def check():
        for item in json.loads(runtime.read_text())['files']:assert sha(item['path'])==item['sha256'],item['path']
        for path,digest in approval['asset_sha256'].items():assert sha(path)==digest,path
    check()
    sys.path.insert(0,str(parsers));from quality_answers_v2 import score
    sys.path.insert(0,'/workspace/r65-tools/migration_bridge/round6_5/tools')
    from static_session import session
    assert 'r66-debug' not in os.environ.get('PYTHONPATH','')
    os.environ.pop('KT_R65_STAGEWISE',None)
    inputs=json.loads(a.inputs.read_text());rows=inputs['rows']
    assert inputs['version']=='r66b-base-protocol-v1' and len(rows)==288
    report=dict(status='RUNNING',rows=[],input_sha256=a.input_sha256,profile_sha256=sha(a.profile),
                runner_sha256=sha(__file__),started_unix=time.time(),population_noninferiority_claim=False)
    try:
        with session(a.output,a.profile,capture=False):
            with (a.output/'requests.jsonl').open('x') as log:
                for row in rows:
                    rid='r66b-'+row['id'];budget=row['max_new_tokens']
                    assert row['input_ids'][0]==100000 and len(row['input_ids'])+budget<=2048
                    payload=dict(rid=rid,input_ids=row['input_ids'],sampling_params=dict(
                        temperature=0,max_new_tokens=budget,ignore_eos=False,stop=row['stop']),
                        return_logprob=True,logprob_start_len=-1,top_logprobs_num=0)
                    if row['benchmark']=='cmmlu':payload['token_ids_logprob']=row['candidate_ids']
                    request=urllib.request.Request('http://127.0.0.1:31000/generate',data=json.dumps(payload).encode(),
                        headers={'Content-Type':'application/json'})
                    start=time.monotonic()
                    with urllib.request.urlopen(request,timeout=900) as response:answer=json.load(response)
                    seconds=time.monotonic()-start
                    log.write(json.dumps(dict(id=row['id'],payload=payload,response=answer,seconds=seconds))+'\n');log.flush()
                    meta=answer['meta_info'];tokens=answer['output_ids'];lp=meta['output_token_logprobs']
                    assert meta['id']==rid and meta['prompt_tokens']==len(row['input_ids'])
                    assert 0<len(tokens)<=budget and len(lp)==len(tokens) and [v[1] for v in lp]==tokens
                    assert all(v[0] is not None and math.isfinite(float(v[0])) for v in lp)
                    finish=meta['finish_reason']['type'];assert finish in ('stop','length')
                    if row['benchmark']=='cmmlu':
                        assert len(tokens)==1
                        values=meta['output_token_ids_logprobs'];assert len(values)==1
                        predicted,probs=score_letters(values[0],row['candidate_ids'])
                        result=dict(correct=predicted==row['answer'],invalid=False,truncated=False,
                                    predicted=predicted,candidate_logprobs=probs)
                    else:result=score('gsm8k',answer['text'],row['answer'],truncated=finish=='length')
                    report['rows'].append(dict(id=row['id'],benchmark=row['benchmark'],subject=row['subject'],
                        generated_tokens=len(tokens),seconds=seconds,finish_reason=finish,**result))
                    progress=dict(completed=len(report['rows']),total=288,elapsed_seconds=time.time()-report['started_unix'])
                    (a.output/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
        assert json.loads((a.output/'lifecycle.json').read_text())['status']=='COMPLETE'
        check();report['status']='QUALITY_SCREEN_COLLECTION_COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc));raise
    finally:
        report['ended_unix']=time.time()
        if a.output.exists():(a.output/'quality-report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
