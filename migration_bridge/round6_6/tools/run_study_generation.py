"""Observer-OFF greedy repetition/prefix/restart or held-out free generation."""
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
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['repeat','restart','free'],required=True)
    p.add_argument('--reference',type=Path)
    a=p.parse_args()
    digest=hashlib.sha256(a.inputs.read_bytes()).hexdigest()
    allowed=['b70449cc6f744307458c21d4463c8496c5ba09e60aa21e686831b3217cf29570',
             'e812852ccd9ca9f9427b31f56c29ebca8167c7b7116ecd27b8c8e2aa973ab15f']
    assert digest in allowed
    requests=json.loads(a.inputs.read_text())['requests']
    if a.kind!='free':
        # Predefined first prompt of each of four families; no result-driven selection.
        families={'english','chinese','numeric','structured'}
        selected={}
        for row in requests:
            if row['family'] in families:selected.setdefault(row['family'],row)
        assert set(selected)==families
        requests=list(selected.values())
    ref={}
    if a.kind=='restart':
        assert a.reference is not None
        previous=json.loads(a.reference.read_text())
        assert previous['status']=='GENERATION_PASS' and previous['kind']=='repeat' and previous['input_sha256']==digest
        assert previous['profile_sha256']==hashlib.sha256(a.profile.read_bytes()).hexdigest()
        ref={r['rid']:r['reference64'] for r in previous['rows']}
    sys.path.insert(0,'/workspace/r65-tools/migration_bridge/round6_5/tools')
    from static_session import session
    os.environ.pop('KT_R65_STAGEWISE',None)
    if 'r66-debug' in os.environ.get('PYTHONPATH',''):raise ValueError('DIAGNOSTIC_EXTENSION')
    report=dict(status='RUNNING',kind=a.kind,input_sha256=digest,rows=[],request_count=0,
                profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),observer='OFF')
    try:
        with session(a.output,a.profile,capture=False):
            with (a.output/'requests.jsonl').open('x') as stream:
                for row in requests:
                    baseline=ref.get(row['rid']);hashes=[];short={}
                    for i,budget in enumerate([64]*10+[8,16,32] if a.kind=='repeat' else [64]):
                        rid=f"{row['rid']}-{a.kind}-{i}"
                        payload=dict(rid=rid,input_ids=row['input_ids'],sampling_params=dict(
                            temperature=0,max_new_tokens=budget,ignore_eos=True),
                            return_logprob=True,logprob_start_len=-1,top_logprobs_num=0)
                        req=urllib.request.Request('http://127.0.0.1:31000/generate',data=json.dumps(payload).encode(),
                                                   headers={'Content-Type':'application/json'})
                        start=time.monotonic()
                        with urllib.request.urlopen(req,timeout=900) as response:answer=json.load(response)
                        stream.write(json.dumps(dict(payload=payload,response=answer,seconds=time.monotonic()-start))+'\n');stream.flush()
                        tokens=answer['output_ids'];meta=answer['meta_info'];lp=meta['output_token_logprobs']
                        assert meta['id']==rid and len(tokens)==budget
                        assert len(lp)==budget and [r[1] for r in lp]==tokens
                        assert all(r[0] is not None and math.isfinite(float(r[0])) for r in lp)
                        if baseline is None:baseline=tokens
                        if tokens!=baseline[:budget]:raise ValueError('SAME_PATH_OR_RESTART_MISMATCH:'+rid)
                        if budget==64:hashes.append(hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
                        else:short[str(budget)]=tokens
                        report['request_count']+=1
                        (a.output/'progress.json').write_text(json.dumps(dict(requests=report['request_count'],rid=rid))+'\n')
                        print(json.dumps(dict(requests=report['request_count'],rid=rid)),flush=True)
                    report['rows'].append(dict(rid=row['rid'],reference64=baseline,hashes64=hashes,short_outputs=short))
        assert json.loads((a.output/'lifecycle.json').read_text())['status']=='COMPLETE'
        report['status']='GENERATION_PASS'
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc));raise
    finally:
        if a.output.exists():(a.output/'generation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
