"""Serial R64 natural-history routing, frozen DEV/HOLDOUT, bounded lookahead."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
from static_session import session
from analyze_scheduling import positions


def records(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.manifest.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if sha!='6aa4f561b7bc3c8bc31c78dbae151b38bd94ecdf14c8143026d0105bc099a187':raise ValueError('CORPUS_DRIFT')
    if json.loads(a.profile.read_text())['total_npu_experts']!=26*64:raise ValueError('R64_ORACLE_REQUIRED')
    spec=json.loads(raw);a.output.mkdir(parents=True,exist_ok=False)
    os.environ.pop('KT_R65_STAGEWISE',None);os.environ['KT_R65_ROUTE_ONLY']='1'
    sys.path.insert(0,'/workspace/results/round6_2-prequalification/tools')
    from run_runtime_determinism import validate
    report=dict(status='RUNNING',manifest_sha256=sha,requests=[],
        timing_qualified=False,history='R64_NATURAL_REFERENCE',started_unix=time.time())
    try:
        with session(a.output/'session',a.profile,capture=False) as root:
            path=root/'routing.jsonl'
            if records(path):raise ValueError('STARTUP_ROUTE_CONTAMINATION')
            sequence=0
            for prompt in spec['rows']:
                rid='r65-'+prompt['id'];budget=64
                payload=dict(rid=rid,input_ids=prompt['input_ids'],sampling_params=dict(
                    temperature=0,max_new_tokens=budget,ignore_eos=True),
                    return_logprob=True,logprob_start_len=-1)
                request=urllib.request.Request('http://127.0.0.1:31000/generate',data=json.dumps(payload).encode(),
                    headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=1200) as response:answer=json.load(response)
                validate(answer,rid,budget)
                expected=(sequence+budget+1)*26;deadline=time.monotonic()+30
                while True:
                    count=len(records(path))
                    if count==expected:break
                    if count>expected or time.monotonic()>deadline:raise ValueError('ROUTE_FORWARD_BOUNDARY')
                    time.sleep(.2)
                report['requests'].append(dict(prompt=prompt,payload=payload,response=answer,
                    sequence_start=sequence,budget=budget))
                sequence+=budget+1
                (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(dict(prompt=prompt['id'],forwards=sequence)),flush=True)
        observed=records(path)
        report['positions']=sum(1 for _ in positions(observed,report['requests']))
        report.update(status='SCHEDULING_ROUTE_CAPTURE_PASS',route_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except BaseException as exc:report.update(status='FAILED',exception=repr(exc));raise
    finally:
        report['ended_unix']=time.time()
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
