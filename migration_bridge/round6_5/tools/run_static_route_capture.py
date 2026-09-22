"""Bounded natural-history diagnostic; parity with uninstrumented static run.

Uses existing opt-in stagewise capture. Never used for timed performance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
from static_session import session


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--determinism',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    det=json.loads(a.determinism.read_text())
    if det['status']!='STATIC_COMPACT_DETERMINISM_PASS':raise ValueError('DET_PREDECESSOR')
    if det['profile_sha256']!=hashlib.sha256(a.profile.read_bytes()).hexdigest():raise ValueError('PROFILE_DRIFT')
    corpus=json.loads(Path('/workspace/results/corpora-v2/s6.json').read_text())
    prompts={r['id']:r for r in corpus['prompts']}
    report=dict(status='RUNNING',requests=[],forward_budget=36,
                instrumentation='EXISTING_DEFAULT_OFF_STAGEWISE',timing_qualified=False)
    os.environ['KT_R65_STAGEWISE']='1'
    try:
        with session(a.output,a.profile,capture=True) as root:
            for pid in ('s6_en_01','s6_zh_01','s6_math_01','s6_json_01'):
                rid='r65-route-'+pid
                payload=dict(rid=rid,input_ids=prompts[pid]['input_ids'],sampling_params=dict(
                    temperature=0,max_new_tokens=8,ignore_eos=True),return_logprob=True,logprob_start_len=-1)
                request=urllib.request.Request('http://127.0.0.1:31000/generate',
                    data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=900) as r:answer=json.load(r)
                report['requests'].append(dict(payload=payload,response=answer))
                (root/'requests.json').write_text(json.dumps(report,indent=2)+'\n')
                if answer['meta_info']['id']!=rid or answer['output_ids']!=det['references'][pid][:8]:
                    raise ValueError('OBSERVER_ON_OFF_PARITY')
        entries=[json.loads(x) for x in (a.output/'index.jsonl').read_text().splitlines()]
        if len(entries)!=36 or [r['sequence'] for r in entries]!=list(range(36)):
            raise ValueError('DIAGNOSTIC_FORWARD_COUNT')
        for i,r in enumerate(entries):
            expected=report['requests'][i//9]['payload']['rid']
            if r['batch']['request_ids']!=[expected]:raise ValueError('DIAGNOSTIC_RID')
        report.update(status='ROUTE_CAPTURE_PARITY_PASS',index=entries)
    except BaseException as exc:report.update(status='FAILED',exception=repr(exc));raise
    finally:
        if a.output.exists():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
