"""One scoped static profile: frozen four prompts, repeat/prefix and restart.

This gate does not claim expert identity or pairwise numerical qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0,'/workspace/results/round6_2-prequalification/tools')
from run_runtime_determinism import validate

PROMPTS=('s6_en_01','s6_zh_01','s6_math_01','s6_json_01')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(16,32),default=32)
    p.add_argument('--repeats',type=int,choices=(3,10),default=3)
    a=p.parse_args()
    corpus=json.loads(a.corpus.read_text())
    actual_sha=hashlib.sha256(json.dumps({k:v for k,v in corpus.items() if k!='sha256'},
        ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if actual_sha!=corpus['sha256']:raise ValueError('CORPUS_BYTES_DRIFT')
    if corpus['sha256']!='e51af126389507e03b58ae37e5b39c912ece14d54c1daa0ba4baea634ff8f9c5':
        raise ValueError('CORPUS_DRIFT')
    prompts={r['id']:r for r in corpus['prompts']}
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',started_unix=time.time(),rows=[],references={},
                profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),
                repeats=a.repeats,workers=a.workers)
    tools=Path(__file__).parent
    def interrupted(signum, frame): raise InterruptedError('USER_OR_GUARD_CANCEL')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for phase in ('repeat','restart'):
            root=a.output/phase
            root.mkdir()
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                sock.bind(('127.0.0.1',31000))
            command=[sys.executable,str(tools/'owned_guard.py'),'--output',str(root/'guard'),
                     '--timeout-seconds','10800','--','taskset','-c',f'0-{a.workers-1}',
                     'bash',str(tools/'launch_static.sh'),str(root),str(a.profile),str(a.workers)]
            with (root/'launcher.log').open('x') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                try:
                    deadline=time.monotonic()+1200
                    while True:
                        if child.poll() is not None: raise RuntimeError('SERVER_EXITED_BEFORE_READY')
                        if time.monotonic()>deadline: raise TimeoutError('READINESS_TIMEOUT')
                        try:
                            with urllib.request.urlopen('http://127.0.0.1:31000/health',timeout=2) as r:
                                if r.status==200: break
                        except (OSError,TimeoutError): pass
                        time.sleep(2)
                    with (root/'requests.jsonl').open('x') as raw:
                        for pid in PROMPTS:
                            for n,budget in enumerate(([64]*a.repeats+[8,16,32]) if phase=='repeat' else [64]):
                                rid=f'r65-{a.profile.stem}-{phase}-{pid}-{n}'
                                payload=dict(rid=rid,input_ids=prompts[pid]['input_ids'],
                                    sampling_params=dict(temperature=0,max_new_tokens=budget,ignore_eos=True),
                                    return_logprob=True,logprob_start_len=-1,top_logprobs_num=0)
                                req=urllib.request.Request('http://127.0.0.1:31000/generate',
                                    data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                                started=time.monotonic()
                                with urllib.request.urlopen(req,timeout=900) as response: answer=json.load(response)
                                raw.write(json.dumps(dict(payload=payload,response=answer,elapsed=time.monotonic()-started))+'\n')
                                raw.flush()
                                tokens=validate(answer,rid,budget)
                                if pid not in report['references']: report['references'][pid]=tokens
                                if tokens!=report['references'][pid][:budget]:
                                    raise ValueError('SAME_PATH_NONDETERMINISM:'+rid)
                                report['rows'].append(dict(rid=rid,exact=True))
                                (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
                                print(json.dumps(report['rows'][-1]),flush=True)
                finally:
                    if child.poll() is None: child.send_signal(signal.SIGTERM)
                    child.wait(timeout=60)
                    if not json.loads((root/'guard/reaping.json').read_text())['complete']:
                        raise RuntimeError('REAPING_FAILED')
                    if json.loads((root/'guard/result.json').read_text()).get('reason') not in ('CANCELLED','EXIT_ZERO'):
                        raise RuntimeError('OWNED_GUARD_FAILURE')
        report.update(status='STATIC_COMPACT_DETERMINISM_PASS',request_count=len(report['rows']))
    except BaseException as exc:
        report.update(status='FAILED',exception=repr(exc))
        raise
    finally:
        report['ended_unix']=time.time()
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
