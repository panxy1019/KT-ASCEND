"""Two independent bounded campaigns; exact outputs, memory evidence not leak proof."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
from static_session import session
from run_static_performance import memory,server_info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--determinism',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();det=json.loads(a.determinism.read_text())
    if det['status']!='STATIC_COMPACT_DETERMINISM_PASS' or det.get('repeats')!=10:
        raise ValueError('FINAL_DETERMINISM_PREDECESSOR')
    if det['profile_sha256']!=hashlib.sha256(a.profile.read_bytes()).hexdigest():raise ValueError('PROFILE_DRIFT')
    corpus=json.loads(Path('/workspace/results/corpora-v2/s6.json').read_text())
    prompts={row['id']:row for row in corpus['prompts']}
    sys.path.insert(0,'/workspace/results/round6_2-prequalification/tools')
    from run_runtime_determinism import validate
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',profile_sha256=det['profile_sha256'],rows=[],samples=[],started_unix=time.time())
    try:
        for campaign in ('A','B'):
            with session(a.output/campaign,a.profile) as root:
                report['samples'].append(dict(campaign=campaign,point='LOADED',memory=memory(),server=server_info()))
                with (root/'requests.jsonl').open('x') as raw:
                    for cycle in range(3):
                        for pid in ('s6_en_01','s6_zh_01','s6_math_01'):
                            rid=f'r65-stability-{campaign}-{cycle}-{pid}'
                            payload=dict(rid=rid,input_ids=prompts[pid]['input_ids'],sampling_params=dict(
                                temperature=0,max_new_tokens=64,ignore_eos=True),return_logprob=True,logprob_start_len=-1)
                            request=urllib.request.Request('http://127.0.0.1:31000/generate',
                                data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                            start=time.monotonic()
                            with urllib.request.urlopen(request,timeout=1200) as response:answer=json.load(response)
                            tokens=validate(answer,rid,64)
                            raw.write(json.dumps(dict(payload=payload,response=answer))+'\n');raw.flush()
                            if tokens!=det['references'][pid]:raise ValueError('STABILITY_TRAJECTORY_DRIFT')
                            row=dict(campaign=campaign,cycle=cycle,prompt=pid,seconds=time.monotonic()-start,
                                token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),exact=True)
                            report['rows'].append(row)
                            report['samples'].append(dict(campaign=campaign,point=rid,memory=memory(),server=server_info()))
                            (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
            report['samples'].append(dict(campaign=campaign,point='STOPPED',memory=memory()))
        report.update(status='STABILITY_TRAJECTORIES_PASS_MEMORY_REVIEW_REQUIRED',generated_tokens=18*64)
    except BaseException as exc:report.update(status='FAILED',exception=repr(exc));raise
    finally:
        report['ended_unix']=time.time()
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
