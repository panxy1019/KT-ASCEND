"""Observer-OFF serial ABBA performance, preserving all raw SSE responses.

First output event is a client-observed upper bound on TTFT (SSE buffering may
delay delivery). Decode throughput uses first-to-final output delivery, excluding
first token. Never substitute total-request tokens/sec for decode throughput.
No diagnostic model hooks, weight readbacks, or migration-audit syncs are enabled.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request


def write(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def summarize_events(events,budget):
    from math import isfinite
    outputs=[(t,x) for t,x in events if x.get('output_ids')]
    if not outputs:raise ValueError('NO_STREAM_OUTPUT')
    answer=outputs[-1][1]
    ids=answer['output_ids']
    if len(ids)!=budget:raise ValueError('TOKEN_BUDGET')
    lp=answer['meta_info']['output_token_logprobs']
    if len(lp)!=budget or [x[1] for x in lp]!=ids:raise ValueError('STREAM_TOKEN_IDENTITY')
    if not all(x[0] is not None and isfinite(float(x[0])) for x in lp):
        raise ValueError('NONFINITE_OUTPUT')
    if answer['meta_info']['finish_reason']['type'] not in ('length','stop'):
        raise ValueError('REQUEST_FINISH')
    ttft=outputs[0][0];finish=outputs[-1][0]
    # A single coalesced event cannot support a separate decode measurement.
    decode=finish-ttft
    return dict(ttft_client_seconds=ttft,final_output_client_seconds=finish,
        decode_client_seconds=decode if decode>0 else None,
        decode_tokens_per_second=(budget-1)/decode if decode>0 else None,
        output_event_count=len(outputs),output_ids=ids,final_meta=answer['meta_info'],
        timing_scope='CLIENT_SSE_UPPER_BOUND_TTFT_NOT_INTERNAL_PREFILL')


def measured_request(root,row,rid,budget):
    payload=dict(rid=rid,input_ids=row['input_ids'],stream=True,
        sampling_params=dict(temperature=0,max_new_tokens=budget,ignore_eos=True),
        return_logprob=True,logprob_start_len=-1,top_logprobs_num=0)
    req=urllib.request.Request('http://127.0.0.1:31000/generate',
        data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    write(root/f'{rid}-input.json',dict(status='STARTING',payload=payload,started_unix=time.time()))
    started_unix=time.time();started=time.perf_counter();events=[]
    with urllib.request.urlopen(req,timeout=3600) as response:
        for raw in response:
            received=time.perf_counter()-started
            line=raw.decode().strip()
            if not line.startswith('data:'):continue
            body=line[5:].strip()
            if body=='[DONE]':break
            value=json.loads(body)
            if 'error' in value:raise RuntimeError(value)
            events.append((received,value))
    total=time.perf_counter()-started
    result=summarize_events(events,budget)
    assert result['final_meta']['id']==rid,'REQUEST_CORRELATION'
    assert result['final_meta']['prompt_tokens']==len(row['input_ids']),'INPUT_IDENTITY'
    result.update(request_seconds=total,input_tokens=len(row['input_ids']),prompt_id=row['id'],
        started_unix=started_unix,ended_unix=time.time())
    write(root/f'{rid}.json',dict(payload=payload,events=events,summary=result))
    return result


def main():
    from static_qualification import config,sha
    from session import session,memory
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--left',choices=['uniform','frequency'],required=True)
    p.add_argument('--right',choices=['uniform','frequency','dynamic','deferred'],required=True)
    p.add_argument('--threshold',type=int)
    p.add_argument('--threshold-freeze',type=Path)
    p.add_argument('--threshold-ack',type=Path)
    p.add_argument('--dynamic-base',action='store_true',help='Evaluate D1 on a selected Dynamic scheduler')
    p.add_argument('--host-observer',type=Path,help='Host-issued samples.jsonl visible through the owned mount')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    freeze=json.loads((a.corpus.parent/'CORPUS_FREEZE.json').read_text())
    assert sha(a.corpus)==freeze['corpora']['HOLDOUT']['sha256'],'HOLDOUT_HASH'
    ack=None
    if a.right=='dynamic' or a.dynamic_base:
        from frozen_threshold import load_threshold
        selected,ack=load_threshold(a.threshold_freeze,a.threshold_ack,a.base/'profiles'/f'{a.left}.json')
        if a.threshold is not None and a.threshold!=selected['threshold']:
            raise ValueError('CLI_THRESHOLD_DIFFERS_FROM_COMMITTED_CANDIDATE')
        a.threshold=selected['threshold']
    elif a.threshold is not None or a.threshold_freeze is not None or a.threshold_ack is not None:
        raise ValueError('THRESHOLD_ARGUMENTS_WITHOUT_DYNAMIC_LANE')
    rows=[json.loads(x) for x in a.corpus.read_text().splitlines()]
    # Deterministic first member of each pre-registered bucket, never selected
    # for observed scores or latency. Record IDs before the first measurement.
    targets=[(bucket,next(x for x in rows if x['bucket']==bucket),budget)
             for bucket,budget in [('short',256),('medium',512),('long',1024)]]
    write(a.output/'protocol.json',dict(order=['left','right','right','left'],blocks=3,
        left=a.left,right=a.right,threshold=a.threshold,dynamic_base=a.dynamic_base,observer=False,
        threshold_commit_ack=ack,
        targets=[dict(bucket=b,id=r['id'],input_tokens=len(r['input_ids']),decode=n) for b,r,n in targets],
        holdout_sha256=sha(a.corpus),host_observer=str(a.host_observer) if a.host_observer else None,
        internal_cpu_wait='NOT_INSTRUMENTED_IN_TIMED_PATH'))
    measurements=[]
    for bucket,row,budget in targets:
        for block in range(3):
            for turn,side in enumerate(('left','right','right','left')):
                mode=a.left if side=='left' else a.right
                underlying=mode if mode in ('uniform','frequency') else a.left
                cfg=json.loads(config(a.base,underlying,False).read_text())
                if mode=='dynamic' or a.dynamic_base:
                    assert a.threshold is not None and a.threshold>0,'FROZEN_THRESHOLD_REQUIRED'
                    cfg.update(dynamic=True,threshold=a.threshold)
                if mode=='deferred':cfg['deferred']=1
                cfgpath=a.output/f'{bucket}-{block}-{turn}-config.json';write(cfgpath,cfg)
                root=a.output/f'{bucket}-{block}-{turn}-{mode}'
                with session(root,cfgpath):
                    from collect_dev import request
                    request(root,row,f'warmup-{bucket}-{block}-{turn}',8)
                    before=memory()
                    result=measured_request(root,row,f'measure-{bucket}-{block}-{turn}',budget)
                    result.update(memory_before=before,memory_after=memory())
                measurements.append(dict(bucket=bucket,block=block,turn=turn,side=side,mode=mode,**result))
                write(a.output/'progress.json',dict(completed=len(measurements),total=36,measurements=measurements))
    summary={}
    for bucket,_,_ in targets:
        sides={side:[x for x in measurements if x['bucket']==bucket and x['side']==side]
               for side in ('left','right')}
        med={side:statistics.median(x['request_seconds'] for x in group) for side,group in sides.items()}
        rates={side:[x['decode_tokens_per_second'] for x in group
                     if x['decode_tokens_per_second'] is not None] for side,group in sides.items()}
        rate_medians={side:statistics.median(v) if v else None for side,v in rates.items()}
        summary[bucket]=dict(median_e2e_seconds=med,median_decode_tokens_per_second=rate_medians,
            latency_improvement_percent=100*(1-med['right']/med['left']),
            throughput_improvement_percent=(100*(rate_medians['right']/rate_medians['left']-1)
                if all(v is not None for v in rate_medians.values()) else None))
    write(a.output/'summary.json',dict(status='COMPLETE',observer=False,serial=True,
        measurements=measurements,workloads=summary,
        limitation='Internal CPU wait and exact device prefill timing require separate untimed telemetry.'))
    if a.host_observer:
        from host_resource_observer import assess_window
        # The observer samples once every5 seconds. This wait occurs after all
        # timing and serving reaping, never inside a measured request.
        time.sleep(10)
        samples=[json.loads(x) for x in a.host_observer.read_text().splitlines()]
        windows=[dict(bucket=x['bucket'],block=x['block'],turn=x['turn'],
            **assess_window(samples,x['started_unix'],x['ended_unix'])) for x in measurements]
        resource=dict(status='PASS' if all(x['formal_eligible'] for x in windows) else 'CONTAMINATED_OR_UNOBSERVED',windows=windows)
    else:resource=dict(status='UNOBSERVED',formal_eligible=False)
    write(a.output/'host-isolation.json',resource)


if __name__=='__main__':main()
