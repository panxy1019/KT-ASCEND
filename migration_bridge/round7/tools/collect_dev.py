"""Exact stat DEV collection. No HOLDOUT/CMMLU fitting and no hidden network."""
import hashlib
import fcntl
import json
from pathlib import Path
import sys
import time
import urllib.request
from session import session
from associate_forwards import associate


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rpc(endpoint):
    with urllib.request.urlopen('http://127.0.0.1:31000/'+endpoint,timeout=120) as r:
        return r.read().decode()


def request(root,row,rid,budget=16):
    from common import validate_answer
    payload=dict(rid=rid,input_ids=row['input_ids'],sampling_params=dict(temperature=0,
        max_new_tokens=budget,ignore_eos=True),return_logprob=True,logprob_start_len=-1,top_logprobs_num=0)
    started=time.time()
    req=urllib.request.Request('http://127.0.0.1:31000/generate',data=json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=3600) as r: answer=json.load(r)
    ids=validate_answer(answer,rid,row['input_ids'],budget,True)
    result=dict(id=rid,prompt_id=row['id'],started_unix=started,ended_unix=time.time(),
                input_tokens=len(row['input_ids']),response=answer,seconds=time.time()-started)
    with (root/'requests.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
    return ids


def associated_counts(root,counts,epoch,rows):
    metadata=[json.loads(x) for x in (root/'forward-metadata.jsonl').read_text().splitlines()]
    metadata=[x for x in metadata if x['epoch']==epoch]
    assert len(metadata)==counts.shape[0], 'STAT_METADATA_ONE_TO_ONE'
    raw=[json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines()]
    lookup={x['id']:x['response']['output_ids'] for x in raw}
    requests=[dict(input_ids=row['input_ids'],output_ids=lookup[rid]) for row,rid in rows]
    attribution=associate(metadata,requests)
    for i,step in enumerate(metadata):
        # All actual routed layers must contain exactly 6 assignments per
        # input row in this same model forward, including retained raw tails.
        assert (counts[i,1:,:].sum(-1)==len(step['input_ids'])*6).all(), 'PASS_ROUTE_COMPLETENESS'
    clean=counts[attribution['kept_indices']]
    (root/f'forward-attribution-epoch-{epoch}.json').write_text(json.dumps(attribution,indent=2)+'\n')
    return clean,attribution


def main():
    base=Path(sys.argv[1]); corpus=Path(sys.argv[2]); output=base/'uniform-dev-v1'
    lease=(base/'collection.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=[json.loads(x) for x in corpus.read_text().splitlines()]
    assert len(rows)==256
    freeze=json.loads((corpus.parent/'CORPUS_FREEZE.json').read_text())
    assert sha(corpus)==freeze['corpora']['DEV']['sha256']
    report=dict(status='RUNNING',phase='DEV_STAT',completed=0,total=256,corpus_sha256=sha(corpus))
    (base/'progress.json').write_text(json.dumps(report,indent=2))
    try:
        with session(output,base/'uniform-dev-config.json') as root:
            # Bring-up repeated request sanity before any threshold work.
            short=next(x for x in rows if x['bucket']=='short')
            a=request(root,short,'r7-dev-sanity-a',8)
            (root/'observer-off').touch()
            b=request(root,short,'r7-dev-sanity-b',8)
            (root/'observer-off').unlink()
            assert a==b,'SAME_PATH_OR_OBSERVER_PARITY'
            report['observer_on_off_parity']=True
            # Prove stat coverage before spending the entire DEV collection.
            rpc('start_expert_distribution_record')
            request(root,short,'r7-stat-pilot',8)
            rpc('stop_expert_distribution_record'); rpc('dump_expert_distribution_record')
            import torch
            pilot_files=list((output/'recorder').glob('expert_distribution_recorder_*.pt'))
            assert len(pilot_files)==1,'PILOT_RECORDER_COUNT'
            pilot=torch.load(pilot_files[0],map_location='cpu',weights_only=False)['logical_count']
            assert pilot.ndim==3 and tuple(pilot.shape[1:])==(27,64),'PILOT_RECORDER_SHAPE'
            pilot,attribution=associated_counts(root,pilot,1,[(short,'r7-stat-pilot')])
            totals=pilot.sum(dim=(0,2)).tolist()
            expected_pilot=(len(short['input_ids'])+7)*6
            assert totals[0]==0 and all(int(v)==expected_pilot for v in totals[1:]),('PILOT_RECORDER_COMPLETENESS',totals)
            pilot_files[0].rename(output/'pilot-stat.pt')
            report['pilot_attribution']=attribution
            rpc('start_expert_distribution_record')
            for i,row in enumerate(rows):
                request(root,row,f'r7-dev-{i:03d}',16)
                report.update(completed=i+1,prompt_id=row['id'],updated_unix=time.time())
                (base/'progress.json').write_text(json.dumps(report,indent=2))
                print(json.dumps(report),flush=True)
            rpc('stop_expert_distribution_record'); rpc('dump_expert_distribution_record')
        import torch
        files=list((output/'recorder').glob('expert_distribution_recorder_*.pt'))
        assert len(files)==1,'RECORDER_ARTIFACT_COUNT'
        data=torch.load(files[0],map_location='cpu',weights_only=False)
        counts=data['logical_count']
        assert counts.ndim==3 and tuple(counts.shape[1:])==(27,64)
        assert torch.isfinite(counts).all() and (counts>=0).all()
        assert counts[:,0,:].sum()==0,'DENSE_LAYER_BUDGET_POLLUTION'
        # Each request has prefill plus 15 decode forwards (last generated token
        # is not consumed); exact all-layer routing count catches circular-buffer
        # truncation and missing recorder calls.
        clean,attribution=associated_counts(output,counts,2,[(row,f'r7-dev-{i:03d}') for i,row in enumerate(rows)])
        expected=sum(len(x['input_ids'])+15 for x in rows)*6
        layer_totals=clean.sum(dim=(0,2)).tolist()
        assert all(int(v)==expected for v in layer_totals[1:]),('RECORDER_COMPLETENESS',expected,layer_totals)
        target=base/'SCHED_DEV_activation_stats.pt'
        torch.save(dict(**data,logical_count_raw=counts,forward_attribution=attribution),base/'SCHED_DEV_activation_raw.pt')
        data['logical_count']=clean
        torch.save(data,target)
        report.update(status='DEV_STAT_COMPLETE',activation_sha256=sha(target),
            logical_count_shape=list(clean.shape),layer_totals=layer_totals,forward_attribution=attribution,
            expected_assignments_per_layer=expected,finite=True)
    except BaseException as e:
        report.update(status='FAIL',error=repr(e));raise
    finally:
        (base/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        (base/'dev-stat-summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
