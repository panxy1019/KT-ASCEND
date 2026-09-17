"""Round7A independent gates, full HOLDOUT and frozen CMMLU evaluator."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request
from collect_dev import request,rpc
from contracts import validate_mask,quality_gate
from session import session


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def config(base,name,observer):
    cfg=json.loads((base/'uniform-dev-config.json').read_text())
    cfg.update(recorder=False,observer=observer,
        provenance=str(base/'provenance.json'),
        profile=str(base/'profiles'/f'{name}.json'),activation_path=str(base/'profiles'/f'{name}.pt'))
    path=base/f'{name}-observer-{int(observer)}.json';write(path,cfg)
    profile=json.loads(Path(cfg['profile']).read_text());validate_mask(profile['rows'])
    assert sha(Path(cfg['activation_path']))==profile['activation_sha256']
    return path


def compact(base,name,rows):
    target=base/f'{name}-compact-v1'
    refs={}
    short=rows[:4]
    cfg=config(base,name,True)
    with session(target,cfg) as root:
        for row in short:
            trajectory=request(root,row,f'{name}-{row["id"]}-ref',16)
            refs[row['id']]=trajectory
            for repeat in range(4):
                assert request(root,row,f'{name}-{row["id"]}-r{repeat}',16)==trajectory,'SAME_PATH_NONDETERMINISM'
            for length in (4,8,12):
                assert request(root,row,f'{name}-{row["id"]}-p{length}',length)==trajectory[:length],'PREFIX_MISMATCH'
    structure=[json.loads(x) for x in (target/'structure.jsonl').read_text().splitlines()]
    assert len(structure)==26 and {x['layer'] for x in structure}==set(range(1,27))
    # Independent process and observer parity use the exact captured inputs.
    with session(base/f'{name}-restart-off-v1',config(base,name,False)) as root:
        for row in short:
            assert request(root,row,f'{name}-{row["id"]}-restart',16)==refs[row['id']],'RESTART_OR_OBSERVER_PARITY'
    result=dict(status='PASS',requests=32,restart_requests=4,same_path=True,prefix=True,
        restart=True,observer_on_off_parity=True,finite=True,mask_mapping_ownership=True,
        weight_layout_audit_sha256=sha(target/'structure.jsonl'),trajectories=refs,
        limitation='Operator representative safety is a separate gate; this report is not full scheduler qualification.')
    write(base/f'{name}-system-summary.json',result)
    return result


def holdout(base,name,rows):
    target=base/f'{name}-holdout-v1'
    cfg=json.loads(config(base,name,True).read_text());cfg['recorder']=True
    cfgpath=base/f'{name}-holdout-config.json';write(cfgpath,cfg)
    with session(target,cfgpath) as root:
        rpc('start_expert_distribution_record')
        for i,row in enumerate(rows):
            request(root,row,f'{name}-holdout-{i:03d}',16)
            write(base/'static-progress.json',dict(phase='HOLDOUT',profile=name,completed=i+1,total=128))
        rpc('stop_expert_distribution_record');rpc('dump_expert_distribution_record')
    # Production overlap can evaluate an unused final token. Exactly associate
    # every forward before computing routing rates, just as for DEV fitting.
    import torch
    from collect_dev import associated_counts
    rawfiles=list((target/'recorder').glob('expert_distribution_recorder_*.pt'))
    assert len(rawfiles)==1,'HOLDOUT_RECORDER_COUNT'
    counts=torch.load(rawfiles[0],map_location='cpu',weights_only=False)['logical_count']
    _,attribution=associated_counts(target,counts,1,[(r,f'{name}-holdout-{i:03d}') for i,r in enumerate(rows)])
    metadata=[json.loads(x) for x in (target/'forward-metadata.jsonl').read_text().splitlines()]
    metadata=[x for x in metadata if x['epoch']==1]
    kept={(metadata[i]['epoch'],metadata[i]['forward_id']) for i in attribution['kept_indices']}
    metrics=dict(assignments=0,npu_hits=0,cpu_hits=0,total_weight=0.0,cpu_weight=0.0,tokens=0,fully_npu_tokens=0)
    layers={};forward_rows={}
    for line in (target/'routing.jsonl').read_text().splitlines():
        r=json.loads(line)
        context=r['forward_context']
        assert context,'UNATTRIBUTED_ROUTING_FORWARD'
        if (context['epoch'],context['forward_id']) not in kept:continue
        key=(context['epoch'],context['forward_id'])
        by_layer=forward_rows.setdefault(key,{})
        assert r['layer'] not in by_layer,'DUPLICATE_LAYER_ROUTING'
        by_layer[r['layer']]=r['fully_npu_row_flags']
        for k in metrics:metrics[k]+=r[k]
        l=layers.setdefault(str(r['layer']),dict(assignments=0,npu_hits=0,cpu_hits=0))
        for k in l:l[k]+=r[k]
    metrics.update(npu_hit_rate=metrics['npu_hits']/metrics['assignments'],
        cpu_route_rate=metrics['cpu_hits']/metrics['assignments'],
        cpu_weight_mass=metrics['cpu_weight']/metrics['total_weight'],per_layer=layers)
    assert metrics['assignments']==sum(len(r['input_ids'])+15 for r in rows)*6*26,'HOLDOUT_ROUTE_COMPLETENESS'
    model_rows=0;fully_npu_model_rows=0
    for by_layer in forward_rows.values():
        assert set(by_layer)==set(range(1,27)),'FORWARD_LAYER_INCOMPLETENESS'
        sizes={len(flags) for flags in by_layer.values()}
        assert len(sizes)==1,'FORWARD_ROW_SHAPE_DRIFT'
        model_rows+=next(iter(sizes))
        fully_npu_model_rows+=sum(all(flags[i] for flags in by_layer.values()) for i in range(next(iter(sizes))))
    metrics.update(model_token_rows=model_rows,fully_npu_model_token_rows=fully_npu_model_rows,
        fully_npu_model_token_rate=fully_npu_model_rows/model_rows,
        model_token_touches_cpu_fraction=1-fully_npu_model_rows/model_rows,
        layer_row_scope='The legacy tokens/fully_npu_tokens counters sum layer rows, not unique model tokens.')
    raw=[json.loads(x) for x in (target/'requests.jsonl').read_text().splitlines()]
    # Observer incurs synchronization. Do not call this isolated performance.
    result=dict(status='PASS',routing=metrics,observer_on=True,formal_performance=False,
        request_latency_median_seconds=statistics.median(x['seconds'] for x in raw),
        requests=128,budget=16,raw_sha256=sha(target/'requests.jsonl'),forward_attribution=attribution)
    write(base/f'{name}-holdout-summary.json',result)
    return result


def quality(base,name,configuration=None,target=None,summary_path=None):
    from common import validate_answer,score_letters
    manifest=Path('/workspace/results/r65-r67-serial-v5/cmmlu.json')
    assert sha(manifest)=='bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04'
    rows=json.loads(manifest.read_text())['rows'];results=[]
    with session(target or base/f'{name}-cmmlu-v1',configuration or config(base,name,False)) as root:
        for i,row in enumerate(rows):
            rid=f'r7-{name}-cmmlu-{row["id"]}'
            payload=dict(rid=rid,input_ids=row['input_ids'],sampling_params=dict(
                temperature=0,max_new_tokens=1,ignore_eos=False,stop=row['stop']),return_logprob=True,
                logprob_start_len=-1,top_logprobs_num=0,token_ids_logprob=row['candidate_ids'])
            started=time.time()
            req=urllib.request.Request('http://127.0.0.1:31000/generate',data=json.dumps(payload).encode(),
                headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=3600) as response:answer=json.load(response)
            validate_answer(answer,rid,row['input_ids'],1,True)
            candidates=answer['meta_info']['output_token_ids_logprobs'];assert len(candidates)==1
            predicted,logits=score_letters(candidates[0],row['candidate_ids'])
            record=dict(id=row['id'],subject=row['subject'],predicted=predicted,
                        correct=predicted==row['answer'],candidate_logprobs=logits,seconds=time.time()-started)
            results.append(record)
            with (root/'raw.jsonl').open('a') as f:f.write(json.dumps(dict(payload=payload,response=answer,score=record,
                started_unix=started,ended_unix=time.time()))+'\n')
            write(base/'static-progress.json',dict(phase='CMMLU',profile=name,completed=i+1,total=256,
                correct=sum(x['correct'] for x in results)))
    correct=sum(x['correct'] for x in results)
    result=dict(status='PASS' if quality_gate(correct,0) else 'QUALITY_FAIL',correct=correct,
        count=256,invalid=0,delta_pp=100*(correct-174)/256,rows=results,manifest_sha256=sha(manifest))
    write(summary_path or base/f'{name}-quality-summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--holdout',type=Path,required=True)
    p.add_argument('--profile',choices=['uniform','frequency'],required=True)
    p.add_argument('--stage',choices=['compact','holdout','quality'],required=True)
    a=p.parse_args();rows=[json.loads(x) for x in a.holdout.read_text().splitlines()]
    freeze=json.loads((a.holdout.parent/'CORPUS_FREEZE.json').read_text())
    assert len(rows)==128 and sha(a.holdout)==freeze['corpora']['HOLDOUT']['sha256']
    if a.stage=='compact':compact(a.base,a.profile,rows)
    else:
        assert json.loads((a.base/f'{a.profile}-system-summary.json').read_text())['status']=='PASS'
        if a.stage=='holdout':holdout(a.base,a.profile,rows)
        else:
            assert json.loads((a.base/'operator-representatives/summary.json').read_text())['status']=='PASS', 'OPERATOR_PREDECESSOR'
            quality(a.base,a.profile)


if __name__=='__main__':main()
