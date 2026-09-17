"""Frozen-threshold held-out Dynamic routing audit; never timed performance."""
import argparse
import json
from pathlib import Path


def main():
    import torch
    from static_qualification import config,sha,write
    from session import session
    from collect_dev import request,rpc,associated_counts
    from dynamic_probe import lines,verify_updates
    from dynamic_route_summary import summarize_decode,correlate_updates
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--holdout',type=Path,required=True)
    p.add_argument('--profile',choices=['uniform','frequency'],required=True)
    p.add_argument('--threshold-freeze',type=Path,required=True)
    p.add_argument('--threshold-ack',type=Path,required=True)
    p.add_argument('--probe-summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    freeze=json.loads((a.holdout.parent/'CORPUS_FREEZE.json').read_text())
    assert sha(a.holdout)==freeze['corpora']['HOLDOUT']['sha256'],'HOLDOUT_IDENTITY'
    from frozen_threshold import load_threshold
    selected,ack=load_threshold(a.threshold_freeze,a.threshold_ack,a.base/'profiles'/f'{a.profile}.json')
    assert json.loads(a.probe_summary.read_text())['status']=='PASS','MIGRATION_SYSTEM_PREDECESSOR'
    rows=lines(a.holdout);assert len(rows)==128
    cfg=json.loads(config(a.base,a.profile,True).read_text())
    cfg.update(dynamic=True,threshold=selected['threshold'],recorder=True)
    cfgpath=a.output/'config.json';write(cfgpath,cfg)
    write(a.output/'protocol.json',dict(holdout_sha256=sha(a.holdout),
        threshold_commit_ack=ack,
        threshold_freeze_sha256=sha(a.threshold_freeze),threshold=selected['threshold'],
        budget=16,observer=True,formal_performance=False,
        scope='INDEPENDENT_ROUTING_NOT_THRESHOLD_FITTING'))
    root=a.output/'process'
    with session(root,cfgpath):
        rpc('start_expert_distribution_record')
        for i,row in enumerate(rows):
            request(root,row,f'dynamic-holdout-{i:03d}',16)
            write(a.output/'progress.json',dict(completed=i+1,total=128))
        rpc('stop_expert_distribution_record');rpc('dump_expert_distribution_record')
    files=list((root/'recorder').glob('expert_distribution_recorder_*.pt'))
    assert len(files)==1,'RECORDER_COUNT'
    counts=torch.load(files[0],map_location='cpu',weights_only=False)['logical_count']
    _,attribution=associated_counts(root,counts,1,[(r,f'dynamic-holdout-{i:03d}') for i,r in enumerate(rows)])
    metadata=[x for x in lines(root/'forward-metadata.jsonl') if x['epoch']==1]
    kept={(metadata[i]['epoch'],metadata[i]['forward_id']) for i in attribution['kept_indices']}
    decode_keys={(metadata[i]['epoch'],metadata[i]['forward_id']) for i in attribution['decode_indices']}
    updates=lines(root/'generation-audit.jsonl')
    profile=json.loads((a.base/'profiles'/f'{a.profile}.json').read_text())
    verify_updates(updates,{r['layer']:len(r['npu_ids']) for r in profile['rows']})
    assert updates,'HOLDOUT_NEVER_TRIGGERED'
    metrics=summarize_decode(lines(root/'routing.jsonl'),kept,decode_keys=decode_keys)
    migration=correlate_updates(updates,lines(root/'requests.jsonl'))
    write(a.output/'summary.json',dict(status='PASS',requests=128,threshold=selected['threshold'],
        decode_routing=metrics,forward_attribution=attribution,
        migration=migration,
        observer_on=True,formal_performance=False,
        limitation='Update diagnostic readbacks and synchronization overhead excluded from performance conclusions.'))


if __name__=='__main__':main()
