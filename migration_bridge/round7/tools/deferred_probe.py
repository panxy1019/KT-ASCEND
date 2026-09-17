"""Selected scheduler D1 repeat/restart/parity and actual Deferred execution.

Uses the already-qualified default-off trace seam. No pending-buffer protocol is
reimplemented here. Timed performance and quality do not enable this trace.
"""
import argparse
import json
from pathlib import Path


def audit_trace(records,terminal=26):
    submit=[x for x in records if x['event']=='submit']
    assert submit,'DEFERRED_TRACE_EMPTY'
    last=[x for x in submit if x['layer_idx']==terminal]
    assert last,'TERMINAL_LAYER_NOT_EXERCISED'
    assert all(x['max_deferred_experts_per_token']==0 and x['deferred_cpu_routes']==0
               and not x['outgoing_pending'] for x in last),'TERMINAL_DEFERRED_NOT_ZERO'
    deferred=sum(x['deferred_cpu_routes'] for x in submit)
    assert deferred>0,'DEFERRED_D1_NOT_EXECUTED'
    assert all(x['deferred_cpu_routes']<=x['rows'] for x in submit),'D1_BUDGET_EXCEEDED'
    return dict(submit_events=len(submit),deferred_cpu_routes=deferred,
        immediate_cpu_routes=sum(x['immediate_cpu_routes'] for x in submit),
        terminal_deferred_zero=True,actual_execution=True)


def main():
    from collect_dev import request
    from static_qualification import config,sha,write
    from session import session
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--holdout',type=Path,required=True)
    p.add_argument('--profile',choices=['uniform','frequency'],required=True)
    p.add_argument('--dynamic-base',action='store_true')
    p.add_argument('--threshold-freeze',type=Path)
    p.add_argument('--threshold-ack',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    freeze=json.loads((a.holdout.parent/'CORPUS_FREEZE.json').read_text())
    assert sha(a.holdout)==freeze['corpora']['HOLDOUT']['sha256']
    rows=[json.loads(x) for x in a.holdout.read_text().splitlines()]
    chosen=[next(x for x in rows if x['bucket']==bucket) for bucket in ('short','medium','long','structured')]
    cfg=json.loads(config(a.base,a.profile,True).read_text())
    cfg.update(deferred=1,deferred_trace=True)
    if a.dynamic_base:
        from frozen_threshold import load_threshold
        frozen,ack=load_threshold(a.threshold_freeze,a.threshold_ack,a.base/'profiles'/f'{a.profile}.json')
        cfg.update(dynamic=True,threshold=frozen['threshold'])
    cfgpath=a.output/'config.json';write(cfgpath,cfg)
    write(a.output/'protocol.json',dict(profile=a.profile,dynamic_base=a.dynamic_base,
        threshold_commit_ack=ack if a.dynamic_base else None,
        ids=[x['id'] for x in chosen],requests=32,repeats=4,prefix=[4,8,12],budget=16,
        holdout_sha256=sha(a.holdout),diagnostic_only=True))
    refs={};root=a.output/'process-a'
    with session(root,cfgpath):
        for row in chosen:
            ids=request(root,row,f'd1-{row["id"]}-ref',16);refs[row['id']]=ids
            for r in range(4):assert request(root,row,f'd1-{row["id"]}-r{r}',16)==ids,'D1_NONDETERMINISM'
            for n in (4,8,12):assert request(root,row,f'd1-{row["id"]}-p{n}',n)==ids[:n],'D1_PREFIX_MISMATCH'
    structures=[json.loads(x) for x in (root/'structure.jsonl').read_text().splitlines()]
    assert len(structures)==26 and next(x for x in structures if x['layer']==26)['resolved_deferred_budget']==0
    trace=[json.loads(x) for x in (root/'deferred-trace.jsonl').read_text().splitlines()]
    assert len(trace)<1000000,'DEFERRED_TRACE_TRUNCATED'
    result=audit_trace(trace)
    # Replay the same request sequence to preserve request-dependent Dynamic
    # resident history. Then observer-OFF parity repeats it independently.
    for label,observer in [('restart',True),('observer-off',False)]:
        next_cfg=cfg.copy();next_cfg.update(observer=observer,deferred_trace=observer)
        nextpath=a.output/f'{label}-config.json';write(nextpath,next_cfg)
        with session(a.output/label,nextpath) as rroot:
            for row in chosen:
                assert request(rroot,row,f'd1-{label}-{row["id"]}-ref',16)==refs[row['id']],'D1_RESTART_OR_OBSERVER_PARITY'
                for r in range(4):request(rroot,row,f'd1-{label}-{row["id"]}-r{r}',16)
                for n in (4,8,12):request(rroot,row,f'd1-{label}-{row["id"]}-p{n}',n)
    result.update(status='PASS',same_process=True,prefix=True,restart=True,observer_parity=True,
        finite=True,mask_mapping_structure=True,requests_per_process=32,processes=3,
        dynamic_base=a.dynamic_base,default_deferred_not_selected=True)
    write(a.output/'summary.json',result)


if __name__=='__main__':main()
