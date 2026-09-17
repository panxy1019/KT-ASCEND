"""DEV-only triggered migration and serial-generation qualification.

This diagnostic runner never selects threshold on HOLDOUT/CMMLU. It records
official selected experts and exact copied weights through default-off hooks.
The auditor's synchronization overhead is excluded from performance evaluation.
"""
import argparse
import json
from pathlib import Path


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def state_at_boundary(root):
    state={}
    for x in lines(root/'generation-consumption.jsonl'):
        state[x['layer']]=dict(generation=x['generation'],resident=x['resident'])
    return state


def verify_updates(records,slots):
    from contracts import validate_generation
    for x in records:
        validate_generation(x['weights'],x['mapping'],x['mask'],x['wrapper_mask'])
        assert x['weights']==x['prior_generation']+1,'NONMONOTONIC_GENERATION'
        assert len(x['selected'])==slots[x['layer']],'LAYER_SLOT_COUNT_CHANGED'
        assert x['weight_identity_exact'] and x['mask_mapping_wrapper_exact'],'MIGRATION_IDENTITY'


def main():
    from collect_dev import request
    from session import session
    from static_qualification import config,sha,write
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--dev',type=Path,required=True)
    p.add_argument('--profile',choices=['uniform','frequency'],required=True)
    p.add_argument('--threshold',type=int,choices=[256,512,1024],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    freeze=json.loads((a.dev.parent/'CORPUS_FREEZE.json').read_text())
    assert sha(a.dev)==freeze['corpora']['DEV']['sha256'],'DEV_IDENTITY'
    assert json.loads((a.base/f'{a.profile}-system-summary.json').read_text())['status']=='PASS'
    rows=[json.loads(x) for x in a.dev.read_text().splitlines()]
    short=next(x for x in rows if x['bucket']=='short' and len(x['input_ids'])<a.threshold)
    # Choose two distinct pre-registered prompt families before any results.
    triggered=[x for x in rows if len(x['input_ids'])>=a.threshold]
    first=next(x for x in triggered if x['family']=='code')
    second=next(x for x in triggered if x['family']=='Chinese')
    profile=json.loads((a.base/'profiles'/f'{a.profile}.json').read_text())
    slots={r['layer']:len(r['npu_ids']) for r in profile['rows']}
    cfg=json.loads(config(a.base,a.profile,True).read_text());cfg.update(dynamic=True,threshold=a.threshold)
    cfgpath=a.output/'diagnostic-config.json';write(cfgpath,cfg)
    write(a.output/'protocol.json',dict(threshold=a.threshold,dev_sha256=sha(a.dev),
        profile=a.profile,profile_sha256=sha(a.base/'profiles'/f'{a.profile}.json'),
        sequence=[short['id'],first['id'],first['id'],second['id'],short['id']],
        diagnostic_only=True,concurrency_qualified=False))
    root=a.output/'process-a';requests=[]
    with session(root,cfgpath):
        request(root,short,'no-trigger-initial',16)
        assert not lines(root/'generation-audit.jsonl'),'SHORT_REQUEST_TRIGGERED_UPDATE'
        initial=state_at_boundary(root)
        assert len(initial)==26 and all(x['generation']==0 for x in initial.values())
        assert all(initial[r['layer']]['resident']==r['npu_ids'] for r in profile['rows'])
        first_ids=request(root,first,'trigger-a-first',16)
        first_state=state_at_boundary(root)
        first_updates=lines(root/'generation-audit.jsonl')
        assert first_updates,'DYNAMIC_DID_NOT_TRIGGER'
        verify_updates(first_updates,slots)
        updated_layers={x['layer'] for x in first_updates}
        # Zero-slot layers may still enter the official full-device prefill
        # and publish an empty resident generation. This is not a budget error.
        assert {l for l,n in slots.items() if n>0}<=updated_layers<=set(slots),'TRIGGER_LAYER_COVERAGE'
        repeated=request(root,first,'trigger-a-repeat',16)
        assert first_ids==repeated,'TRIGGERED_SAME_PATH_NONDETERMINISM'
        repeat_state=state_at_boundary(root)
        assert all(first_state[l]['resident']==repeat_state[l]['resident'] for l in slots),'SELECTION_NONDETERMINISM'
        second_ids=request(root,second,'trigger-b-workload-shift',16)
        second_state=state_at_boundary(root)
        changed=[l for l in slots if repeat_state[l]['resident']!=second_state[l]['resident']]
        request(root,short,'no-trigger-after-update',16)
        assert state_at_boundary(root)==second_state,'NO_TRIGGER_MUTATED_STATE'
        updates=lines(root/'generation-audit.jsonl');verify_updates(updates,slots)
        requests=lines(root/'requests.jsonl')
    # Fresh process starts from frozen static mask, not prior Dynamic state.
    restart=a.output/'process-b'
    with session(restart,cfgpath):
        request(restart,short,'restart-initial-short',16)
        restarted=state_at_boundary(restart)
        assert restarted==initial,'PERSISTED_DYNAMIC_STATE'
        assert request(restart,first,'restart-trigger-a',16)==first_ids,'TRIGGER_RESTART_MISMATCH'
        restart_updates=lines(restart/'generation-audit.jsonl');verify_updates(restart_updates,slots)
        assert [(x['layer'],x['selected']) for x in restart_updates]==[
            (x['layer'],x['selected']) for x in first_updates],'RESTART_SELECTION_MISMATCH'
    # Same official path without diagnostic hooks must retain output trajectory.
    off=cfg.copy();off['observer']=False;offpath=a.output/'observer-off-config.json';write(offpath,off)
    with session(a.output/'observer-off',offpath) as r:
        request(r,short,'off-initial-short',16)
        assert request(r,first,'off-trigger-a',16)==first_ids,'DYNAMIC_OBSERVER_PERTURBATION'
        request(r,first,'off-repeat-a',16)
        assert request(r,second,'off-trigger-b',16)==second_ids,'DYNAMIC_OBSERVER_PERTURBATION'
    write(a.output/'summary.json',dict(status='PASS' if changed else 'WORKLOAD_SHIFT_NOT_OBSERVED',
        threshold=a.threshold,no_trigger=True,trigger=True,weight_identity_exact=True,
        generation_consistency=True,deterministic_selection=True,restart=True,observer_parity=True,
        shift_changed_layers=changed,diagnostic_updates=len(updates),requests=requests,
        diagnostic_migration_bytes=sum(x['migration_bytes'] for x in updates),
        boundary='SERIAL_TP1_BATCH1_ONLY',
        limitation='Diagnostic update duration includes exact-weight readbacks; do not use as production migration latency.'))


if __name__=='__main__':main()
