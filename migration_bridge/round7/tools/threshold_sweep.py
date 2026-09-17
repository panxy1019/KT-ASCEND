"""DEV-only exploratory threshold selection, frozen before any held-out use.

Twelve pre-selected DEV cases cover short/medium/long equally. This is a DEV
screen, not formal performance qualification; final performance uses independent
HOLDOUT workloads and ABBA. Complete trigger/generation probes precede selection.
"""
import argparse
import json
from pathlib import Path
import statistics


def probe_summaries(root,prefix):
    if prefix not in ('threshold-','dynamic-dev-probe-'):
        raise ValueError('UNSUPPORTED_PROBE_PREFIX')
    paths=[root/f'{prefix}{threshold}'/'summary.json' for threshold in (256,512,1024)]
    for path in paths:
        assert json.loads(path.read_text())['status']=='PASS','TRIGGER_GENERATION_PREDECESSOR'
    return paths


def main():
    from collect_dev import request
    from static_qualification import config,sha,write
    from session import session
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--dev',type=Path,required=True)
    p.add_argument('--profile',choices=['uniform','frequency'],required=True)
    p.add_argument('--probes',type=Path,required=True,help='threshold-256/512/1024 subdirectories')
    p.add_argument('--probe-prefix',choices=['threshold-','dynamic-dev-probe-'],default='threshold-')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    freeze=json.loads((a.dev.parent/'CORPUS_FREEZE.json').read_text())
    assert sha(a.dev)==freeze['corpora']['DEV']['sha256']
    rows=[json.loads(x) for x in a.dev.read_text().splitlines()]
    selected=[r for bucket in ('short','medium','long')
              for r in [x for x in rows if x['bucket']==bucket][:4]]
    assert len(selected)==12
    candidates=[256,512,1024]
    predecessors=probe_summaries(a.probes,a.probe_prefix)
    write(a.output/'protocol.json',dict(dev_sha256=sha(a.dev),profile=a.profile,
        profile_sha256=sha(a.base/'profiles'/f'{a.profile}.json'),candidates=candidates,
        ids=[r['id'] for r in selected],decode_budget=64,observer=False,
        selector='minimum median matched-case E2E ratio; tie prefers larger threshold',
        qualification_scope='DEV_EXPLORATORY_NOT_FORMAL_ABBA',
        predecessor_sha256={str(path):sha(path) for path in predecessors}))
    groups={}
    for threshold in [0]+candidates:
        cfg=json.loads(config(a.base,a.profile,False).read_text())
        cfg.update(dynamic=threshold>0,threshold=threshold)
        cfgpath=a.output/f'config-{threshold}.json';write(cfgpath,cfg)
        root=a.output/f'process-{threshold}'
        with session(root,cfgpath):
            request(root,selected[0],f'warmup-{threshold}',8)
            for i,row in enumerate(selected):
                request(root,row,f'dev-threshold-{threshold}-{row["id"]}',64)
                write(a.output/'progress.json',dict(threshold=threshold,completed=i+1,total=12))
        raw=[json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines()]
        groups[threshold]={x['prompt_id']:x for x in raw if x['id'].startswith('dev-threshold-')}
        assert len(groups[threshold])==12
    results=[]
    for threshold in candidates:
        ratios=[groups[threshold][r['id']]['seconds']/groups[0][r['id']]['seconds'] for r in selected]
        per_bucket={bucket:statistics.median(groups[threshold][r['id']]['seconds']/groups[0][r['id']]['seconds']
            for r in selected if r['bucket']==bucket) for bucket in ('short','medium','long')}
        trigger_expected=sum(len(r['input_ids'])>=threshold for r in selected)
        results.append(dict(threshold=threshold,median_e2e_ratio=statistics.median(ratios),
            matched_case_ratios=ratios,per_bucket_e2e_ratio=per_bucket,
            expected_trigger_requests=trigger_expected,requests=12))
    winner=min(results,key=lambda x:(x['median_e2e_ratio'],-x['threshold']))
    write(a.output/'DYNAMIC_THRESHOLD_CANDIDATES.json',dict(status='DEV_SCREEN_COMPLETE',results=results))
    write(a.output/'DYNAMIC_THRESHOLD_SELECTED.json',dict(status='FROZEN_NOT_HELDOUT_QUALIFIED',
        threshold=winner['threshold'],selection=winner,dev_sha256=sha(a.dev),
        profile_sha256=sha(a.base/'profiles'/f'{a.profile}.json'),
        protocol_sha256=sha(a.output/'protocol.json'),
        candidate_results_sha256=sha(a.output/'DYNAMIC_THRESHOLD_CANDIDATES.json'),
        quality_used_for_selection=False,heldout_used_for_selection=False))


if __name__=='__main__':main()
