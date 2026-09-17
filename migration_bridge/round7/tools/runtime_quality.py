"""Dynamic/Deferred use the same immutable CMMLU evaluator, no sample fitting."""
import argparse
import json
from pathlib import Path


def main():
    from static_qualification import config,sha,write,quality
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--profile',choices=['uniform','frequency'],required=True)
    p.add_argument('--mode',choices=['dynamic','deferred'],required=True)
    p.add_argument('--dynamic-base',action='store_true')
    p.add_argument('--threshold-freeze',type=Path)
    p.add_argument('--threshold-ack',type=Path)
    p.add_argument('--system-summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert json.loads(a.system_summary.read_text())['status']=='PASS','RUNTIME_SYSTEM_PREDECESSOR'
    assert json.loads((a.base/'operator-representatives/summary.json').read_text())['status']=='PASS'
    cfg=json.loads(config(a.base,a.profile,False).read_text())
    protocol=dict(mode=a.mode,profile=a.profile,manifest_sha256='bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04',
        reference_correct=174,count=256,minimum_correct=162,invalid_required=0,
        system_predecessor_sha256=sha(a.system_summary))
    if a.mode=='dynamic' or a.dynamic_base:
        from frozen_threshold import load_threshold
        frozen,ack=load_threshold(a.threshold_freeze,a.threshold_ack,a.base/'profiles'/f'{a.profile}.json')
        cfg.update(dynamic=True,threshold=frozen['threshold'])
        protocol.update(threshold=frozen['threshold'],threshold_freeze_sha256=sha(a.threshold_freeze),
            threshold_commit_ack=ack)
    if a.mode=='deferred':cfg['deferred']=1
    if cfg['dynamic']:cfg['dynamic_telemetry']=True
    cfgpath=a.output/'config.json';write(cfgpath,cfg);write(a.output/'protocol.json',protocol)
    result=quality(a.base,a.profile,cfgpath,a.output/'process',a.output/'summary.json')
    result.update(mode=a.mode,dynamic=cfg['dynamic'],deferred=cfg['deferred'],protocol=protocol)
    if cfg['dynamic']:
        from dynamic_probe import lines
        from dynamic_route_summary import correlate_updates
        raw=lines(a.output/'process/raw.jsonl')
        windows=[dict(id=x['response']['meta_info']['id'],started_unix=x['started_unix'],
                      ended_unix=x['ended_unix']) for x in raw]
        events=lines(a.output/'process/dynamic-telemetry.jsonl')
        result['dynamic_migration']=correlate_updates(events,windows)
        result['cmmlu_dynamic_trigger_rate']=result['dynamic_migration']['actual_trigger_rate']
        result['telemetry_scope']='UNTIMED_QUALITY_NOT_PERFORMANCE_OR_FULL_WEIGHT_CERTIFICATION'
    write(a.output/'summary.json',result)


if __name__=='__main__':main()
