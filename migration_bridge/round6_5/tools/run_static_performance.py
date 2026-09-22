"""Fixed KV capacity, observer-OFF compact static capacity/performance phase."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.request
from static_session import session


def memory():
    paths=['/proc/meminfo','/proc/vmstat','/sys/fs/cgroup/memory.current',
           '/sys/fs/cgroup/memory.stat','/sys/fs/cgroup/memory.events','/sys/fs/cgroup/memory.pressure']
    return dict(time_unix=time.time(),files={p:Path(p).read_text() for p in paths if Path(p).exists()})


def server_info():
    with urllib.request.urlopen('http://127.0.0.1:31000/server_info',timeout=20) as response:data=json.load(response)
    states=data['internal_states']
    if len(states)!=1 or states[0]['memory_usage']['token_capacity']!=4096:raise ValueError('KV_CAPACITY_DRIFT')
    if states[0]['memory_usage']['graph']!=0:raise ValueError('GRAPH_MEMORY_UNEXPECTED')
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    raw=a.manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='caa1cb3bd77cf0376c035c6a8b5ebd40800f6ad26f11cfc7408fc6d3fcf8ffd5':
        raise ValueError('PERFORMANCE_MANIFEST_DRIFT')
    manifest=json.loads(raw)
    legacy=Path('/workspace/results/r64-migration-tools-v6/tools')
    if hashlib.sha256((legacy/'stream_timing.py').read_bytes()).hexdigest()!='5d3e7fa077db07f70d830954cd95d1ad915bfd182ab0e2b68dfefca3c85a367e':
        raise ValueError('TIMING_IMPLEMENTATION_DRIFT')
    sys.path.insert(0,str(legacy))
    from stream_timing import request
    os.environ.pop('KT_R65_STAGEWISE',None)
    report=dict(status='RUNNING',manifest_sha256=hashlib.sha256(raw).hexdigest(),
        profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),rows=[],phases=[],
        memory_before_model=memory(),timing_scope='CLIENT_STREAMING_NOT_KERNEL_PROFILER',
        memory_weight_scope='SERVER_REPORTED_DEVICE_FREE_DELTA_NOT_EXACT_TENSOR_BYTES')
    try:
        with session(a.output,a.profile,capture=False) as root:
            report['server_after_load']=server_info();report['memory_after_load']=memory()
            with (root/'requests.jsonl').open('x') as log:
                for shape in manifest['shapes']:
                    reference=None
                    for i in range(5):
                        rid=f'r65-perf-{shape["id"]}-{i}'
                        row=request(rid,shape['input_ids'],64)
                        if row['response']['meta_info']['prompt_tokens']!=len(shape['input_ids']):raise ValueError('PERF_INPUT_COUNT')
                        ids=row['response']['output_ids']
                        if reference is None:reference=ids
                        if ids!=reference:raise ValueError('PERF_SAME_PATH_NONDETERMINISM')
                        log.write(json.dumps(dict(shape=shape['id'],warmup=i<2,**row))+'\n');log.flush()
                        report['rows'].append(dict(shape=shape['id'],warmup=i<2,
                            **{k:v for k,v in row.items() if k!='response'}))
                        if i==1:report['phases'].append(dict(shape=shape['id'],phase='AFTER_WARMUP',memory=memory(),server=server_info()))
                    report['phases'].append(dict(shape=shape['id'],phase='AFTER_TIMED',memory=memory(),server=server_info()))
                    (root/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
            if (root/'stagewise').exists() or list((root/'tensors').rglob('*.pt')):raise ValueError('PERF_OBSERVER_CONTAMINATION')
        report['summary']={shape['id']:{metric:statistics.median(r[metric] for r in report['rows']
            if r['shape']==shape['id'] and not r['warmup']) for metric in
            ('seconds','ttft_seconds','decode_tokens_per_second')} for shape in manifest['shapes']}
        report['status']='COMPACT_PERFORMANCE_COMPLETE'
    except BaseException as exc:report.update(status='FAILED',exception=repr(exc));raise
    finally:
        report['memory_after_stop']=memory()
        if a.output.exists():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
