"""Frozen ABBA; requires a fresh host-issued exclusive-performance lease."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from common import sha,write
from serial_session import session,memory
sys.path.append('/workspace/r67-tools-v5/migration_bridge/round6_7/tools')
from run_serial_sweep import request

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--canary',type=Path,required=True);p.add_argument('--lease',type=Path,required=True)
a=p.parse_args()
assert json.loads(a.canary.read_text())['status']=='PASS','CANARY_PREDECESSOR'
manifest=Path('/workspace/c64-dev/performance-freeze-v1/perf_prompt_set.json')
f=json.loads(manifest.read_text())
assert f['blocks']==3 and f['order']==['C32','C64','C64','C32']
def lease():
    state=json.loads(a.lease.read_text())
    assert state['status']=='EXCLUSIVE' and time.time()-state['time_unix']<30,'HOST_BANDWIDTH_NOT_EXCLUSIVE'
    assert state['container']=='kt-r67-c64-development-v1'
    return state
lease()
a.output.mkdir(parents=True,exist_ok=False)
report=dict(status='RUNNING',rows=[],manifest_sha256=sha(manifest),canary_sha256=sha(a.canary),
            started=time.time(),lease_initial=lease())
references={}
try:
    for block in range(3):
        for position,mode in enumerate(f['order']):
            lease()
            os.environ.update(C64_TP='2' if mode=='C64' else '1',KT_R67_NPU_ID='0',
                              KT_R67_CPU_SET='128-191' if mode=='C64' else '128-159')
            for key in ('KT_C64_CAPTURE_DIR','KT_MOE_PHASE_TIMING'):os.environ.pop(key,None)
            with session(a.output/f'b{block}-{position}-{mode}',a.canary.parent/'R16.json') as root:
                lease()
                request(root,f['rows'][-1],f'warm-{block}-{position}',1)
                for row in f['rows']:
                    lease();before=memory()
                    _,ids,seconds=request(root,row,f'perf-{block}-{position}-{row["id"]}',1)
                    lease()
                    key=(mode,row['id'])
                    if key in references:assert ids==references[key],'PERFORMANCE_NONDETERMINISM'
                    references[key]=ids
                    report['rows'].append(dict(block=block,position=position,mode=mode,shape=row['id'],
                        seconds=seconds,input_tokens=len(row['input_ids']),input_tokens_per_second=len(row['input_ids'])/seconds,
                        output_ids=ids,memory_before=before,memory_after=memory()))
                    write(a.output/'progress.json',report)
    pairs=[]
    for block in range(3):
        for row in f['rows']:
            lat={mode:statistics.median(x['seconds'] for x in report['rows']
                if x['block']==block and x['mode']==mode and x['shape']==row['id']) for mode in ('C32','C64')}
            pairs.append(dict(block=block,shape=row['id'],**lat,ratio=lat['C64']/lat['C32']))
    ratio=statistics.median(x['ratio'] for x in pairs)
    rng=random.Random(67064)
    boot=sorted(statistics.median(rng.choice(pairs)['ratio'] for _ in pairs) for _ in range(20000))
    report.update(status='PASS',paired_shapes=pairs,median_latency_ratio=ratio,
        paired_shape_bootstrap_95ci=[boot[499],boot[19499]],
        verdict='BENEFICIAL' if ratio<=.85 else ('MARGINAL' if ratio<=.90 else 'NO_MEANINGFUL_SPEEDUP'),
        statistical_scope='PAIRED_SHAPES_NOT_INDEPENDENT_HOST_REPLICATIONS',
        phase_breakdown='NOT_IN_PRIMARY_TIMING_REQUIRES_SEPARATE_DIAGNOSTIC',host_bandwidth_gbps=None)
except BaseException as exc:
    report.update(status='FAIL',error=repr(exc));raise
finally:
    report['ended']=time.time();write(a.output/'progress.json',report);write(a.output/'summary.json',report)
