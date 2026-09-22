"""Hook-free Static/Dynamic timing with the user's frozen short/medium inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import requests
from qwen_static_perf import make_inputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=['static','dynamic'],required=True)
    p.add_argument('--block',type=int,choices=range(3),required=True)
    p.add_argument('--position',type=int,choices=range(4),required=True);a=p.parse_args()
    root=Path('/workspace/round8');out=root/'qwen-dynamic-perf-v2'/f'{a.profile}-{a.block}-{a.position}'
    out.mkdir(exist_ok=False);inputs=make_inputs()
    protocol=dict(inputs=inputs,temperature=0,ignore_eos=True,max_running_requests=1,
        order=['short','medium'],diagnostic_hooks=False,graph=False,deferred=False)
    blob=json.dumps(protocol,sort_keys=True,indent=2)+'\n';(out/'protocol.json').write_text(blob)
    url='http://127.0.0.1:18400/generate'
    warm=requests.post(url,json=dict(input_ids=inputs['short']['input_ids'],
        sampling_params=dict(temperature=0,max_new_tokens=8,ignore_eos=True)),timeout=3600)
    warm.raise_for_status();assert len(warm.json()['output_ids'])==8
    results=[]
    with (out/'raw.jsonl').open('x') as raw:
        for name in ('short','medium'):
            item=inputs[name];payload=dict(input_ids=item['input_ids'],sampling_params=dict(
                temperature=0,max_new_tokens=item['decode'],ignore_eos=True))
            started=time.time();t=time.perf_counter();response=requests.post(url,json=payload,timeout=7200)
            elapsed=time.perf_counter()-t;data=response.json()
            raw.write(json.dumps(dict(workload=name,payload=payload,response=data,status=response.status_code,
                start_unix=started,end_unix=time.time(),client_seconds=elapsed))+'\n');raw.flush()
            response.raise_for_status();assert len(data['output_ids'])==item['decode']
            results.append(dict(workload=name,client_seconds=elapsed,
                server_e2e_seconds=data['meta_info'].get('e2e_latency'),
                ttft='NOT_SEPARATELY_MEASURED',decode_tps='NOT_INFERRED_FROM_E2E'))
    (out/'summary.json').write_text(json.dumps(dict(status='CELL_COLLECTED',profile=a.profile,
        block=a.block,position=a.position,protocol_sha256=hashlib.sha256(blob.encode()).hexdigest(),
        results=results),indent=2))

if __name__=='__main__':main()
