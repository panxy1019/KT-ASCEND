"""Only after all static scale gates, run observer-OFF all-MoE timing phases."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predecessor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    state=dict(status='WAIT_STATIC_MATRIX',completed=[])
    def save():(a.output/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
    tools=Path(__file__).parent;save()
    try:
        deadline=time.monotonic()+28800
        while True:
            try:data=json.loads(a.predecessor.read_text())
            except json.JSONDecodeError:time.sleep(1);continue
            if data['status']=='FAILED':raise ValueError('STATIC_PREDECESSOR_FAILED')
            if data['status']=='STATIC_SCALE_MATRIX_PASS_CAPACITY_PERFORMANCE_PENDING':break
            if time.monotonic()>deadline:raise TimeoutError('STATIC_PREDECESSOR_TIMEOUT')
            time.sleep(15)
        summaries={}
        for resident in (64,32,16,8):
            name=f'all-R{resident}'
            state.update(status='RUNNING',active=name);save()
            command=[sys.executable,str(tools/'run_static_performance.py'),'--profile',
                f'/workspace/results/r65-profiles-v1/{name}.json','--manifest',
                str(tools.parent/'fixtures/performance.json'),'--output',str(a.output/name)]
            with (a.output/(name+'.log')).open('x') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{name}:{result.returncode}')
            report=json.loads((a.output/name/'report.json').read_text())
            if report['status']!='COMPACT_PERFORMANCE_COMPLETE':raise ValueError('PERF_NOT_COMPLETE')
            summaries[name]=dict(summary=report['summary'],server_memory=report['server_after_load']['internal_states'][0]['memory_usage'])
            state['completed'].append(name);save()
        (a.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        state['status']='CAPACITY_TIMING_COMPLETE_ROUTING_AND_REVIEW_PENDING'
    except BaseException as exc:state.update(status='FAILED',exception=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
