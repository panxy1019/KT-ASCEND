"""Frozen single -> four -> all static gates, stop on first failure.

No performance, frequency or Deferred promotion is implicit in this queue.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--predecessor',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    tools=Path(__file__).parent;results=Path('/workspace/results')
    state=dict(status='WAIT_PREDECESSOR',completed=[],started_unix=time.time())
    def save():(a.output/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
    def run(name,args):
        mem={line.split(':')[0]:int(line.split()[1])*1024
             for line in Path('/proc/meminfo').read_text().splitlines() if line.endswith(' kB')}
        if mem['MemAvailable']<max(32*1024**3,mem['MemTotal']*.10):
            raise RuntimeError('HOST_CAPACITY_RESERVE')
        state.update(status='RUNNING',active=name,command=list(map(str,args)));save()
        with (a.output/(name+'.log')).open('x') as log:
            r=subprocess.run([sys.executable,*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{name}:exit={r.returncode}')
        state['completed'].append(name);save()
    def expert(profile,route,name,layer,tokens):
        out=a.output/f'{name}-L{layer}-T{tokens}'
        probe=[sys.executable,tools/'probe_heavy_expert.py',
            '--model-dir',results.parent/'models/DeepSeek-V2-Lite-604d5664',
            '--gguf',results/'r65-assets-v1/all-moe-f32.gguf','--layer',str(layer),
            '--expert','1','--profile',profile,'--num-tokens',str(tokens),
            '--cpu-repeats','10','--cpuinfer-threads','32','--output',out/'result.json',
            '--hidden-file',route/f'capture/stagewise/cpu_layer{layer:02d}_input_hidden_after_sync_call0001.pt']
        shell='set -e; source /usr/local/Ascend/ascend-toolkit/set_env.sh; '
        shell+='source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh; '
        shell+='source /usr/local/Ascend/nnal/atb/set_env.sh; '
        shell+='export ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; '
        shell+='export PYTHONPATH="/workspace/kt-src/third_party/sglang/python:/workspace/kt-src:${PYTHONPATH:-}"; '
        shell+='exec taskset -c 0-31 '+shlex.join(map(str,probe))
        run(f'{name}-expert-L{layer}-T{tokens}',[tools/'owned_guard.py','--output',out/'guard',
            '--timeout-seconds','1800','--','bash','-lc',shell])
    save()
    try:
        deadline=time.monotonic()+7200
        while True:
            try:data=json.loads(a.predecessor.read_text())
            except json.JSONDecodeError:
                if time.monotonic()>deadline:raise TimeoutError('PREDECESSOR_PARTIAL_JSON')
                time.sleep(1);continue
            if data['status']=='FAILED':raise ValueError('ROUTE_PREDECESSOR_FAILED')
            if data['status']=='STATIC_ROUTE_CONTROL_PASS':break
            if time.monotonic()>deadline:raise TimeoutError('PREDECESSOR_TIMEOUT')
            time.sleep(10)
        initial=results/'r65-profiles-v1/single-R32.json'
        for t in (1,4):expert(initial,a.predecessor.parent,'single-R32',17,t)
        names=['single-R16','single-R8']+[f'{scope}-R{r}' for scope in ('four','all') for r in (32,16,8)]
        for name in names:
            profile=results/'r65-profiles-v1'/f'{name}.json'
            root=a.output/name;root.mkdir()
            run(name+'-det',[tools/'run_static_determinism.py','--corpus',results/'corpora-v2/s6.json',
                '--profile',profile,'--output',root/'det'])
            run(name+'-numeric',[tools/'capture_static_logits.py','--inputs',tools.parent/'fixtures/numerical.json',
                '--profile',profile,'--force',results/'r65-numeric-r64-v1/force.json','--output',root/'numeric'])
            run(name+'-pairwise',[tools/'evaluate_static_pairwise.py','--baseline',results/'r65-numeric-r64-v1',
                '--hybrid',root/'numeric','--output',root/'pairwise.json'])
            run(name+'-routes',[tools/'run_route_control_gate.py','--profile',profile,
                '--determinism',root/'det/report.json','--output',root/'routes'])
            for layer in json.loads(profile.read_text())['selected_layers']:
                expert(profile,root/'routes',name,layer,4)
            expert(profile,root/'routes',name,17,1)
            run(name+'-guard-audit',[tools/'audit_static_lifecycle.py','--root',root,'--output',root/'guard-audit.json'])
            (root/'gate.json').write_text(json.dumps(dict(status='COMPACT_STATIC_GATES_PASS',profile=name))+'\n')
        state['status']='STATIC_SCALE_MATRIX_PASS_CAPACITY_PERFORMANCE_PENDING'
    except BaseException as exc:state.update(status='FAILED',exception=repr(exc));raise
    finally:state['ended_unix']=time.time();save()


if __name__=='__main__':main()
