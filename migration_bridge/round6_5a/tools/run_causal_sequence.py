"""Bounded WP4->WP5->WP6 after the completed WP3 capture; no promotion."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--bisection',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    here=Path(__file__).parent;old=Path('/workspace/r65-tools/migration_bridge/round6_5/tools')
    trace=Path('/workspace/results/r65-four-R32-localization-v1')
    state=dict(status='WAIT_WP3',completed=[],scope='DIAGNOSTIC_ONLY_NO_PROMOTION')
    def save():(a.output/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
    def run(name,cmd,env=None):
        state.update(status='RUNNING',active=name);save()
        with (a.output/(name+'.log')).open('x') as f:
            subprocess.run(list(map(str,cmd)),env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        state['completed'].append(name);save()
    save()
    try:
        deadline=time.monotonic()+1200
        while True:
            status=json.loads((a.bisection/'progress.json').read_text())['status']
            if status=='BISECTION_CAPTURE_COMPLETE_REVIEW_REQUIRED':break
            if status=='FAILED':raise RuntimeError('WP3_FAILED')
            if time.monotonic()>deadline:raise TimeoutError('WP3_WAIT')
            time.sleep(10)
        command=[sys.executable,here/'replay_router.py','--trace',trace,
            '--model','/workspace/models/DeepSeek-V2-Lite-604d5664','--output',a.output/'router_boundary.json']
        shell='source /usr/local/Ascend/ascend-toolkit/set_env.sh; '
        shell+='source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh; '
        shell+='source /usr/local/Ascend/nnal/atb/set_env.sh; '
        shell+='export ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; '
        shell+='export PYTHONPATH="/workspace/kt-src/third_party/sglang/python:/workspace/kt-src:${PYTHONPATH:-}"; '
        shell+='exec taskset -c 0-31 '+shlex.join(list(map(str,command)))
        run('router',[sys.executable,old/'owned_guard.py','--output',a.output/'router-guard',
                      '--timeout-seconds','600','--','bash','-c',shell])
        router=json.loads((a.output/'router_boundary.json').read_text())
        if router['status']!='ROUTER_REPLAY_EXACT':raise RuntimeError('ROUTER_REPLAY_NOT_EXACT')
        profiles=dict(C1='/workspace/results/r65-5a-bisection-v1/C1',C2=str(a.bisection/'C2'),
                      C3=str(a.bisection/'C3'),C4=str(trace/'hybrid'))
        for mode in ('baseline_routes','npu_for_cpu'):
            root=a.output/mode
            env=dict(os.environ,R65A_INTERVENTION_MODE=mode,R65A_INTERVENTION_ROOT=str(a.output/(mode+'-interventions')))
            run(mode,[sys.executable,here/'capture_counterfactual.py','--inputs',trace/'inputs.json',
                '--force',trace/'force.json','--profile','/workspace/results/r65-profiles-v1/four-R32.json',
                '--output',root],env)
            run(mode+'-compare',[sys.executable,here/'compare_diagnostic.py',
                '--baseline','/workspace/results/r65-numeric-r64-v1','--hybrid',root,
                '--output',a.output/(mode+'-pairwise.json')])
            interventions=[json.loads(s) for s in (a.output/(mode+'-interventions')/'interventions.jsonl').read_text().splitlines()]
            expected_layers=range(1,27) if mode=='baseline_routes' else (1,9,17,26)
            actual={(r['layer'],r['call']) for r in interventions}
            expected={(layer,call) for layer in expected_layers for call in range(1,66)}
            if len(actual)!=len(interventions) or actual!=expected:raise ValueError('INTERVENTION_COVERAGE')
            profiles[mode]=str(root)
        profile_file=a.output/'diagnostic_profiles.json';profile_file.write_text(json.dumps(profiles,indent=2)+'\n')
        run('layer-analysis',[sys.executable,here/'summarize_causal_profiles.py','--profiles',profile_file,
                              '--output',a.output/'counterfactual_replay.json'])
        state['status']='CAUSAL_EXPERIMENTS_COMPLETE_REVIEW_REQUIRED'
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
