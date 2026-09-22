"""After single R32 numerical gate only, run real-weight local safety controls."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('/workspace/results/r65-single-controls-v1')
    root.mkdir(exist_ok=False)
    tools=Path(__file__).parent
    predecessor=Path('/workspace/results/r65-single-entry-v1/progress.json')
    state=dict(status='WAIT_PREDECESSOR',completed=[])
    def save(): (root/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        deadline=time.monotonic()+7200
        while True:
            data=json.loads(predecessor.read_text())
            if data['status']=='FAILED':raise ValueError('PREDECESSOR_FAILED')
            if data['status']=='SINGLE_R32_ENTRY_COMPLETE_LOCAL_CONTROLS_PENDING':break
            if time.monotonic()>deadline:raise TimeoutError('PREDECESSOR_TIMEOUT')
            time.sleep(10)
        for tokens in (1,4):
            name=f'E1-T{tokens}'
            out=root/name
            command=[sys.executable,str(tools/'owned_guard.py'),'--output',str(out/'guard'),
                '--timeout-seconds','1800','--','bash','-lc',
                'source /usr/local/Ascend/ascend-toolkit/set_env.sh; '
                'source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh; '
                'source /usr/local/Ascend/nnal/atb/set_env.sh; '
                'export ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; '
                'export PYTHONPATH="/workspace/kt-src/third_party/sglang/python:/workspace/kt-src:${PYTHONPATH:-}"; '
                f'exec taskset -c 0-31 /workspace/results/venv-r6/bin/python {tools}/probe_heavy_expert.py '
                '--model-dir /workspace/models/DeepSeek-V2-Lite-604d5664 '
                '--gguf /workspace/results/r65-assets-v1/all-moe-f32.gguf '
                '--layer 17 --expert 1 --profile /workspace/results/r65-profiles-v1/single-R32.json '
                f'--num-tokens {tokens} --cpu-repeats 10 --cpuinfer-threads 32 --output {out}/result.json']
            state.update(status='RUNNING',active=name,command=command);save()
            with (root/(name+'.log')).open('x') as log:
                r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError(f'{name}_FAILED_{r.returncode}')
            result=json.loads((out/'result.json').read_text())
            if not result['cpu_not_hit_control_exact_zero'] or result['metrics']['relative_l2']>.01:
                raise ValueError('EXPERT_SAFETY')
            state['completed'].append(name);save()
        state['status']='LOCAL_REAL_WEIGHT_CONTROLS_PASS'
    except BaseException as exc:state.update(status='FAILED',exception=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
