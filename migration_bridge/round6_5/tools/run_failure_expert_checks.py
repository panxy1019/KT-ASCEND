"""Bounded same-operand experts implicated by the first-failure trace."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    audit=json.loads((a.trace/'transfer-audit.json').read_text())
    if audit['status']!='FAILURE_TRANSFER_AND_MERGE_AUDIT_PASS':raise ValueError('TRANSFER_PREDECESSOR')
    tests=[(r['layer'],r['first_routed_cpu_expert_at_failure']) for r in audit['rows']]
    tests.append((17,29)) # first changed router member, diagnostic only
    a.output.mkdir(parents=True,exist_ok=False);tools=Path(__file__).parent
    report=dict(status='RUNNING',scope='FIRST_FAILURE_SAME_OPERAND_DIAGNOSTIC',rows=[])
    try:
        for layer,expert in tests:
            root=a.output/f'L{layer}-E{expert}'
            command=[sys.executable,tools/'probe_heavy_expert.py','--model-dir','/workspace/models/DeepSeek-V2-Lite-604d5664',
                '--gguf','/workspace/results/r65-assets-v1/all-moe-f32.gguf','--profile','/workspace/results/r65-profiles-v1/four-R32.json',
                '--layer',str(layer),'--expert',str(expert),'--num-tokens','1','--cpu-repeats','10','--cpuinfer-threads','32',
                '--hidden-file',a.trace/f'hybrid/stagewise/cpu_layer{layer:02d}_input_hidden_after_sync_call0055.pt',
                '--output',root/'result.json']
            shell='set -e; source /usr/local/Ascend/ascend-toolkit/set_env.sh; '
            shell+='source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh; '
            shell+='source /usr/local/Ascend/nnal/atb/set_env.sh; '
            shell+='export ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; '
            shell+='export PYTHONPATH="/workspace/kt-src/third_party/sglang/python:/workspace/kt-src:${PYTHONPATH:-}"; '
            shell+='exec taskset -c 0-31 '+shlex.join(map(str,command))
            report['active']=dict(layer=layer,expert=expert)
            (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
            with (a.output/f'L{layer}-E{expert}.log').open('x') as f:
                code=subprocess.run([sys.executable,str(tools/'owned_guard.py'),'--output',str(root/'guard'),
                    '--timeout-seconds','600','--','bash','-lc',shell],stdout=f,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError(f'EXPERT_CHECK:{layer}:{expert}:{code}')
            report['rows'].append(json.loads((root/'result.json').read_text()))
        report['status']='IMPLICATED_EXPERT_SAFETY_PASS'
    except BaseException as exc:report.update(status='FAILED',exception=repr(exc));raise
    finally:(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
