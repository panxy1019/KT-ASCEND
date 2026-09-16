"""Read-only actual-deployment code/launch audit, distinct from runtime proof."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    src=Path('/workspace/kt-src');installed=Path('/workspace/results/venv-r6/lib/python3.11/site-packages/kt_kernel')
    files={
        installed/'utils/llamafile.py':['output_dtype','bf16_numerical_compat','threadpool_count'],
        installed/'experts_base.py':['subpool_count','subpool_numa_map','subpool_thread_count','copy_forward_output_to_device'],
        src/'kt-kernel/operators/llamafile/moe.hpp':['round_to_bf16_value','bf16_numerical_compat'],
        src/'third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py':['routed_scaling_factor=None','_kt_return_fp32_contribution','output = output + cpu_output'],
        src/'third_party/sglang/python/sglang/srt/layers/quantization/unquant.py':['topk_weights.to(x.dtype)','finalize_hidden_states','npu_moe_finalize_routing'],
        src/'third_party/sglang/python/sglang/srt/models/deepseek_v2.py':['final_hidden_states *= self.routed_scaling_factor','final_hidden_states += shared_output'],
    }
    rows=[]
    for path,needles in files.items():
        lines=path.read_text().splitlines()
        hits=[i for i,line in enumerate(lines) if any(n in line for n in needles)]
        rows.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            evidence_state='CODE_INSPECTED',snippets=[dict(line=i+1,text='\n'.join(lines[max(0,i-2):i+4])) for i in hits]))
    command=json.loads(Path('/workspace/results/r65-5a-bisection-v2/C3/resolved-command.json').read_text())
    argv=command['argv']
    def flag(name):return argv[argv.index(name)+1]
    if flag('--kt-cpuinfer')!='32' or flag('--kt-threadpool-count')!='1' or flag('--kt-numa-nodes')!='0':
        raise ValueError('TOPOLOGY_DRIFT')
    if flag('--kt-max-deferred-experts-per-token')!='0':raise ValueError('DEFERRED_DRIFT')
    cfg=json.loads(Path('/workspace/models/DeepSeek-V2-Lite-604d5664/config.json').read_text())
    result=dict(status='DEPLOYED_ARITHMETIC_SOURCE_AUDITED',files=rows,workers=32,threadpool_count=1,
        subpool_count=1,subpool_numa_map=[0],routed_scaling_factor=cfg['routed_scaling_factor'],
        topology_scope='LAUNCH_AND_WORKER_CONFIG_SOURCE_NOT_INSTRUCTION_TRACE',
        scaling_limitation='factor=1; multiplicity established by source, not sensitivity of outputs',
        reduction_tree_root_cause_proven=False)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':main()
